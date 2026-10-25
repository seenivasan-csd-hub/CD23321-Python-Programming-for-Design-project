"""Train and evaluate FarmIQ's optional PlantDoc MobileNetV3 classifier."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import uuid

import numpy as np
from PIL import Image

from plantdoc_data import collect_dataset, count_rows, validate_for_training, validation_split, write_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "plantdoc" / "raw"
MODEL_ROOT = PROJECT_ROOT / "models" / "plantdoc"
REPORT_PATH = PROJECT_ROOT / "data" / "plantdoc" / "dataset-report.json"
IMAGE_SIZE = 224
RESIZE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_CARD_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
WEIGHT_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"


class PlantDocImages:
    def __init__(self, records, class_to_index, transform):
        self.records = records
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            with Image.open(record.path) as opened:
                image = opened.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Image became unreadable after dataset validation: {record.path}") from exc
        return tensor, self.class_to_index[record.label]


def _imports():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        import onnx  # noqa: F401
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install CPU PyTorch/torchvision from the official "
            "CPU wheel index, then install requirements-training.txt."
        ) from exc
    return torch, nn, DataLoader, transforms, MobileNet_V3_Small_Weights, mobilenet_v3_small


def _seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _metrics(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    per_class = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "support": total,
    }


def _softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("Calibration logits must be a finite two-dimensional array.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("Calibration temperature must be finite and positive.")
    shifted = logits / float(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _calibration_metrics(y_true, probabilities, bins=15):
    targets = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Calibration requires matching non-empty targets and class probabilities.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Calibration probabilities must be finite and non-negative.")
    sums = probabilities.sum(axis=1)
    if np.any(sums <= 0):
        raise ValueError("Each calibration probability row must have positive total mass.")
    probabilities = probabilities / sums[:, None]
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("Calibration targets do not match the class count.")
    rows = np.arange(len(targets))
    chosen = probabilities.argmax(axis=1)
    confidence = probabilities[rows, chosen]
    correct = chosen == targets
    nll = -np.log(np.clip(probabilities[rows, targets], 1e-15, 1.0)).mean()
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {"nll": float(nll), "brier_score": float(brier), "expected_calibration_error": float(ece)}


def _fit_temperature(logits, y_true):
    """Fit one scalar temperature on validation logits by minimizing multiclass NLL."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(y_true, dtype=np.int64)
    if logits.ndim != 2 or len(targets) != len(logits) or not len(targets) or not np.isfinite(logits).all():
        raise ValueError("Temperature fitting requires matching finite validation logits and targets.")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("Temperature fitting targets do not match the class count.")
    row_indices = np.arange(len(targets))

    def nll(log_temperature):
        scaled = logits / math.exp(float(log_temperature))
        maxima = scaled.max(axis=1)
        log_partition = maxima + np.log(np.exp(scaled - maxima[:, None]).sum(axis=1))
        return float(np.mean(log_partition - scaled[row_indices, targets]))

    # Golden-section search over a broad positive range avoids a dependency on SciPy.
    lower, upper = math.log(0.02), math.log(50.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value, right_value = nll(left), nll(right)
    for _ in range(100):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = nll(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = nll(right)
    return float(math.exp((lower + upper) / 2.0))


def _select_rejection_threshold(probabilities, y_true, target_accuracy=0.90, minimum_coverage=0.10):
    """Choose the lowest validation score cut meeting an explicit selective-accuracy policy."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(y_true, dtype=np.int64)
    if probabilities.ndim != 2 or len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Threshold selection requires matching non-empty validation data.")
    if not 0 < target_accuracy <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("Accuracy and coverage policy values must be in (0, 1].")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    required = max(1, int(math.ceil(len(targets) * minimum_coverage)))
    for threshold in sorted(set(float(value) for value in confidence)):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        if count >= required and float(correct[accepted].mean()) >= target_accuracy:
            return {
                "threshold": threshold,
                "validation_coverage": count / len(targets),
                "validation_selective_accuracy": float(correct[accepted].mean()),
                "accepted_count": count,
                "policy": {"target_selective_accuracy": target_accuracy, "minimum_coverage": minimum_coverage},
                "target_met": True,
            }
    return {
        "threshold": 1.01,
        "validation_coverage": 0.0,
        "validation_selective_accuracy": None,
        "accepted_count": 0,
        "policy": {"target_selective_accuracy": target_accuracy, "minimum_coverage": minimum_coverage},
        "target_met": False,
    }


def _selective_metrics(probabilities, y_true, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(y_true, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1) >= float(threshold)
    count = int(accepted.sum())
    return {
        "accepted_count": count,
        "coverage": float(count / len(targets)) if len(targets) else 0.0,
        "accuracy": float((predictions[accepted] == targets[accepted]).mean()) if count else None,
    }


def _collect_logits(model, loader, device):
    import torch

    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            batch_logits = model(images.to(device, non_blocking=device.type == "cuda"))
            logits.append(batch_logits.detach().cpu().numpy())
            targets.extend(labels.cpu().tolist())
    if not logits:
        raise ValueError("The validation or test loader contains no images.")
    return np.concatenate(logits, axis=0), np.asarray(targets, dtype=np.int64)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    predictions, targets = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_size
            total_count += batch_size
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
    return (total_loss / max(1, total_count), predictions, targets)


def _atomic_json(path: Path, content: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _plant_by_class(labels):
    plants = [
        "bell_pepper", "blueberry", "strawberry", "soyabean", "raspberry",
        "tomato", "potato", "squash", "cherry", "apple", "grape", "peach", "corn",
    ]
    result = {}
    for label in labels:
        folded = label.casefold().replace(" ", "_")
        match = next((name for name in plants if folded.startswith(name)), None)
        result[label] = match.replace("_", " ").title() if match else None
    return result


def _write_card(path: Path, metadata: dict, metrics: dict):
    lines = [
        "# FarmIQ PlantDoc image classifier",
        "",
        f"- Run: `{metadata['model_version']}`",
        f"- Architecture: `{metadata['architecture']}` (MobileNetV3 Small)",
        f"- PyTorch / TorchVision: `{metadata['torch_version']}` / `{metadata['torchvision_version']}`",
        f"- PlantDoc source commit: `{metadata['dataset']['commit'] or 'not recorded'}`",
        f"- Transfer-learning weights: [TorchVision MobileNetV3 Small ImageNet-1K V1]({WEIGHT_URL})",
        "- Pretraining terms: ImageNet dataset terms limit use to non-commercial research and educational purposes; review the current terms before use.",
        f"- Preprocessing: RGB, resize shortest edge to {RESIZE_SIZE}, center crop {IMAGE_SIZE}, ImageNet mean/std normalization.",
        f"- Confidence: validation temperature-scaled score; warning threshold {metadata['low_confidence_threshold']:.4f}.",
        f"- Validation calibration: temperature {metadata['temperature']:.4f}; selective-accuracy target met: {metadata['rejection_policy']['target_met']}.",
        "",
        "## Intended use and limits",
        "",
        "Research/demo triage for the PlantDoc folder classes listed below. A predicted class is not a diagnosis, "
        "does not establish cause or severity, and is not validated for Tamil Nadu field use. PlantDoc images "
        "are web-collected and may contain label noise. The classifier does not reliably detect out-of-distribution "
        "photos, so an in-range score is not proof that an image belongs to a supported class.",
        "",
        "Temperature scaling and the warning threshold are fitted on a held-out validation subset; this is not "
        "field validation or an out-of-distribution detector. Do not select pesticides from this prediction. Confirm uncertain or severe symptoms with a local "
        "agricultural extension professional. Guidance shown by FarmIQ is separate and linked to its source.",
        "",
        "## PlantDoc class labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in metadata["classes"])
    lines.extend([
        "",
        "## Official PlantDoc test split evaluation",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro precision: {metrics['macro_precision']:.4f}",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
        f"- Test calibration ECE: {metrics['calibration']['expected_calibration_error']:.4f}",
        f"- Test images: {metrics['support']}",
        "",
        "Per-class precision, recall, F1, and the confusion matrix are in `metrics.json` and `confusion_matrix.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args) -> Path:
    torch, nn, DataLoader, transforms, weights_enum, model_builder = _imports()
    data_root = Path(args.data_root).expanduser().resolve()
    records, report = collect_dataset(data_root)
    labels = validate_for_training(records, report)
    train_records, val_records = validation_split(records["train"], args.validation_fraction, args.seed)
    if not train_records or not val_records:
        raise ValueError("A training/validation split could not be made; each usable class needs enough unique images.")
    class_to_index = {label: index for index, label in enumerate(labels)}
    train_counts = count_rows(train_records)
    val_counts = count_rows(val_records)
    test_counts = count_rows(records["test"])
    print(f"Classes: {len(labels)} | train: {len(train_records)} | validation: {len(val_records)} | official test: {len(records['test'])}")
    print(f"CPU cores visible to Python: {os.cpu_count() or 1}")

    _seed_everything(torch, args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    cpu_threads = args.cpu_threads or max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(cpu_threads)
    print(f"Training device: {device}; PyTorch threads: {cpu_threads}")

    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.78, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        PlantDocImages(train_records, class_to_index, train_transform),
        shuffle=True, generator=generator, **loader_options,
    )
    validation_loader = DataLoader(
        PlantDocImages(val_records, class_to_index, evaluation_transform), shuffle=False, **loader_options,
    )
    # The official test loader is constructed for the final evaluation only, after early stopping.
    test_loader = DataLoader(
        PlantDocImages(records["test"], class_to_index, evaluation_transform), shuffle=False, **loader_options,
    )

    weights = weights_enum.IMAGENET1K_V1
    try:
        model = model_builder(weights=weights, progress=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the pinned TorchVision ImageNet weights. Training did not start; "
            "check network access and the TorchVision model cache."
        ) from exc
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(labels))
    model.to(device)

    frequencies = np.array([train_counts[label] for label in labels], dtype=np.float64)
    class_weights = np.sqrt(frequencies.sum() / (len(labels) * frequencies))
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    source_path = data_root / "plantdoc-source.json"
    dataset_source = {}
    if source_path.is_file():
        dataset_source = json.loads(source_path.read_text(encoding="utf-8"))
    report["training_classes"] = labels
    report["derived_split_counts"] = {"train": train_counts, "validation": val_counts, "official_test": test_counts}
    report["validation_fraction"] = args.validation_fraction
    report["seed"] = args.seed
    write_report(REPORT_PATH, report)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    staging_root = MODEL_ROOT / "runs"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".training-", dir=staging_root))
    final_run = staging_root / f"plantdoc-{run_id}"
    try:
        checkpoint_path = staging / "best_checkpoint.pth"
        history_path = staging / "epoch_metrics.csv"
        history = []
        per_class_history_path = staging / "validation_per_class.jsonl"
        best_val_loss = math.inf
        best_epoch = 0
        stale_epochs = 0
        with history_path.open("w", newline="", encoding="utf-8") as handle, per_class_history_path.open("w", encoding="utf-8") as per_class_handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1"])
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
                validation_loss, val_predictions, val_targets = _run_epoch(model, validation_loader, criterion, device)
                val_stats = _metrics(val_targets, val_predictions, labels)
                row = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "validation_loss": round(validation_loss, 6),
                    "validation_accuracy": round(val_stats["accuracy"], 6),
                    "validation_macro_f1": round(val_stats["macro_f1"], 6),
                }
                history.append(row)
                writer.writerow(row)
                handle.flush()
                per_class_handle.write(json.dumps({"epoch": epoch, "per_class": val_stats["per_class"]}, ensure_ascii=False) + "\n")
                per_class_handle.flush()
                print(
                    f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
                    f"val_loss={validation_loss:.4f} val_accuracy={val_stats['accuracy']:.4f} "
                    f"val_macro_f1={val_stats['macro_f1']:.4f}"
                )
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({
                        "state_dict": model.state_dict(),
                        "classes": labels,
                        "best_epoch": best_epoch,
                        "validation_loss": best_val_loss,
                        "seed": args.seed,
                    }, checkpoint_path)
                else:
                    stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {epoch} epochs; best validation loss was at epoch {best_epoch}.")
                    break

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # The untouched repository test split is evaluated once, after model selection.
        validation_logits, validation_targets = _collect_logits(model, validation_loader, device)
        temperature = _fit_temperature(validation_logits, validation_targets)
        validation_probabilities = _softmax(validation_logits, temperature)
        rejection_policy = _select_rejection_threshold(
            validation_probabilities,
            validation_targets,
            target_accuracy=args.target_selective_accuracy,
            minimum_coverage=args.minimum_coverage,
        )
        validation_calibration = _calibration_metrics(validation_targets, validation_probabilities)

        test_loss, test_predictions, test_targets = _run_epoch(model, test_loader, criterion, device)
        test_logits, test_targets_from_logits = _collect_logits(model, test_loader, device)
        if not np.array_equal(np.asarray(test_targets), test_targets_from_logits):
            raise RuntimeError("Test loader target order changed during final evaluation.")
        test_probabilities = _softmax(test_logits, temperature)
        test_metrics = _metrics(test_targets, test_predictions, labels)
        test_metrics["loss"] = test_loss
        test_metrics["evaluation_split"] = "official PlantDoc test"
        test_metrics["test_split_modified"] = False
        test_metrics["calibration"] = _calibration_metrics(test_targets, test_probabilities)
        test_metrics["validation_selected_rejection_policy"] = rejection_policy
        test_metrics["selective_test_metrics"] = _selective_metrics(
            test_probabilities, test_targets, rejection_policy["threshold"]
        )
        (staging / "metrics.json").write_text(
            json.dumps({
                "validation_history": history,
                "validation_calibration": validation_calibration,
                "validation_rejection_policy": rejection_policy,
                "test": test_metrics,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual\\predicted", *labels])
            writer.writerows([[label, *row] for label, row in zip(labels, test_metrics["confusion_matrix"])])

        classes_path = staging / "classes.json"
        classes_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        source_commit = dataset_source.get("commit", "")
        metadata = {
            "model_name": "FarmIQ PlantDoc MobileNetV3 Small",
            "model_version": f"plantdoc@{source_commit[:12]}-{run_id}" if source_commit else f"plantdoc-{run_id}",
            "architecture": "mobilenet_v3_small",
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "classes": labels,
            "plant_by_class": _plant_by_class(labels),
            "input_size": IMAGE_SIZE,
            "resize_size": RESIZE_SIZE,
            "input_layout": "NCHW",
            "color_mode": "RGB",
            "mean": MEAN,
            "std": STD,
            "confidence_type": "temperature_scaled_validation_score",
            "calibration": "temperature_scaling_on_validation",
            "temperature": temperature,
            "calibration_metrics_validation": validation_calibration,
            "rejection_policy": rejection_policy,
            "low_confidence_threshold": rejection_policy["threshold"],
            "dataset": {
                "repository": dataset_source.get("repository", MODEL_CARD_URL),
                "revision": dataset_source.get("revision"),
                "commit": source_commit,
                "archive_sha256": dataset_source.get("archive_sha256"),
                "license": "CC BY 4.0 as stated by the official PlantDoc repository",
                "train_images": len(train_records),
                "validation_images": len(val_records),
                "official_test_images": len(records["test"]),
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
            },
            "pretrained_weights": {
                "name": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
                "url": WEIGHT_URL,
                "terms_url": "https://image-net.org/accessagreement",
                "license_notice": "ImageNet terms limit use to non-commercial research and educational purposes; verify permissions for the intended use.",
                "downloaded_only_during_training": True,
            },
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "confidence_note": "Temperature-scaled scores are not field probabilities; validation calibration does not detect all out-of-domain photos.",
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is required for the Flask inference service; install requirements-disease.txt.") from exc
        dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
        onnx_path = staging / "model.onnx"
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                dynamo=False,
            )
            inference_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            dummy_numpy = dummy.cpu().numpy()
            smoke = inference_session.run(None, {inference_session.get_inputs()[0].name: dummy_numpy})[0]
            if smoke.shape != (1, len(labels)) or not np.isfinite(smoke).all():
                raise RuntimeError("Exported ONNX output did not match the saved class labels.")
            with torch.inference_mode():
                pytorch_smoke = model(dummy).detach().cpu().numpy()
            if not np.allclose(pytorch_smoke, smoke, rtol=1e-3, atol=1e-4):
                max_difference = float(np.max(np.abs(pytorch_smoke - smoke)))
                raise RuntimeError(f"PyTorch and ONNX logits differ beyond tolerance (max absolute difference {max_difference:.6g}).")
        except Exception as exc:
            raise RuntimeError(f"ONNX export or inference verification failed: {exc}") from exc

        _write_card(staging / "model_card.md", metadata, test_metrics)
        staging.rename(final_run)
        active = {
            "run": final_run.relative_to(MODEL_ROOT).as_posix(),
            "model_version": metadata["model_version"],
        }
        if args.activate:
            _atomic_json(MODEL_ROOT / "active.json", active)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Official test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Official test macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"Official test macro recall: {test_metrics['macro_recall']:.4f}")
    print(f"Official test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Official test calibration ECE: {test_metrics['calibration']['expected_calibration_error']:.4f}")
    print(f"Official test selective coverage: {test_metrics['selective_test_metrics']['coverage']:.4f}")
    print(f"Class metrics and confusion matrix: {final_run}")
    if args.activate:
        print(f"PlantDoc ONNX model activated: {final_run / 'model.onnx'}")
    else:
        print(f"PlantDoc ONNX model saved for review (not activated): {final_run / 'model.onnx'}")
    return final_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--target-selective-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-coverage", type=float, default=0.10)
    parser.add_argument("--activate", action="store_true", help="Make this validated run the active scanner model after export checks.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs, patience, and batch size must be positive; workers must not be negative")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation fraction must be in (0,1)")
    if not 0 < args.target_selective_accuracy <= 1 or not 0 < args.minimum_coverage <= 1:
        parser.error("selective accuracy and minimum coverage must be in (0,1]")
    try:
        train(args)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
