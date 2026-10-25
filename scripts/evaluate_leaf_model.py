"""Evaluate the active PlantDoc ONNX model on the untouched official test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from plantdoc_data import collect_dataset, validate_for_training
from train_leaf_model import _calibration_metrics, _metrics, _selective_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "plantdoc" / "raw"
sys.path.insert(0, str(PROJECT_ROOT))
from services.disease_model import ModelUnavailable, model_status, predict


def evaluate(data_root: Path) -> dict:
    records, report = collect_dataset(data_root)
    validate_for_training(records, report)
    status = model_status()
    if not status["available"] or status["kind"] != "plantdoc":
        raise RuntimeError("No trained PlantDoc model is active. Train it before running this evaluation.")
    metadata = status["metadata"]
    source_path = Path(data_root) / "plantdoc-source.json"
    if source_path.is_file():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        trained_commit = metadata.get("dataset", {}).get("commit")
        current_commit = source.get("commit")
        if trained_commit and current_commit and trained_commit != current_commit:
            raise RuntimeError(
                "The local dataset commit differs from the commit used by this model; "
                "select the matching PlantDoc snapshot before evaluating."
            )

    labels = status["classes"]
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown_test_labels = sorted({row.label for row in records["test"]} - set(labels), key=str.casefold)
    if unknown_test_labels:
        raise RuntimeError("Official test split has labels absent from the active model: " + ", ".join(unknown_test_labels))

    actual, predicted, probability_rows = [], [], []
    for index, row in enumerate(records["test"], start=1):
        try:
            result = predict(row.path, selected_crop=None)
        except ModelUnavailable as exc:
            raise RuntimeError(f"Test image could not be evaluated: {row.path.name}: {exc}") from exc
        actual.append(label_to_index[row.label])
        predicted.append(label_to_index[result["raw_label"]])
        probability_rows.append(result["class_scores"])
        if index % 25 == 0 or index == len(records["test"]):
            print(f"Evaluated {index}/{len(records['test'])} official test images")
    metrics = _metrics(actual, predicted, labels)
    probabilities = np.asarray(probability_rows, dtype=np.float64)
    metrics["calibration"] = _calibration_metrics(actual, probabilities)
    threshold = float(metadata.get("low_confidence_threshold", 1.01))
    metrics["selective_metrics"] = _selective_metrics(probabilities, actual, threshold)
    metrics["confidence_type"] = metadata.get("confidence_type", "unspecified")
    metrics["rejection_threshold"] = threshold
    metrics.update({
        "evaluation_split": "official PlantDoc test",
        "test_split_modified": False,
        "model_version": status["version"],
        "dataset_commit": metadata.get("dataset", {}).get("commit"),
    })
    run_dir = status["model_path"].parent
    metrics_path = run_dir / "independent_test_evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (run_dir / "independent_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        writer.writerows([[label, *row] for label, row in zip(labels, metrics["confusion_matrix"])])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(args.data_root)
    except (FileNotFoundError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    print(f"Official test accuracy: {metrics['accuracy']:.4f}")
    print(f"Official test macro precision: {metrics['macro_precision']:.4f}")
    print(f"Official test macro recall: {metrics['macro_recall']:.4f}")
    print(f"Official test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Official test calibration ECE: {metrics['calibration']['expected_calibration_error']:.4f}")
    print(f"Test images: {metrics['support']}")
    print(f"Per-class metrics and confusion matrix: {model_status()['model_path'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
