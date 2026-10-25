"""Lazy ONNX inference for trained PlantDoc and the retained local demo model."""

from functools import lru_cache
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "plantdoc"
ACTIVE_MODEL_PATH = MODEL_DIR / "active.json"
LEGACY_MODEL_DIR = BASE_DIR / "models" / "plant-disease"
LEGACY_MODEL_PATH = LEGACY_MODEL_DIR / "model.onnx"
LEGACY_CLASSES_PATH = LEGACY_MODEL_DIR / "classes.json"
# Kept as aliases for older callers and tests.
MODEL_PATH = LEGACY_MODEL_PATH
CLASSES_PATH = LEGACY_CLASSES_PATH
MODEL_VERSION = "Gopina/maize-disease-classifier@812be29"
MODEL_URL = "https://huggingface.co/Gopina/maize-disease-classifier/resolve/812be29/model.onnx"
MODEL_CARD_URL = "https://huggingface.co/Gopina/maize-disease-classifier"
MODEL_LICENSE = "MIT (license metadata declared by the model author)"
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_CROPS = {"maize", "cassava", "bean"}
LEGACY_CLASS_ORDER = ("healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot")
LEGACY_EXPECTED_CROP = {
    "maize_rust": "maize",
    "maize_blight": "maize",
    "cassava_mosaic": "cassava",
    "bean_spot": "bean",
}
LOW_DETAIL_RMS_THRESHOLD = 8.0


class ModelUnavailable(Exception):
    pass


class ModelCompatibilityError(Exception):
    pass


def _read_labels(path):
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels or not all(isinstance(item, str) and item for item in labels):
        raise ModelCompatibilityError("The model class list is invalid.")
    if len({item.casefold() for item in labels}) != len(labels):
        raise ModelCompatibilityError("The model class list contains duplicate labels.")
    return labels


def _active_bundle():
    if not ACTIVE_MODEL_PATH.is_file():
        return None
    try:
        pointer = json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
        relative = Path(pointer["run"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid model run path")
        run_dir = (MODEL_DIR / relative).resolve()
        run_dir.relative_to(MODEL_DIR.resolve())
        model_path = run_dir / "model.onnx"
        classes_path = run_dir / "classes.json"
        metadata_path = run_dir / "metadata.json"
        if not all(path.is_file() for path in (model_path, classes_path, metadata_path)):
            raise FileNotFoundError("PlantDoc model artifacts are incomplete")
        labels = _read_labels(classes_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("classes") != labels or metadata.get("architecture") != "mobilenet_v3_small":
            raise ValueError("PlantDoc model metadata does not match the model classes")
        if metadata.get("input_size") != 224 or metadata.get("input_layout") != "NCHW":
            raise ValueError("PlantDoc preprocessing metadata is unsupported")
        plant_by_class = metadata.get("plant_by_class", {})
        if not isinstance(plant_by_class, dict):
            raise ValueError("PlantDoc crop metadata is invalid")
        if any(value is not None and not isinstance(value, str) for value in plant_by_class.values()):
            raise ValueError("PlantDoc crop metadata is invalid")
        return {
            "kind": "plantdoc",
            "name": metadata.get("model_name", "FarmIQ PlantDoc classifier"),
            "model_path": model_path,
            "classes_path": classes_path,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "labels": labels,
            "version": metadata.get("model_version", pointer.get("model_version", "plantdoc")),
            "model_url": PLANTDOC_URL,
            "model_card_url": PLANTDOC_URL,
            "license": "PlantDoc CC BY 4.0; ImageNet pretrained weights have dataset-derived use terms",
            "supported_crops": sorted({name for name in plant_by_class.values() if name}),
            "weight_size": model_path.stat().st_size,
        }
    except (ModelCompatibilityError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError):
        # A partial/invalid active pointer never prevents the previous scanner from working.
        return None


def _legacy_bundle(model_path=LEGACY_MODEL_PATH, classes_path=LEGACY_CLASSES_PATH):
    model_path, classes_path = Path(model_path), Path(classes_path)
    labels = []
    if classes_path.is_file():
        try:
            labels = _read_labels(classes_path)
        except (OSError, ValueError, json.JSONDecodeError, ModelCompatibilityError):
            labels = []
    return {
        "kind": "legacy",
        "name": "Local research/demo classifier",
        "model_path": model_path,
        "classes_path": classes_path,
        "metadata_path": None,
        "metadata": {"input_size": 224, "resize_size": 224, "normalization": "scale_0_1"},
        "labels": labels,
        "class_mapping_valid": tuple(labels) == LEGACY_CLASS_ORDER,
        "version": MODEL_VERSION,
        "model_url": MODEL_URL,
        "model_card_url": MODEL_CARD_URL,
        "license": MODEL_LICENSE,
        "supported_crops": sorted(SUPPORTED_CROPS),
        "weight_size": model_path.stat().st_size if model_path.is_file() else 0,
    }


def resolve_model(model_path=None, classes_path=None):
    """Use the trained PlantDoc export when active; otherwise retain the prior scanner."""
    if model_path is not None or classes_path is not None:
        return _legacy_bundle(model_path or LEGACY_MODEL_PATH, classes_path or LEGACY_CLASSES_PATH)
    return _active_bundle() or _legacy_bundle()


def model_status(model_path=None, classes_path=None):
    bundle = resolve_model(model_path, classes_path)
    if not bundle["model_path"].is_file() or not bundle["classes_path"].is_file() or not bundle["labels"]:
        return {
            **bundle,
            "available": False,
            "message": "Model weights are unavailable; uploads will not be labeled as predictions.",
            "classes": bundle["labels"],
        }
    if bundle["kind"] == "legacy" and not bundle["class_mapping_valid"]:
        return {
            **bundle,
            "available": False,
            "message": "The demo model class order does not match its verified ONNX label mapping; no predictions will be made.",
            "classes": bundle["labels"],
        }
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return {
            **bundle,
            "available": False,
            "message": "Install the optional disease runtime to enable local predictions.",
            "classes": bundle["labels"],
        }
    if bundle["kind"] == "plantdoc" and bundle["metadata"].get("calibration") == "temperature_scaling_on_validation":
        message = "Trained PlantDoc classifier ready. Research use only; validation calibration is recorded, with no out-of-distribution detector."
    elif bundle["kind"] == "plantdoc":
        message = "Trained PlantDoc classifier ready, but validation calibration metadata is missing; results will be uncertain suggestions."
    else:
        message = "Local ONNX demo classifier ready. Research use only; scores are uncalibrated and all results are treated as uncertain suggestions."
    return {**bundle, "available": True, "message": message, "classes": bundle["labels"]}


def _load_session(model_path, classes_path, metadata_path_text=""):
    try:
        import onnxruntime as ort

        labels = _read_labels(classes_path)
        metadata_path = Path(metadata_path_text) if metadata_path_text else None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path and metadata_path.is_file() else {
            "input_size": 224, "resize_size": 224, "normalization": "scale_0_1"
        }
        if metadata.get("classes") is not None and metadata["classes"] != labels:
            raise ModelCompatibilityError("Model metadata class names do not match classes.json.")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ModelCompatibilityError("The model must expose one image input and one output.")
        shape = session.get_inputs()[0].shape
        if len(shape) != 4:
            raise ModelCompatibilityError("The model input must be a four-dimensional image tensor.")
        size = int(metadata.get("input_size", 224))
        if shape[1:] not in ([3, size, size], [size, size, 3], (3, size, size), (size, size, 3)):
            raise ModelCompatibilityError("Model input dimensions do not match its preprocessing metadata.")
        return session, labels, metadata
    except ModelCompatibilityError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError("The optional disease model could not be loaded.") from exc


@lru_cache(maxsize=4)
def _cached_session(model_path_text, classes_path_text, metadata_path_text=""):
    return _load_session(model_path_text, classes_path_text, metadata_path_text)


def _validate_image(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise ModelUnavailable("The selected image file is unavailable.")
    if path.suffix.casefold() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ModelUnavailable("Choose a JPG, PNG, or WEBP image.")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ModelUnavailable("The image exceeds the 5 MB limit.")
    try:
        from PIL import Image

        with Image.open(path) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP"}:
                raise ModelUnavailable("Choose a JPG, PNG, or WEBP image.")
            if min(opened.size) < 128 or opened.width * opened.height > 20_000_000:
                raise ModelUnavailable("Use a valid image between 128 pixels and 20 megapixels.")
            opened.verify()
    except ModelUnavailable:
        raise
    except Exception as exc:
        raise ModelUnavailable("This image could not be decoded. Choose another leaf photo.") from exc
    return path


def assess_image_quality(image_path):
    """Return a conservative detail/blur warning; this heuristic is not field validated."""
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

        with Image.open(image_path) as opened:
            gray = ImageOps.exif_transpose(opened).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        high_frequency = ImageStat.Stat(ImageChops.difference(gray, gray.filter(ImageFilter.GaussianBlur(2)))).rms[0]
    except Exception:
        return {
            "detail_score": None,
            "warning": "Photo detail could not be checked. Retake a sharp, well-lit image if the result looks unclear.",
        }
    warning = None
    if high_frequency < LOW_DETAIL_RMS_THRESHOLD:
        warning = "The photo may be blurry or lack visible leaf detail. Retake a sharp, well-lit close-up; this simple quality check is only a heuristic."
    return {"detail_score": float(high_frequency), "warning": warning}


def preprocess_image(image_path, input_shape, metadata=None):
    """Use the model's saved RGB resize/crop/normalization configuration."""
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ModelUnavailable("Install Pillow, NumPy, and the optional disease runtime to analyze images.") from exc
    metadata = metadata or {"input_size": 224, "resize_size": 224, "normalization": "scale_0_1"}
    size = int(metadata.get("input_size", 224))
    resize_size = int(metadata.get("resize_size", size))
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if metadata.get("normalization") == "scale_0_1":
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        else:
            width, height = image.size
            scale = resize_size / min(width, height)
            resized = (int(width * scale), int(height * scale))
            image = image.resize(resized, Image.Resampling.BILINEAR)
            left = max(0, int(round((image.width - size) / 2)))
            top = max(0, int(round((image.height - size) / 2)))
            image = image.crop((left, top, left + size, top + size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    mean = metadata.get("mean")
    std = metadata.get("std")
    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(std_array <= 0):
            raise ModelCompatibilityError("Model normalization standard deviations must be positive.")
        array = (array - mean_array) / std_array
    if tuple(input_shape[1:]) in ((3, size, size), [3, size, size]):
        array = np.transpose(array, (2, 0, 1))
    elif tuple(input_shape[1:]) not in ((size, size, 3), [size, size, 3]):
        raise ModelCompatibilityError("Expected NHWC or NCHW image input matching the saved model size.")
    return np.expand_dims(array, axis=0)


def predict(image_path, selected_crop=None, model_path=None, classes_path=None, metadata_path=None):
    bundle = resolve_model(model_path, classes_path)
    if metadata_path is not None:
        bundle["metadata_path"] = Path(metadata_path)
    if bundle["kind"] == "legacy" and selected_crop not in SUPPORTED_CROPS:
        raise ModelUnavailable("The current demo model requires maize, cassava, or bean to be selected.")
    if not bundle["model_path"].is_file() or not bundle["classes_path"].is_file():
        raise ModelUnavailable("Model weights are missing; the image was saved without a disease prediction.")
    if bundle["kind"] == "legacy" and not bundle["class_mapping_valid"]:
        raise ModelCompatibilityError("The demo model class order does not match its verified ONNX label mapping.")

    safe_image = _validate_image(image_path)
    metadata_path_text = str(bundle["metadata_path"]) if bundle.get("metadata_path") else ""
    session, labels, metadata = _cached_session(
        str(bundle["model_path"]), str(bundle["classes_path"]), metadata_path_text
    )
    input_meta = session.get_inputs()[0]
    tensor = preprocess_image(safe_image, input_meta.shape, metadata)
    try:
        import numpy as np

        output = np.asarray(session.run(None, {input_meta.name: tensor})[0], dtype=np.float64).reshape(-1)
    except (ValueError, RuntimeError, TypeError) as exc:
        raise ModelCompatibilityError("The model could not process this image.") from exc
    if len(output) != len(labels) or not np.all(np.isfinite(output)):
        raise ModelCompatibilityError("Model output classes do not match the class mapping.")
    is_probability_vector = np.all(output >= 0) and np.all(output <= 1) and 0.98 <= float(output.sum()) <= 1.02
    if bundle["kind"] == "legacy" and is_probability_vector:
        scores = output
    else:
        temperature = float(metadata.get("temperature", 1.0))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ModelCompatibilityError("The model calibration temperature is invalid.")
        shifted = output / temperature
        shifted -= float(shifted.max())
        exp_values = np.exp(shifted)
        scores = exp_values / float(exp_values.sum())
    index = int(scores.argmax())
    raw_label = labels[index]
    confidence = float(scores[index])
    threshold = float(metadata.get("low_confidence_threshold", 0.60))
    plant = metadata.get("plant_by_class", {}).get(raw_label)
    selected_plant = selected_crop.title() if selected_crop else None
    mismatch = False
    if bundle["kind"] == "legacy":
        plant = LEGACY_EXPECTED_CROP.get(raw_label, selected_crop).title()
        mismatch = raw_label != "healthy" and LEGACY_EXPECTED_CROP.get(raw_label) != selected_crop
    reasons = []
    if mismatch:
        reasons.append(f"The selected crop is {selected_plant}, but the top class maps to {plant}.")
    if confidence < threshold:
        reasons.append("The top model score is below the configured warning threshold.")
    order = np.argsort(scores)[::-1][:min(3, len(labels))]
    top_predictions = []
    for candidate_index in order:
        candidate_label = labels[int(candidate_index)]
        candidate_crop = (
            LEGACY_EXPECTED_CROP.get(candidate_label, selected_crop)
            if bundle["kind"] == "legacy"
            else metadata.get("plant_by_class", {}).get(candidate_label)
        )
        top_predictions.append({
            "label": candidate_label,
            "crop": candidate_crop.title() if candidate_crop else None,
            "score": float(scores[int(candidate_index)]),
        })
    has_validation_calibration = bundle["kind"] == "plantdoc" and metadata.get("calibration") == "temperature_scaling_on_validation"
    if bundle["kind"] == "legacy":
        reasons.append("This demo model has no local validation calibration or out-of-distribution check.")
    elif not has_validation_calibration:
        reasons.append("This model has no validation-fitted calibration metadata.")
    if bundle["kind"] == "plantdoc" and not plant:
        reasons.append("The model class has no verified crop mapping in FarmIQ.")
    return {
        "label": "uncertain" if reasons else raw_label,
        "raw_label": raw_label,
        "class_name": "Uncertain result" if reasons else raw_label,
        "crop": plant,
        "predicted_crop": plant,
        "selected_crop": selected_plant,
        "crop_mismatch": mismatch,
        "uncertain": bool(reasons),
        "confidence": confidence,
        "score_label": (
            "Validation temperature-scaled score; not a field probability"
            if has_validation_calibration
            else "Uncalibrated raw model score; not a probability"
        ),
        "top_predictions": top_predictions,
        "class_scores": [float(score) for score in scores],
        "uncertainty_reasons": reasons,
        "ood_detection_available": False,
        "model_version": bundle["version"],
        "model_card_url": bundle["model_card_url"],
        "model_kind": bundle["kind"],
    }
