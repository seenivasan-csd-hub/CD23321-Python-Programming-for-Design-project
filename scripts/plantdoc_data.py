"""Shared PlantDoc discovery, validation, and leakage checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
SPLITS = ("train", "test")
MAX_DATASET_IMAGE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    sha256: str


def _is_class_split(path: Path) -> bool:
    try:
        return any(child.is_dir() and not child.name.startswith(".") for child in path.iterdir())
    except OSError:
        return False


def discover_split_roots(dataset_root: Path) -> dict[str, Path]:
    """Find train/test class-folder roots without assuming the archive's top folder."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PlantDoc dataset directory does not exist: {root}")

    candidates: dict[str, list[Path]] = {split: [] for split in SPLITS}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.casefold() in candidates and _is_class_split(path):
            candidates[path.name.casefold()].append(path)

    result = {}
    for split, paths in candidates.items():
        if not paths:
            raise ValueError(
                f"Could not find a {split}/<class>/<image> directory under {root}. "
                "Keep the official train and test folders intact."
            )
        shallowest = min(len(path.relative_to(root).parts) for path in paths)
        nearest = [path for path in paths if len(path.relative_to(root).parts) == shallowest]
        if len(nearest) != 1:
            found = ", ".join(str(path) for path in nearest)
            raise ValueError(f"Found multiple equally likely {split} splits: {found}")
        result[split] = nearest[0]
    return result


def _image_files(class_directory: Path):
    for path in sorted(class_directory.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_file() and not path.is_symlink() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            yield path


def scan_split(split_root: Path) -> tuple[list[ImageRecord], dict]:
    """Validate image files; preserve directory labels exactly as authored."""
    split_root = Path(split_root)
    class_dirs = sorted(
        (path for path in split_root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda item: item.name.casefold(),
    )
    if not class_dirs:
        raise ValueError(f"No class directories found in split: {split_root}")
    folded = [path.name.casefold() for path in class_dirs]
    if len(folded) != len(set(folded)):
        raise ValueError(f"Class names that differ only by case are ambiguous in {split_root}")

    records: list[ImageRecord] = []
    counts = {}
    corrupt = []
    too_large = []
    empty_classes = []
    for class_dir in class_dirs:
        valid_count = 0
        discovered_count = 0
        oversized_before = len(too_large)
        for image_path in _image_files(class_dir):
            discovered_count += 1
            try:
                if image_path.stat().st_size > MAX_DATASET_IMAGE_BYTES:
                    too_large.append(image_path.relative_to(split_root).as_posix())
                    continue
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    if image.width < 1 or image.height < 1 or image.width * image.height > 40_000_000:
                        raise ValueError("image dimensions are empty or exceed the 40 megapixel safety limit")
            except Exception as exc:  # Pillow decoder exceptions vary by plugin/format.
                corrupt.append({"path": image_path.relative_to(split_root).as_posix(), "error": type(exc).__name__})
                continue
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            records.append(ImageRecord(image_path, class_dir.name, digest))
            valid_count += 1
        counts[class_dir.name] = {
            "images_found": discovered_count,
            "images_valid": valid_count,
            "images_corrupt": discovered_count - valid_count - (len(too_large) - oversized_before),
            "images_too_large": len(too_large) - oversized_before,
        }
        if valid_count == 0:
            empty_classes.append(class_dir.name)
    return records, {
        "class_counts": counts,
        "corrupt_images": corrupt,
        "too_large_images": too_large,
        "empty_classes": empty_classes,
        "valid_images": len(records),
    }


def collect_dataset(dataset_root: Path) -> tuple[dict[str, list[ImageRecord]], dict]:
    roots = discover_split_roots(dataset_root)
    records: dict[str, list[ImageRecord]] = {}
    report = {"dataset_root": str(Path(dataset_root).resolve()), "splits": {}, "warnings": []}
    source_path = Path(dataset_root).resolve() / "plantdoc-source.json"
    if source_path.is_file():
        try:
            report["source"] = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("PlantDoc source metadata is unreadable; re-download or review the dataset snapshot.") from exc
    for split in SPLITS:
        records[split], report["splits"][split] = scan_split(roots[split])
        report["splits"][split]["directory"] = str(roots[split])
        if not records[split]:
            report["warnings"].append(f"The {split} split contains no readable images.")
        if report["splits"][split]["empty_classes"]:
            report["warnings"].append(
                f"The {split} split has class folders with no readable images: "
                + ", ".join(report["splits"][split]["empty_classes"])
            )
        if report["splits"][split]["too_large_images"]:
            report["warnings"].append(
                f"The {split} split has {len(report['splits'][split]['too_large_images'])} image(s) over the size limit."
            )

    train_labels = {record.label for record in records["train"]}
    test_labels = {record.label for record in records["test"]}
    missing_train = sorted(test_labels - train_labels, key=str.casefold)
    if missing_train:
        report["test_classes_without_training_images"] = missing_train
        report["warnings"].append(
            "Test classes are absent from training and cannot be predicted: " + ", ".join(missing_train)
        )

    by_hash: dict[str, list[tuple[str, ImageRecord]]] = defaultdict(list)
    for split in SPLITS:
        for record in records[split]:
            by_hash[record.sha256].append((split, record))
    duplicate_groups = []
    cross_split = []
    conflicting_labels = []
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        entry = {
            "sha256": digest,
            "images": [
                {"split": split, "label": record.label, "path": record.path.as_posix()}
                for split, record in group
            ],
        }
        duplicate_groups.append(entry)
        if len({record.label for _, record in group}) > 1:
            conflicting_labels.append(entry)
        if {split for split, _ in group} == {"train", "test"}:
            cross_split.append(entry)
    report["duplicate_groups"] = duplicate_groups
    report["cross_split_duplicate_groups"] = cross_split
    report["conflicting_label_duplicate_groups"] = conflicting_labels
    if cross_split:
        report["warnings"].append(
            f"Found {len(cross_split)} exact image duplicate group(s) across the official train/test split."
        )
    if conflicting_labels:
        report["warnings"].append(
            f"Found {len(conflicting_labels)} exact image duplicate group(s) with different labels."
        )
    return records, report


def validation_split(
    training_records: list[ImageRecord], fraction: float = 0.2, seed: int = 1729
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split training images by content hash so exact duplicates cannot cross subsets."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be greater than 0 and less than 1.")
    import random

    groups: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in training_records:
        groups[record.label][record.sha256].append(record)
    rng = random.Random(seed)
    train, validation = [], []
    for label in sorted(groups, key=str.casefold):
        hashes = sorted(groups[label])
        rng.shuffle(hashes)
        if len(hashes) < 2:
            validation_hashes = set()
        else:
            validation_groups = min(len(hashes) - 1, max(1, round(len(hashes) * fraction)))
            validation_hashes = set(hashes[:validation_groups])
        for image_hash, duplicate_records in groups[label].items():
            target = validation if image_hash in validation_hashes else train
            target.extend(duplicate_records)
    return train, validation


def validate_for_training(records: dict[str, list[ImageRecord]], report: dict) -> list[str]:
    if not records["train"] or not records["test"]:
        raise ValueError("Both official PlantDoc train and test splits must contain readable images.")
    if report.get("cross_split_duplicate_groups"):
        raise ValueError(
            "Exact duplicate images cross the official train/test split. See the dataset report; "
            "training stopped to avoid leakage and the test split was left untouched."
        )
    if report.get("conflicting_label_duplicate_groups"):
        raise ValueError(
            "Identical image bytes have conflicting labels. Review the dataset report before training."
        )
    empty_classes = {
        (split, label)
        for split in SPLITS
        for label in report["splits"][split].get("empty_classes", [])
    }
    if empty_classes:
        descriptions = ", ".join(f"{split}/{label}" for split, label in sorted(empty_classes))
        raise ValueError("Class folders have no readable images; review the dataset report: " + descriptions)
    missing_train = report.get("test_classes_without_training_images", [])
    if missing_train:
        raise ValueError("Test classes missing from training: " + ", ".join(missing_train))
    labels = sorted({record.label for record in records["train"]}, key=str.casefold)
    if len(labels) < 2:
        raise ValueError("At least two readable training classes are required.")
    return labels


def write_report(path: Path, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_rows(records: list[ImageRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.label for record in records).items(), key=lambda item: item[0].casefold()))
