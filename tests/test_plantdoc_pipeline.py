import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from plantdoc_data import collect_dataset, validate_for_training, validation_split  # noqa: E402
from prepare_plantdoc import _safe_extract  # noqa: E402
from train_leaf_model import (  # noqa: E402
    _calibration_metrics, _fit_temperature, _metrics, _select_rejection_threshold, _softmax,
)
from services import disease_model  # noqa: E402
from services.disease_guidance import guidance_for_prediction  # noqa: E402


def save_image(path, color=(40, 120, 70)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 144), color).save(path, format="JPEG")


class PlantDocPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "archive-wrapper" / "PlantDoc-Dataset-master"

    def tearDown(self):
        self.temp.cleanup()

    def test_discovers_official_train_test_and_keeps_directory_labels_exact(self):
        save_image(self.root / "train" / "Corn rust leaf" / "train_1.JPG")
        save_image(self.root / "train" / "Corn rust leaf" / "train_2.jpeg", (70, 50, 140))
        save_image(self.root / "train" / "Tomato leaf" / "tomato.png", (130, 60, 30))
        save_image(self.root / "test" / "Corn rust leaf" / "test.webp", (110, 140, 50))
        save_image(self.root / "test" / "Tomato leaf" / "test.jpeg", (20, 40, 160))
        records, report = collect_dataset(self.root)
        labels = {row.label for row in records["train"]}
        self.assertEqual(labels, {"Corn rust leaf", "Tomato leaf"})
        self.assertEqual(report["splits"]["train"]["valid_images"], 3)
        self.assertEqual(report["splits"]["test"]["valid_images"], 2)
        validate_for_training(records, report)

    def test_reports_and_skips_corrupt_images_without_silently_counting_them(self):
        save_image(self.root / "train" / "Apple Scab Leaf" / "valid.jpg")
        bad = self.root / "train" / "Apple Scab Leaf" / "broken.png"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not an image")
        save_image(self.root / "test" / "Apple Scab Leaf" / "test.jpg")
        save_image(self.root / "test" / "Apple leaf" / "test-2.jpg")
        records, report = collect_dataset(self.root)
        self.assertEqual(report["splits"]["train"]["class_counts"]["Apple Scab Leaf"], {
            "images_found": 2, "images_valid": 1, "images_corrupt": 1, "images_too_large": 0,
        })
        self.assertEqual(len(records["train"]), 1)

    def test_validation_split_keeps_byte_identical_images_together(self):
        for index in range(4):
            save_image(self.root / "train" / "Grape leaf" / f"image-{index}.jpg", (index * 20, 80, 100))
        # These are the exact same bytes and must be assigned to one side as a group.
        source = self.root / "train" / "Grape leaf" / "image-0.jpg"
        (source.parent / "duplicate.jpg").write_bytes(source.read_bytes())
        save_image(self.root / "test" / "Grape leaf" / "test.jpg")
        records, _ = collect_dataset(self.root)
        train, validation = validation_split(records["train"], fraction=0.5, seed=21)
        train_hashes = {row.sha256 for row in train}
        validation_hashes = {row.sha256 for row in validation}
        self.assertFalse(train_hashes & validation_hashes)
        self.assertEqual(len(train) + len(validation), len(records["train"]))

    def test_evaluation_metrics_compute_macro_f1_and_per_class_counts(self):
        metrics = _metrics([0, 0, 1], [0, 1, 1], ["Corn rust leaf", "Tomato leaf"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_precision"], 0.75)
        self.assertAlmostEqual(metrics["macro_recall"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 2 / 3) / 2)
        self.assertEqual(metrics["per_class"]["Corn rust leaf"]["support"], 2)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])

    def test_validation_calibration_and_rejection_threshold_are_data_driven(self):
        logits = np.array([[4.0, 0.0], [0.0, 4.0], [0.4, 0.0], [0.0, 0.4]])
        targets = np.array([0, 1, 1, 0])
        temperature = _fit_temperature(logits, targets)
        self.assertTrue(np.isfinite(temperature) and temperature > 0)
        probabilities = _softmax(logits, temperature)
        metrics = _calibration_metrics(targets, probabilities)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        policy = _select_rejection_threshold(probabilities, targets, target_accuracy=0.9, minimum_coverage=0.25)
        self.assertGreater(policy["threshold"], 0)
        self.assertLessEqual(policy["validation_coverage"], 1)
        self.assertIn("target_met", policy)

    def test_legacy_mismatch_is_uncertain_even_with_a_high_model_score(self):
        labels = list(disease_model.LEGACY_CLASS_ORDER)
        with tempfile.TemporaryDirectory() as temp_name:
            image_path = Path(temp_name) / "bean.jpg"
            save_image(image_path)

            class TensorInfo:
                name = "image"
                shape = [1, 224, 224, 3]

            class FakeSession:
                def get_inputs(self):
                    return [TensorInfo()]

                def run(self, _outputs, _inputs):
                    return [np.array([[0.0001, 0.0001, 0.0001, 0.9996, 0.0001]], dtype=np.float32)]

            with patch.object(disease_model, "_cached_session", return_value=(FakeSession(), labels, {
                "input_size": 224, "resize_size": 224, "normalization": "scale_0_1",
            })):
                prediction = disease_model.predict(image_path, selected_crop="bean")

        self.assertAlmostEqual(prediction["confidence"], 0.9996, places=4)
        self.assertEqual(prediction["selected_crop"], "Bean")
        self.assertEqual(prediction["predicted_crop"], "Cassava")
        self.assertTrue(prediction["crop_mismatch"])
        self.assertTrue(prediction["uncertain"])
        self.assertEqual(prediction["class_name"], "Uncertain result")
        self.assertIn("not a probability", prediction["score_label"])

    def test_supported_healthy_and_disease_labels_remain_advisory_without_validation(self):
        labels = list(disease_model.LEGACY_CLASS_ORDER)
        with tempfile.TemporaryDirectory() as temp_name:
            image_path = Path(temp_name) / "leaf.jpg"
            save_image(image_path)

            class TensorInfo:
                name = "image"
                shape = [1, 224, 224, 3]

            class FakeSession:
                def __init__(self, values):
                    self.values = values

                def get_inputs(self):
                    return [TensorInfo()]

                def run(self, _outputs, _inputs):
                    return [np.asarray([self.values], dtype=np.float32)]

            for crop, values, expected in (
                ("maize", [0.9996, 0.0001, 0.0001, 0.0001, 0.0001], "healthy"),
                ("maize", [0.0001, 0.9996, 0.0001, 0.0001, 0.0001], "maize_rust"),
            ):
                with self.subTest(expected=expected), patch.object(
                    disease_model, "_cached_session", return_value=(FakeSession(values), labels, {
                        "input_size": 224, "resize_size": 224, "normalization": "scale_0_1",
                    })
                ):
                    prediction = disease_model.predict(image_path, selected_crop=crop)
                    self.assertEqual(prediction["raw_label"], expected)
                    self.assertTrue(prediction["uncertain"])

    def test_low_confidence_calibrated_prediction_is_rejected(self):
        labels = ["Bean leaf spot", "Maize rust"]
        with tempfile.TemporaryDirectory() as temp_name:
            model_root = Path(temp_name) / "plantdoc"
            run_dir = model_root / "runs" / "test-run"
            run_dir.mkdir(parents=True)
            model_path = run_dir / "model.onnx"
            model_path.write_bytes(b"mock")
            (run_dir / "classes.json").write_text(json.dumps(labels), encoding="utf-8")
            metadata = {
                "model_name": "test", "model_version": "plantdoc@test", "architecture": "mobilenet_v3_small",
                "classes": labels, "input_size": 224, "input_layout": "NCHW",
                "calibration": "temperature_scaling_on_validation", "temperature": 1.0,
                "low_confidence_threshold": 0.6,
                "plant_by_class": {labels[0]: "Bean", labels[1]: "Maize"},
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            active_path = model_root / "active.json"
            active_path.write_text(json.dumps({"run": "runs/test-run"}), encoding="utf-8")
            image_path = Path(temp_name) / "leaf.jpg"
            save_image(image_path)

            class TensorInfo:
                name = "image"
                shape = [1, 3, 224, 224]

            class FakeSession:
                def get_inputs(self):
                    return [TensorInfo()]

                def run(self, _outputs, _inputs):
                    return [np.asarray([[0.1, 0.0]], dtype=np.float32)]

            with patch.object(disease_model, "MODEL_DIR", model_root), patch.object(
                disease_model, "ACTIVE_MODEL_PATH", active_path
            ), patch.object(disease_model, "_cached_session", return_value=(FakeSession(), labels, metadata)):
                prediction = disease_model.predict(image_path)

        self.assertTrue(prediction["uncertain"])
        self.assertLess(prediction["confidence"], 0.6)
        self.assertIn("below the configured warning threshold", prediction["uncertainty_reasons"][0])

    def test_preprocessing_keeps_rgb_order_and_expected_layout(self):
        with tempfile.TemporaryDirectory() as temp_name:
            image_path = Path(temp_name) / "red.jpg"
            Image.new("RGB", (224, 224), (255, 0, 0)).save(image_path, format="PNG")
            nhwc = disease_model.preprocess_image(
                image_path, [1, 224, 224, 3], {"input_size": 224, "resize_size": 224, "normalization": "scale_0_1"}
            )
            nchw = disease_model.preprocess_image(
                image_path, [1, 3, 224, 224], {"input_size": 224, "resize_size": 224, "normalization": "scale_0_1"}
            )
        self.assertEqual(nhwc.shape, (1, 224, 224, 3))
        self.assertEqual(nchw.shape, (1, 3, 224, 224))
        self.assertGreater(float(nhwc[0, 100, 100, 0]), 0.99)
        self.assertLess(float(nhwc[0, 100, 100, 1]), 0.01)
        self.assertLess(float(nhwc[0, 100, 100, 2]), 0.01)

    def test_image_quality_heuristic_warns_for_blurred_detail(self):
        from PIL import ImageDraw, ImageFilter

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            sharp_path = root / "sharp.png"
            blurred_path = root / "blurred.png"
            sharp = Image.new("RGB", (224, 224), "black")
            draw = ImageDraw.Draw(sharp)
            for x in range(0, 224, 8):
                for y in range(0, 224, 8):
                    if (x // 8 + y // 8) % 2:
                        draw.rectangle((x, y, x + 7, y + 7), fill="white")
            sharp.save(sharp_path)
            sharp.filter(ImageFilter.GaussianBlur(4)).save(blurred_path)
            sharp_check = disease_model.assess_image_quality(sharp_path)
            blurred_check = disease_model.assess_image_quality(blurred_path)

        self.assertIsNone(sharp_check["warning"])
        self.assertIsNotNone(blurred_check["warning"])
        self.assertIn("heuristic", blurred_check["warning"])

    def test_legacy_class_order_mismatch_disables_model(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            model_path = root / "model.onnx"
            classes_path = root / "classes.json"
            model_path.write_bytes(b"placeholder")
            classes_path.write_text(json.dumps(list(reversed(disease_model.LEGACY_CLASS_ORDER))), encoding="utf-8")
            status = disease_model.model_status(model_path, classes_path)
        self.assertFalse(status["available"])
        self.assertIn("class order", status["message"])

    def test_training_stops_on_exact_duplicates_across_official_splits(self):
        save_image(self.root / "train" / "Corn rust leaf" / "train.jpg")
        save_image(self.root / "train" / "Corn leaf blight" / "other.jpg", (80, 30, 100))
        duplicate = self.root / "test" / "Corn rust leaf" / "test.jpg"
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_bytes((self.root / "train" / "Corn rust leaf" / "train.jpg").read_bytes())
        save_image(self.root / "test" / "Corn leaf blight" / "test-2.jpg", (30, 90, 120))
        records, report = collect_dataset(self.root)
        self.assertEqual(len(report["cross_split_duplicate_groups"]), 1)
        with self.assertRaisesRegex(ValueError, "avoid leakage"):
            validate_for_training(records, report)

    def test_conflicting_exact_duplicate_labels_stop_training(self):
        source = self.root / "train" / "Apple Scab Leaf" / "image.jpg"
        save_image(source)
        duplicate = self.root / "train" / "Apple leaf" / "same.jpg"
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_bytes(source.read_bytes())
        save_image(self.root / "train" / "Apple Scab Leaf" / "other.jpg", (100, 120, 45))
        save_image(self.root / "test" / "Apple Scab Leaf" / "test.jpg", (30, 40, 130))
        save_image(self.root / "test" / "Apple leaf" / "test-2.jpg", (20, 100, 60))
        records, report = collect_dataset(self.root)
        self.assertEqual(len(report["conflicting_label_duplicate_groups"]), 1)
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            validate_for_training(records, report)

    def test_zip_extraction_rejects_parent_directory_paths(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../../outside.txt", "unsafe")
        buffer.seek(0)
        with tempfile.TemporaryDirectory() as destination:
            with zipfile.ZipFile(buffer) as archive:
                with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                    _safe_extract(archive, Path(destination))

    def test_plantdoc_model_is_preferred_and_returns_exact_class_metadata(self):
        with tempfile.TemporaryDirectory() as temp_name:
            model_root = Path(temp_name) / "plantdoc"
            run_dir = model_root / "runs" / "test-run"
            run_dir.mkdir(parents=True)
            model_path = run_dir / "model.onnx"
            model_path.write_bytes(b"fake onnx file used with a mocked inference session")
            labels = ["Corn rust leaf", "Tomato leaf"]
            (run_dir / "classes.json").write_text(json.dumps(labels), encoding="utf-8")
            metadata = {
                "model_name": "Test PlantDoc model",
                "model_version": "plantdoc@test",
                "architecture": "mobilenet_v3_small",
                "classes": labels,
                "input_size": 224,
                "input_layout": "NCHW",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "calibration": "temperature_scaling_on_validation",
                "temperature": 1.0,
                "plant_by_class": {"Corn rust leaf": "Corn", "Tomato leaf": "Tomato"},
                "low_confidence_threshold": 0.5,
                "dataset": {"commit": "test"},
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (model_root / "active.json").write_text(json.dumps({"run": "runs/test-run"}), encoding="utf-8")
            image_path = Path(temp_name) / "leaf.jpg"
            save_image(image_path)

            class TensorInfo:
                name = "image"
                shape = [None, 3, 224, 224]

            class FakeSession:
                def get_inputs(self):
                    return [TensorInfo()]

                def run(self, _outputs, _inputs):
                    import numpy as np
                    return [np.asarray([[3.0, 0.1]], dtype=np.float32)]

            with patch.object(disease_model, "MODEL_DIR", model_root), patch.object(
                disease_model, "ACTIVE_MODEL_PATH", model_root / "active.json"
            ), patch.object(disease_model, "_cached_session", return_value=(FakeSession(), labels, metadata)):
                status = disease_model.model_status()
                prediction = disease_model.predict(image_path)

        self.assertEqual(status["kind"], "plantdoc")
        self.assertEqual(status["classes"], labels)
        self.assertEqual(prediction["class_name"], "Corn rust leaf")
        self.assertEqual(prediction["crop"], "Corn")
        self.assertEqual(prediction["model_version"], "plantdoc@test")
        self.assertEqual(prediction["score_label"], "Validation temperature-scaled score; not a field probability")

    def test_guidance_is_exactly_mapped_and_other_plantdoc_classes_stay_general(self):
        maize = guidance_for_prediction({"raw_label": "Corn rust leaf", "uncertain": False})
        tomato = guidance_for_prediction({"raw_label": "Tomato Septoria leaf spot", "uncertain": False})
        self.assertIn("TNAU", maize["source_title"])
        self.assertIn("No source-verified guidance", tomato["name"])
        self.assertNotIn("pesticide", tomato["next_steps"].casefold())


if __name__ == "__main__":
    unittest.main()
