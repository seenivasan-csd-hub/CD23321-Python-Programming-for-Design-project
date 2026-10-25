# FarmIQ PlantDoc Leaf Classifier Model Card

## Current status

The FarmIQ PlantDoc training pipeline is implemented, but no PlantDoc dataset or trained PlantDoc checkpoint is present in this workspace. The PlantDoc model is not active. Until training finishes and its ONNX export passes verification, the scanner uses the existing local demo model, treating every output as an uncertain suggestion. No local model accuracy, calibration, or F1 result is available yet.

After a successful run, the generated `model_card.md`, `metadata.json`, `metrics.json`, per-class validation history, and confusion matrix in `models/plantdoc/runs/<run>/` are the authoritative record for that model. They include the immutable dataset commit and measured test results. A trained run is saved for review and is not activated by default.

## Intended use

Research/demo classification of leaf images into the exact PlantDoc class-folder labels seen during training. A model label is a candidate class, not a plant-health diagnosis or a treatment recommendation. Ask a local agricultural extension professional to confirm important or severe symptoms.

## Architecture and input

- MobileNetV3 Small initialized from TorchVision `MobileNet_V3_Small_Weights.IMAGENET1K_V1` and fine-tuned on PlantDoc.
- RGB input, shortest-edge resize to 256 pixels, center crop to 224 pixels, float32 conversion, and ImageNet mean/std normalization.
- Training exports an ONNX model with saved class names and preprocessing metadata. Flask loads it lazily and uses ONNX Runtime on CPU; training is never performed at Flask startup.
- Training code downloads pretrained weights only when explicitly run. Their source is the official PyTorch checkpoint URL in the README. TorchVision notes pretrained weights can inherit dataset terms; ImageNet's access agreement limits use to non-commercial research and education. Verify terms for your intended use.

## Supported classes

Class names are discovered from the downloaded official repository's training directories and stored without merging or renaming. The final list is generated into each trained run's `classes.json` and `model_card.md`; it is unavailable until the dataset has been downloaded and training has completed.

## Training and evaluation

The pipeline reserves a deterministic validation subset from the official training split, groups byte-identical images before splitting, reports exact duplicates across official splits, applies class-weighted loss and moderate training augmentation, and early-stops on validation loss. It fits scalar temperature scaling on validation logits and chooses the lowest rejection threshold meeting the configured validation selective-accuracy target and minimum coverage. If no threshold meets that policy, it rejects all scores. The official test split is not used for model selection. Export verification compares PyTorch and ONNX logits numerically.

**Measured official test metrics:** Not available in this workspace. Run `scripts/train_leaf_model.py` after downloading PlantDoc; it will report measured accuracy, macro precision/recall/F1, per-class precision/recall/F1, calibration metrics (NLL, Brier score, ECE), selective accuracy/coverage, and a confusion matrix. The generated metrics must be used instead of estimates or dataset-paper metrics.

## Limitations and risks

- PlantDoc is a relatively small, web-collected dataset. Its class labels and image quality may contain noise.
- The model has not been validated on Tamil Nadu field photographs, local crop varieties, or local disease prevalence.
- Temperature-scaled output is a validation-calibrated score, not a field probability. Calibration can fail to transfer to field photos; the warning threshold is selected against a validation policy, not field outcomes.
- The model does not reliably detect out-of-distribution images. An unsupported plant can still receive a supported class label.
- A simple high-frequency detail check can warn about blur or low-detail images, but its threshold is not validated on field photos and it is not a crop/disease detector.
- A class-folder label may not describe a precise disease state. FarmIQ shows the exact class label and only provides specific symptom guidance where an extension source has been checked.
- No pesticide or dosage is inferred or recommended from a prediction.
- The legacy demo classifier and PlantDoc have incompatible label taxonomies; their overall metrics are not directly comparable. A new run is not activated by default, and no local before/after comparison exists until data and a compatible evaluation are available.

## Data sources and attribution

PlantDoc: [official dataset repository](https://github.com/pratikkayal/PlantDoc-Dataset), stated by the repository to be CC BY 4.0. Dataset paper: Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra, “PlantDoc: A Dataset for Visual Plant Disease Detection,” CoDS-COMAD 2020, [DOI 10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196), [arXiv:1911.10317](https://arxiv.org/abs/1911.10317). The paper describes internet-scraped images; dataset attribution does not necessarily resolve rights in every source image.
