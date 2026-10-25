# FarmIQ

Course repository: `CD23321-Python-Programming-for-Design-project`
Project: Python Programming for Design project -Farm Decision Assistant

FarmIQ is a local-first smart agriculture and decision-support application for Tamil Nadu. It provides a district-aware crop calendar, source-labelled soil-test records, farm profile, weather forecasts, market-price tools, and an optional local plant-image scanner. Its original green-and-gold SVG emblem and favicon are served from the app's local static assets.

FarmIQ accounts use email and password only. There are no external identity-provider sign-in routes or buttons.

## Technology stack

- Python and Flask for the application and HTTP routes.
- SQLite for local farm profiles, reports, advisories and cache records.
- Jinja templates, responsive CSS and vanilla JavaScript for the interface, location consent and camera capture.
- Requests for weather and market data providers; Pillow for image validation; optional NumPy and CPU ONNX Runtime for local disease-model inference.
- No external front-end CDN is required.

## Accounts and data access

- Create an account at `/register` and sign in at `/login`. Email addresses are stored case-insensitively and unique. Passwords must be at least 12 characters and are stored as Werkzeug scrypt hashes; plaintext passwords are never stored.
- The Flask session is signed, HTTP-only, same-site and expires after 12 hours. CSRF checks cover login, registration, logout and the existing state-changing forms. All farm, soil, advisory, imported-price, weather and scan records are scoped to the signed-in account.
- This project previously had local farm records but no account system. The first account must confirm ownership of those unassigned local records; later registrations cannot claim them. Migration preserves the legacy tables and their rows.
- Password reset/email recovery is not implemented. The app generates a random session key at startup if `FLASK_SECRET_KEY` is unset, which signs users out after a restart. Configure a private persistent key through the environment for deployments. Set `FLASK_COOKIE_SECURE=1` when serving over HTTPS.

This repository root is the Flask application root. In the original VS Code starter workspace, this repository folder sits next to the workspace-level `.venv`; the included `run.ps1` detects either that environment or a `.venv` created inside a clone.

## Start in VS Code on Windows

Open a PowerShell terminal at the repository root. For a fresh clone, create and activate a virtual environment and install the app requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

Start the application from the repository root:

```powershell
.\run.ps1
```

Open **http://127.0.0.1:5000**. The runner uses the repository's `.venv` when present, or the parent workspace `.venv` in the original VS Code layout.

To activate the existing workspace environment and launch directly instead, use `..\.venv\Scripts\Activate.ps1` from this folder. Then run:

```powershell
python .\app.py
```

For a fresh clone, activate its local `.venv` instead. If PowerShell blocks activation, `run.ps1` can still launch the interpreter directly; `Set-ExecutionPolicy -Scope Process Bypass` affects only the current PowerShell process.

## PlantDoc leaf-disease scanner

The scanner prefers a trained FarmIQ PlantDoc model when one is active. Until then, it keeps using the existing local Gopina demo model, but every demo-model output is presented as uncertain because that model has no local validation calibration or out-of-distribution detector. Flask never downloads training data or pretrained weights and never trains during startup. The demo model is limited to its own five labels and three crop selections; the trained PlantDoc model shows the exact class-folder labels it learned.

### Dataset, attribution, and license

The preparation script downloads the official [PlantDoc-Dataset repository](https://github.com/pratikkayal/PlantDoc-Dataset), resolves the selected branch or tag to an immutable Git commit, records the commit and archive SHA-256, and keeps the repository's `train/<class>/` and `test/<class>/` splits intact under `data/plantdoc/raw/`. It discovers class directories rather than normalizing or merging their names. It reports readable, missing, and corrupt image counts, and checks byte-identical images for split leakage. Dataset and model artifacts are ignored by Git.

The repository states that PlantDoc is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset paper is Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra, "PlantDoc: A Dataset for Visual Plant Disease Detection," CoDS-COMAD 2020, DOI [10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196); see also [arXiv:1911.10317](https://arxiv.org/abs/1911.10317). The authors describe images collected from the web, so check source-image rights and attribution obligations for your use.

### Training and evaluation

The current workspace uses Python 3.14.3. PyTorch's official Windows installer supports Python 3.10-3.14. The training pipeline uses CPU PyTorch by default with TorchVision MobileNetV3 Small and its named `IMAGENET1K_V1` pretrained weights. TorchVision downloads those weights only when the training command runs; the checkpoint is served by PyTorch at [mobilenet_v3_small-047dcff4.pth](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth). TorchVision notes that pretrained-weight terms can follow the training dataset; [ImageNet's access agreement](https://image-net.org/accessagreement) limits ImageNet use to non-commercial research and educational purposes. Review those terms for your use, and select a separately licensed initialization for commercial work. Preprocessing is RGB, resize-short-edge 256, center-crop 224, and the ImageNet mean/std. Training checkpoints and ONNX exports stay in ignored `models/plantdoc/` versioned runs; Flask loads the active ONNX export once on first inference.

From PowerShell at the repository root, activate the local virtual environment. In the original VS Code starter workspace, activate the parent workspace environment with `..\.venv\Scripts\Activate.ps1`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the CPU framework wheels from the official PyTorch index, then the app's optional scanner/training packages:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r .\requirements-training.txt
```

Download and inspect the official dataset:

```powershell
python .\scripts\prepare_plantdoc.py --download --revision master
```

Train on the official train split, hold out a deterministic validation subset from that training data, early-stop on validation loss, then evaluate once on the untouched official test split:

```powershell
python .\scripts\train_leaf_model.py --epochs 15 --device auto
```

Training writes the best validation checkpoint, epoch/per-class logs, class labels, preprocessing metadata, ONNX model, official test accuracy, macro precision/recall/F1, per-class metrics, calibration metrics, selective coverage, confusion matrix, and model card into a new versioned run. Calibration temperature and a conservative rejection threshold are fitted on validation data only (target selective accuracy 90%, minimum coverage 10%; both can be changed with flags). The pipeline checks PyTorch and ONNX logits numerically. A new run is saved for review and is not activated by default; use `--activate` only after reviewing its test metrics and licensing. The existing five-class demo model and PlantDoc use incompatible class taxonomies, so their overall metrics cannot be directly compared.

```powershell
python .\scripts\evaluate_leaf_model.py
```

Start or restart FarmIQ from the repository root and open the Leaf Scanner at `/disease`:

```powershell
.\run.ps1
```

Sign in, open **http://127.0.0.1:5000/disease**, choose a JPG, PNG, or WEBP leaf image, and submit it. For the automated regression checks, run from the application folder:

```powershell
python -m unittest discover -s .\tests -p 'test_*.py' -v
```

The PlantDoc prediction is a model class, not a field diagnosis. A validation temperature-scaled score is still not a field probability. The scanner rejects scores below its validation-selected threshold and flags a possible blur/low-detail photo with a simple, unvalidated image heuristic. It has no out-of-distribution detector, so an unsupported plant can still receive an in-range class. PlantDoc has not been validated for Tamil Nadu field performance. Only maize rust and leaf blight classes with matching TNAU sources receive specific symptom notes; other classes show general expert-confirmation guidance. No pesticide is selected from a prediction. Nearby-help search opens Google Maps only after the user enters a place or explicitly shares browser location; FarmIQ does not store those coordinates or fabricate local listings.

The [PlantDoc model card](docs/plantdoc-model-card.md) records the intended use, current training status, supported-class policy, licensing notes, and limitations. No PlantDoc download or training metrics are included in this source tree. Run the commands above in an environment with network access before relying on the newly trained PlantDoc model.

## Features and data limits

- Farm profile for Tamil Nadu district, village, crop, area, soil type, irrigation and sowing date. Browser location is requested only after **Use My Location**; coordinates are saved only after profile consent. Weather requests send decimal coordinates to AgroMonitoring only after form submission.
- Soil report history accepts partial values and labels laboratory-reported versus user-entered data. The app does not estimate soil values from location. TNAU nutrient bands are screening interpretations; there are no fertilizer dosage calculations. A reliable regional soil-data provider is not configured.
- Crop calendar compares a curated set of district, month and water windows with source links. Preliminary categories remain visibly preliminary. Soil and crop-stage thresholds are not sufficiently encoded for an agronomic ranking; rainfall is not inferred. Read the [TNAU Crop Production Guide](https://tnau.ac.in/site/research/wp-content/uploads/sites/60/2020/02/Agriculture-CPG-2020.pdf) and crop-specific [TNAU rice](https://www.agritech.tnau.ac.in/agriculture/agri_seasonandvarieties_rice.html), [groundnut](https://www.agritech.tnau.ac.in/agriculture/agri_seasonandvarieties_groundnut.html), [maize](https://agritech.tnau.ac.in/agriculture/agri_maize_season_varieties.html), [redgram](https://agritech.tnau.ac.in/agriculture/CropProduction/Pulses/pulses_redgram-2.html), and [cotton](https://www.agritech.tnau.ac.in/agriculture/agri_seasonandvarieties_cotton.html) guidance before planting.
- Weather uses AgroMonitoring's current-weather and forecast endpoints. It displays the returned current conditions and timestamped forecast intervals; it does not infer daily forecasts or rain probabilities. The API key is loaded from the project `.env` file and `.env` is excluded from Git. Stored forecasts are scoped to the user's consented farm coordinates and marked stale after six hours.
- Live mandi-price lookup uses the official [data.gov.in AGMARKNET daily price dataset](https://www.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi). It requires a user-supplied `DATA_GOV_IN_API_KEY` environment variable. Without one, live lookup is unavailable. Cached prices older than 24 hours are marked stale; imported CSV rows are visibly separated and never represented as live. API availability and field names can change.
- The original relative-range demo and CSV import remain available; demo scores and imported prices are labeled as such.

For a market API key, create a key through [data.gov.in](https://data.gov.in/help). Set `DATA_GOV_IN_API_KEY` in the environment used to start the app. Avoid putting the key in source files or terminal command history. A key is only used when you submit a live market request.

For weather, create an AgroMonitoring API key and copy `.env.example` to `.env` from the repository root. Set `AGROMONITORING_API_KEY` in that local `.env` file, then restart FarmIQ. The `.env` file is ignored by Git. Current weather uses `units=metric` and both current and forecast requests are sent from the server; the key is never sent to the browser.

## Verification

Run the focused test suite from the repository root:

```powershell
python -m unittest discover -s .\tests -p 'test_*.py' -v
```

The local database is `farm_assistant.db`. Private scan images are stored under `instance\uploads`; the Flask static route does not serve them. The app binds to `127.0.0.1:5000` by default. Optional `PORT` can change the port. Do not enable Flask debug mode on a network-facing deployment.
