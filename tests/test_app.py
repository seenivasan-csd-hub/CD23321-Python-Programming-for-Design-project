import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
import requests
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import app as farm_app  # noqa: E402
from services import db as database  # noqa: E402
from services.crop_advisory import recommend  # noqa: E402
from services.disease_model import ModelUnavailable, model_status, predict  # noqa: E402
from services.providers.market import MarketUnavailable, fetch_prices  # noqa: E402
from services.providers import weather as weather_provider  # noqa: E402
from services.providers.weather import WeatherUnavailable, fetch_forecast  # noqa: E402
from services.soil import SoilValidationError, validate_soil_test  # noqa: E402


class FarmAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old_db_path = farm_app.DB_PATH
        self.old_upload_dir = farm_app.UPLOAD_DIR
        farm_app.DB_PATH = root / "test.db"
        farm_app.UPLOAD_DIR = root / "private-uploads"
        farm_app.UPLOAD_DIR.mkdir(parents=True)
        database.initialize(farm_app.DB_PATH, PROJECT_DIR / "schema.sql")
        farm_app.app.config.update(TESTING=True, SECRET_KEY="test-secret", MAX_CONTENT_LENGTH=5 * 1024 * 1024)
        self.client = farm_app.app.test_client()
        with self.client.session_transaction() as session:
            session["_csrf_token"] = "test-csrf-token"
        created = self.post("/register", {
            "full_name": "Primary Farm User", "email": "farmer@example.test",
            "password": "correct horse battery staple", "confirm_password": "correct horse battery staple",
            "legacy_data_consent": "yes",
        })
        if created.status_code != 302:
            raise AssertionError(f"Test account setup failed with HTTP {created.status_code}")
        connection = database.connect(farm_app.DB_PATH)
        try:
            self.user_id = connection.execute("SELECT user_id FROM users WHERE email=?", ("farmer@example.test",)).fetchone()[0]
        finally:
            connection.close()

    def tearDown(self):
        farm_app.DB_PATH = self.old_db_path
        farm_app.UPLOAD_DIR = self.old_upload_dir
        self.temp.cleanup()

    def post(self, url, data, client=None):
        client = client or self.client
        with client.session_transaction() as session:
            token = session.get("_csrf_token")
            if not token:
                token = "test-csrf-token"
                session["_csrf_token"] = token
        return client.post(url, data=data, headers={"X-CSRF-Token": token})

    def test_main_routes_render(self):
        for path in ("/", "/farm", "/soil", "/crops", "/crops/demo", "/disease", "/market", "/weather", "/health"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
        anonymous = farm_app.app.test_client()
        self.assertEqual(anonymous.get("/login").status_code, 200)
        self.assertEqual(anonymous.get("/register").status_code, 200)

    def test_leaf_scanner_offers_opt_in_agricultural_help(self):
        response = self.client.get("/disease")
        self.assertIn(b"Find nearby agricultural help", response.data)
        self.assertIn(b"data-help-geolocate", response.data)
        self.assertIn(b"TNAU ATIC source and services", response.data)
        self.assertIn(b"TNAU pathology department source", response.data)
        script = self.client.get("/static/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn(b"getCurrentPosition", script.data)
        self.assertIn(b"data-help-geolocate", script.data)
        script.close()

    def test_farmiq_titles_logo_and_favicon_are_local_and_consistent(self):
        expected_titles = {
            "/": "FarmIQ | Smart Agriculture", "/farm": "Farm Profile | FarmIQ",
            "/soil": "Soil Health | FarmIQ", "/crops": "Crop Advisor | FarmIQ",
            "/crops/demo": "Crop Demo | FarmIQ", "/disease": "Leaf Scanner | FarmIQ",
            "/market": "Market Prices | FarmIQ", "/weather": "Weather | FarmIQ",
        }
        for path, title in expected_titles.items():
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertIn(f"<title>{title}</title>".encode(), response.data)
                self.assertNotIn(b"Farm Decision Assistant", response.data)
        for asset in ("logo.svg", "favicon.svg"):
            with self.subTest(asset=asset):
                response = self.client.get(f"/static/{asset}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/svg+xml")
                self.assertIn(b"<svg", response.data)
                self.assertIn(b'viewBox="0 0 128 128"', response.data)
                self.assertNotIn(b"#f5f1df", response.data)
                self.assertNotIn(b"<rect", response.data)
                response.close()
        dashboard = self.client.get("/")
        self.assertIn(b'src="/static/logo.svg" alt=""', dashboard.data)
        self.assertIn(b'class="hero-mark" aria-hidden="true"><img src="/static/logo.svg" alt=""></div>', dashboard.data)
        self.assertNotIn("🌾".encode("utf-8"), dashboard.data)
        self.assertIn(b'href="/static/favicon.svg"', dashboard.data)
        for target in (b'href="/"', b'href="/farm"', b'href="/soil"', b'href="/crops"',
                       b'href="/weather"', b'href="/market"', b'href="/disease"', b"?lang=ta"):
            self.assertIn(target, dashboard.data)
        tamil_dashboard = self.client.get("/?lang=ta")
        self.assertIn(b'<html lang="ta">', tamil_dashboard.data)
        self.assertIn(b"FarmIQ", tamil_dashboard.data)

    def test_login_failed_then_successful_and_protected_route_redirect(self):
        client = farm_app.app.test_client()
        protected = client.get("/farm")
        self.assertEqual(protected.status_code, 302)
        self.assertIn("/login", protected.headers["Location"])
        failed = self.post("/login", {"email": "farmer@example.test", "password": "wrong password"}, client)
        self.assertEqual(failed.status_code, 200)
        self.assertIn(b"Email or password is incorrect", failed.data)
        with client.session_transaction() as session:
            self.assertNotIn("user_id", session)
        success = self.post("/login", {"email": "FARMER@example.test", "password": "correct horse battery staple"}, client)
        self.assertEqual(success.status_code, 302)
        self.assertEqual(client.get("/farm").status_code, 200)

    def test_registration_validation_duplicate_email_and_password_hashing(self):
        client = farm_app.app.test_client()
        invalid = self.post("/register", {
            "full_name": "New Farmer", "email": "new@example.test", "password": "short",
            "confirm_password": "different", "legacy_data_consent": "yes",
        }, client)
        self.assertEqual(invalid.status_code, 200)
        self.assertIn(b"between 12 and 256 characters", invalid.data)
        success = self.post("/register", {
            "full_name": "New Farmer", "email": "new@example.test",
            "password": "a very strong passphrase", "confirm_password": "a very strong passphrase",
        }, client)
        self.assertEqual(success.status_code, 302)
        self.assertEqual(client.get("/farm").status_code, 200)
        connection = database.connect(farm_app.DB_PATH)
        try:
            saved_hash = connection.execute("SELECT password_hash FROM users WHERE email=?", ("new@example.test",)).fetchone()[0]
            self.assertNotEqual(saved_hash, "a very strong passphrase")
            self.assertTrue(saved_hash.startswith("scrypt:"))
        finally:
            connection.close()
        duplicate_client = farm_app.app.test_client()
        duplicate = self.post("/register", {
            "full_name": "Duplicate Farmer", "email": "FARMER@example.test",
            "password": "another strong password", "confirm_password": "another strong password",
        }, duplicate_client)
        self.assertEqual(duplicate.status_code, 200)
        self.assertIn(b"already exists", duplicate.data)

    def test_logout_clears_session_and_reprotects_routes(self):
        signed_out = self.post("/logout", {})
        self.assertEqual(signed_out.status_code, 302)
        with self.client.session_transaction() as session:
            self.assertNotIn("user_id", session)
        protected = self.client.get("/farm")
        self.assertEqual(protected.status_code, 302)
        self.assertIn("/login", protected.headers["Location"])

    def test_database_migration_preserves_legacy_rows(self):
        legacy_path = Path(self.temp.name) / "legacy.db"
        connection = sqlite3.connect(legacy_path)
        connection.executescript("""
            CREATE TABLE farms(farm_id INTEGER PRIMARY KEY, farmer_name TEXT, location TEXT NOT NULL,
                area_acres REAL, soil_type TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE farm_profiles(profile_id INTEGER PRIMARY KEY CHECK(profile_id=1), farm_id INTEGER UNIQUE,
                district TEXT NOT NULL, village TEXT NOT NULL, latitude REAL, longitude REAL, crop TEXT,
                area_acres REAL, soil_type TEXT, irrigation_method TEXT, sowing_date TEXT,
                consented_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE soil_test_records(soil_test_id INTEGER PRIMARY KEY, profile_id INTEGER,
                nitrogen REAL, phosphorus REAL, potassium REAL, soil_ph REAL, electrical_conductivity REAL,
                organic_carbon REAL, moisture REAL, phosphorus_method TEXT, source TEXT NOT NULL,
                lab_name TEXT, confidence_label TEXT NOT NULL, sample_date TEXT, notes TEXT, recorded_at TEXT NOT NULL);
            CREATE TABLE crop_advice_history(advice_id INTEGER PRIMARY KEY, profile_id INTEGER, district TEXT NOT NULL,
                sowing_month INTEGER NOT NULL, water_availability TEXT NOT NULL, results_json TEXT NOT NULL,
                knowledge_version TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE disease_scans(scan_id INTEGER PRIMARY KEY, image_filename TEXT NOT NULL,
                predicted_label TEXT, confidence REAL, created_at TEXT NOT NULL);
            CREATE TABLE weather_records(weather_id INTEGER PRIMARY KEY, latitude REAL, longitude REAL,
                forecast_date TEXT, temp_max REAL, temp_min REAL, precipitation_mm REAL,
                precipitation_probability REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE market_prices(price_id INTEGER PRIMARY KEY, date TEXT NOT NULL, crop_name TEXT NOT NULL,
                market_name TEXT NOT NULL, variety TEXT, modal_price REAL NOT NULL,
                source TEXT DEFAULT 'user_import', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE crop_recommendations(recommendation_id INTEGER PRIMARY KEY, farm_location TEXT,
                crop_name TEXT NOT NULL, score INTEGER NOT NULL, explanation TEXT NOT NULL, created_at TEXT NOT NULL);
            INSERT INTO farms(farm_id,location,area_acres,soil_type) VALUES (7,'Old local village, Salem',2.5,'red');
            INSERT INTO farm_profiles(profile_id,farm_id,district,village,crop,area_acres,soil_type,irrigation_method,consented_at,updated_at)
                VALUES (1,7,'Salem','Old local village','Maize',2.5,'red','rainfed','2025-01-01','2025-01-01');
            INSERT INTO soil_test_records(soil_test_id,profile_id,nitrogen,phosphorus,potassium,soil_ph,source,confidence_label,recorded_at)
                VALUES (9,1,280,11,118,6.5,'user_entered','Legacy report','2025-01-01');
            INSERT INTO crop_advice_history(advice_id,profile_id,district,sowing_month,water_availability,results_json,knowledge_version,created_at)
                VALUES (4,1,'Salem',6,'rainfed','[]','legacy','2025-01-01');
            INSERT INTO crop_recommendations(recommendation_id,farm_location,crop_name,score,explanation,created_at)
                VALUES (3,'Salem','Paddy',75,'Legacy demo','2025-01-01');
            INSERT INTO disease_scans(image_filename,predicted_label,confidence,created_at)
                VALUES ('legacy.jpg','unanalysed',NULL,'2025-01-01');
        """)
        connection.commit()
        connection.close()
        database.initialize(legacy_path, PROJECT_DIR / "schema.sql")
        migrated = database.connect(legacy_path)
        try:
            row = migrated.execute("SELECT image_filename,scan_status,uncertain FROM disease_scans").fetchone()
            self.assertEqual(row["image_filename"], "legacy.jpg")
            self.assertEqual(row["scan_status"], "unanalysed")
            self.assertEqual(row["uncertain"], 1)
            self.assertIsNotNone(migrated.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='farm_profiles'").fetchone())
            profile = migrated.execute("SELECT profile_id,user_id,village FROM farm_profiles").fetchone()
            self.assertEqual((profile["profile_id"], profile["user_id"], profile["village"]), (1, None, "Old local village"))
            self.assertEqual(migrated.execute("SELECT soil_test_id FROM soil_test_records").fetchone()[0], 9)
            self.assertEqual(migrated.execute("SELECT advice_id FROM crop_advice_history").fetchone()[0], 4)
            self.assertEqual(migrated.execute("SELECT recommendation_id FROM crop_recommendations").fetchone()[0], 3)
            self.assertFalse(migrated.execute("PRAGMA foreign_key_check").fetchall())
        finally:
            migrated.close()

    def test_first_account_claims_preserved_unowned_local_records(self):
        legacy_path = Path(self.temp.name) / "unowned.db"
        original_path = farm_app.DB_PATH
        farm_app.DB_PATH = legacy_path
        database.initialize(legacy_path, PROJECT_DIR / "schema.sql")
        connection = database.connect(legacy_path)
        try:
            connection.execute("INSERT INTO farms(farm_id,location) VALUES (1,'Existing farm')")
            connection.execute(
                "INSERT INTO farm_profiles(profile_id,farm_id,district,village,consented_at,updated_at) VALUES (1,1,'Salem','Existing village','2025-01-01','2025-01-01')"
            )
            connection.execute(
                "INSERT INTO soil_test_records(profile_id,nitrogen,source,confidence_label,recorded_at) VALUES (1,280,'user_entered','Old local report','2025-01-01')"
            )
            connection.execute(
                "INSERT INTO crop_recommendations(farm_location,crop_name,score,explanation,created_at) VALUES ('Salem','Paddy',75,'Old local demo','2025-01-01')"
            )
            connection.commit()
        finally:
            connection.close()

        client = farm_app.app.test_client()
        created = self.post("/register", {
            "full_name": "Legacy Farm Owner", "email": "legacy-owner@example.test",
            "password": "a secure legacy passphrase", "confirm_password": "a secure legacy passphrase",
            "legacy_data_consent": "yes",
        }, client)
        self.assertEqual(created.status_code, 302)
        self.assertIn(b"Existing village", client.get("/farm").data)
        connection = database.connect(legacy_path)
        try:
            owner = connection.execute("SELECT user_id FROM users WHERE email=?", ("legacy-owner@example.test",)).fetchone()[0]
            self.assertEqual(connection.execute("SELECT user_id FROM farm_profiles WHERE profile_id=1").fetchone()[0], owner)
            self.assertEqual(connection.execute("SELECT user_id FROM soil_test_records").fetchone()[0], owner)
            self.assertEqual(connection.execute("SELECT user_id FROM crop_recommendations").fetchone()[0], owner)
        finally:
            connection.close()
            farm_app.DB_PATH = original_path

    def test_profile_requires_consent_and_saves_validated_tn_location(self):
        refused = self.post("/farm", {"district": "Salem", "village": "Example village"})
        self.assertEqual(refused.status_code, 200)
        connection = database.connect(farm_app.DB_PATH)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM farm_profiles WHERE user_id=?", (self.user_id,)).fetchone()[0], 0)
        finally:
            connection.close()

    def test_farm_and_soil_records_are_scoped_to_account(self):
        self.post("/farm", {
            "consent": "yes", "district": "Salem", "village": "Owner Field",
            "crop": "Maize", "irrigation_method": "rainfed",
        })
        self.post("/soil", {"store_soil": "yes", "nitrogen": "280"})
        connection = database.connect(farm_app.DB_PATH)
        try:
            soil_id = connection.execute("SELECT soil_test_id FROM soil_test_records WHERE user_id=?", (self.user_id,)).fetchone()[0]
        finally:
            connection.close()

        second = farm_app.app.test_client()
        created = self.post("/register", {
            "full_name": "Second Farmer", "email": "second@example.test",
            "password": "second strong passphrase", "confirm_password": "second strong passphrase",
        }, second)
        self.assertEqual(created.status_code, 302)
        dashboard = second.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertNotIn(b"Owner Field", dashboard.data)
        self.assertIn(b"No farm location is saved", dashboard.data)
        self.post(f"/soil/{soil_id}/delete", {}, second)
        connection = database.connect(farm_app.DB_PATH)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM soil_test_records WHERE soil_test_id=?", (soil_id,)).fetchone()[0], 1)
        finally:
            connection.close()
        saved = self.post("/farm", {
            "consent": "yes", "district": "Salem", "village": "Example village",
            "area_acres": "2.5", "crop": "Maize", "soil_type": "red",
            "irrigation_method": "rainfed", "latitude": "11.66", "longitude": "78.14",
        })
        self.assertEqual(saved.status_code, 302)
        connection = database.connect(farm_app.DB_PATH)
        try:
            profile = connection.execute("SELECT district,crop FROM farm_profiles WHERE user_id=?", (self.user_id,)).fetchone()
            self.assertEqual(profile["district"], "Salem")
            self.assertEqual(profile["crop"], "Maize")
        finally:
            connection.close()

    def test_soil_validation_rejects_nonfinite_and_labels_sources(self):
        with self.assertRaises(SoilValidationError):
            validate_soil_test({"nitrogen": "NaN"})
        record = validate_soil_test({"nitrogen": "280", "phosphorus": "11", "phosphorus_method": "olsen"})
        self.assertEqual(record["source"], "user_entered")
        self.assertEqual(record["confidence_label"], "User-entered; not independently verified")
        lab = validate_soil_test({"soil_ph": "6.5", "lab_confirmed": "yes", "lab_name": "District Soil Lab"})
        self.assertEqual(lab["source"], "laboratory")

    def test_soil_route_saves_partial_report_and_history(self):
        response = self.post("/soil", {"store_soil": "yes", "nitrogen": "280", "phosphorus": "11", "phosphorus_method": "olsen"})
        self.assertEqual(response.status_code, 302)
        page = self.client.get("/soil")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"User-entered soil test", page.data)
        self.assertIn(b"Unavailable", page.data)

    def test_crop_advisory_uses_documented_district_month_and_water_windows(self):
        irrigated = recommend("Salem", 6, "irrigated")
        maize = next(item for item in irrigated["results"] if item["crop"] == "Maize")
        self.assertTrue(maize["matched"])
        rainfed = recommend("Salem", 6, "rainfed")
        maize_rainfed = next(item for item in rainfed["results"] if item["crop"] == "Maize")
        self.assertFalse(maize_rainfed["matched"])
        response = self.post("/crops", {"district": "Salem", "month": "6", "water_availability": "irrigated"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Maize", response.data)

    def test_upload_validation_does_not_store_invalid_image(self):
        response = self.post("/disease", {
            "crop": "maize", "leaf_image": (io.BytesIO(b"not an image"), "leaf.jpg"),
        })
        self.assertEqual(response.status_code, 200)
        connection = database.connect(farm_app.DB_PATH)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM disease_scans").fetchone()[0], 0)
        finally:
            connection.close()
        self.assertEqual(list(farm_app.UPLOAD_DIR.iterdir()), [])

    def test_valid_scan_runs_local_model_and_delete_removes_private_image(self):
        from PIL import Image, ImageDraw

        if not model_status()["available"]:
            self.skipTest("Optional local model/runtime are not installed")
        image = Image.new("RGB", (240, 200), "#7cae56")
        ImageDraw.Draw(image).ellipse((25, 20, 215, 180), fill="#347c48", outline="#e4d794", width=4)
        content = io.BytesIO()
        image.save(content, format="PNG")
        response = self.post("/disease", {
            "crop": "maize", "leaf_image": (io.BytesIO(content.getvalue()), "leaf.png", "image/png"),
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Uncalibrated", response.data)
        connection = database.connect(farm_app.DB_PATH)
        try:
            scan = connection.execute("SELECT scan_id,image_path,predicted_label FROM disease_scans").fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(scan["predicted_label"])
        stored_image = farm_app.PROJECT_DIR / scan["image_path"] if hasattr(farm_app, "PROJECT_DIR") else PROJECT_DIR / scan["image_path"]
        self.assertTrue(stored_image.is_file())
        owned_photo = self.client.get(f"/disease/image/{scan['scan_id']}")
        self.assertEqual(owned_photo.status_code, 200)
        self.assertEqual(owned_photo.mimetype, "image/jpeg")
        self.assertIn("no-store", owned_photo.headers.get("Cache-Control", ""))
        owned_photo.close()

        other_client = farm_app.app.test_client()
        registered = self.post("/register", {
            "full_name": "Another Farmer", "email": "another@example.test",
            "password": "correct horse battery staple", "confirm_password": "correct horse battery staple",
        }, client=other_client)
        self.assertEqual(registered.status_code, 302)
        self.assertEqual(other_client.get(f"/disease/image/{scan['scan_id']}").status_code, 404)

        private_url = "/static/uploads/" + stored_image.name
        self.assertEqual(self.client.get(private_url).status_code, 404)
        deleted = self.post(f"/disease/{scan['scan_id']}/delete", {})
        self.assertEqual(deleted.status_code, 302)
        self.assertFalse(stored_image.exists())

    def test_plantdoc_scanner_uses_model_classes_without_a_crop_selector(self):
        from PIL import Image

        content = io.BytesIO()
        Image.new("RGB", (240, 200), "#517c48").save(content, format="PNG")
        plantdoc_status = {
            "available": True, "kind": "plantdoc", "message": "PlantDoc test model ready.",
            "name": "PlantDoc test model", "classes": ["Corn rust leaf", "Tomato leaf"],
            "supported_crops": [], "model_url": "https://github.com/pratikkayal/PlantDoc-Dataset",
            "license": "CC BY 4.0", "weight_size": 1024,
        }
        prediction = {
            "raw_label": "Corn rust leaf", "class_name": "Corn rust leaf", "label": "Corn rust leaf",
            "crop": "Corn", "predicted_crop": "Corn", "confidence": 0.42, "uncertain": True,
            "score_label": "Uncalibrated model score; not a field probability",
            "model_version": "plantdoc@test", "model_card_url": plantdoc_status["model_url"], "model_kind": "plantdoc",
        }
        with patch.object(farm_app, "model_status", return_value=plantdoc_status), patch.object(
            farm_app, "predict_disease", return_value=prediction
        ):
            response = self.post("/disease", {
                "leaf_image": (io.BytesIO(content.getvalue()), "corn.png", "image/png"),
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Corn rust leaf", response.data)
        self.assertIn(b"Model crop class:</strong> Corn", response.data)
        self.assertIn(b"Retake a clear image and seek expert confirmation", response.data)
        connection = database.connect(farm_app.DB_PATH)
        try:
            row = connection.execute("SELECT crop_name,model_version,predicted_label FROM disease_scans").fetchone()
            self.assertEqual(tuple(row), ("Corn", "plantdoc@test", "Corn rust leaf"))
        finally:
            connection.close()

    def test_bean_selected_cassava_model_result_is_shown_as_uncertain(self):
        from PIL import Image

        content = io.BytesIO()
        Image.new("RGB", (240, 200), "#517c48").save(content, format="PNG")
        legacy_status = {
            "available": True, "kind": "legacy", "message": "Legacy research model; results uncertain.",
            "name": "Local demo classifier", "classes": ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"],
            "supported_crops": ["maize", "cassava", "bean"], "model_url": "https://huggingface.co/Gopina/maize-disease-classifier",
            "license": "MIT", "weight_size": 1024,
        }
        prediction = {
            "raw_label": "cassava_mosaic", "class_name": "Uncertain result", "label": "uncertain",
            "crop": "Cassava", "predicted_crop": "Cassava", "selected_crop": "Bean", "crop_mismatch": True,
            "confidence": 0.9996, "uncertain": True, "uncertainty_reasons": [
                "The selected crop is Bean, but the top class maps to Cassava.",
                "This demo model has no local validation calibration or out-of-distribution check.",
            ], "top_predictions": [], "score_label": "Uncalibrated raw model score; not a probability",
            "model_version": "legacy@test", "model_card_url": legacy_status["model_url"], "model_kind": "legacy",
        }
        with patch.object(farm_app, "model_status", return_value=legacy_status), patch.object(
            farm_app, "predict_disease", return_value=prediction
        ):
            response = self.post("/disease", {
                "crop": "bean", "leaf_image": (io.BytesIO(content.getvalue()), "bean.png", "image/png"),
            })

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Uncertain result", response.data)
        self.assertIn(b"selected crop is Bean", response.data)
        self.assertIn(b"maps to Cassava", response.data)
        self.assertIn(b"0.9996", response.data)
        self.assertIn(b"not a probability", response.data)
        self.assertIn(b"Model class maps to Cassava; selected crop was Bean", response.data)
        self.assertIn(b"TNAU Agricultural Technology Information Centre", response.data)
        self.assertNotIn(b"Bean angular leaf spot pattern", response.data)

    def test_low_quality_warning_overrides_a_high_score_as_uncertain(self):
        from PIL import Image

        content = io.BytesIO()
        Image.new("RGB", (240, 200), "#517c48").save(content, format="PNG")
        status = {
            "available": True, "kind": "legacy", "message": "Demo model.", "name": "Demo",
            "classes": ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"],
            "supported_crops": ["maize", "cassava", "bean"], "model_url": "https://example.test/model",
            "license": "MIT", "weight_size": 1,
        }
        prediction = {
            "raw_label": "maize_rust", "class_name": "maize_rust", "label": "maize_rust",
            "crop": "Maize", "predicted_crop": "Maize", "selected_crop": "Maize", "crop_mismatch": False,
            "confidence": 0.999, "uncertain": False, "uncertainty_reasons": [], "top_predictions": [],
            "score_label": "Uncalibrated raw model score; not a probability", "model_version": "test",
            "model_card_url": status["model_url"], "model_kind": "legacy",
        }
        warning = "The photo may be blurry or lack visible leaf detail. Retake a sharp photo; this check is only a heuristic."
        with patch.object(farm_app, "model_status", return_value=status), patch.object(
            farm_app, "predict_disease", return_value=prediction
        ), patch.object(farm_app, "assess_image_quality", return_value={"detail_score": 1.2, "warning": warning}):
            response = self.post("/disease", {
                "crop": "maize", "leaf_image": (io.BytesIO(content.getvalue()), "blurred.png", "image/png"),
            })

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Photo quality check", response.data)
        self.assertIn(b"may be blurry", response.data)
        self.assertIn(b"Uncertain result", response.data)
        self.assertNotIn(b"Maize common rust pattern", response.data)

    def test_scanner_never_claims_prediction_when_weights_are_missing(self):
        root = Path(self.temp.name) / "missing-model"
        self.assertFalse(model_status(root / "model.onnx", root / "classes.json")["available"])
        with self.assertRaises(ModelUnavailable):
            predict(root / "leaf.jpg", "maize", root / "model.onnx", root / "classes.json")

    def test_weather_provider_calls_agromonitoring_and_parses_documented_responses(self):
        current_payload = {
            "dt": 1792934400,
            "weather": [{"id": 500, "description": "light rain", "icon": "10d"}],
            "main": {"temp": 27.5, "humidity": 78},
            "wind": {"speed": 2.4}, "clouds": {"all": 85}, "rain": {"3h": 1.2},
        }
        forecast_payload = [{
            "dt": 1792945200,
            "weather": [{"id": 801, "description": "few clouds", "icon": "02d"}],
            "main": {"temp": 28.1, "humidity": 71},
            "wind": {"speed": 2.0}, "clouds": {"all": 20},
        }]

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Client:
            def __init__(self):
                self.calls = []
                self.responses = [Response(current_payload), Response(forecast_payload)]

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return self.responses.pop(0)

        client = Client()
        result = fetch_forecast(13.007696, 79.995269, http=client, api_key="private-weather-test-key")
        self.assertEqual([call[0] for call in client.calls], [
            "https://api.agromonitoring.com/agro/1.0/weather",
            "https://api.agromonitoring.com/agro/1.0/weather/forecast",
        ])
        for _url, kwargs in client.calls:
            self.assertEqual(kwargs["params"]["lat"], 13.007696)
            self.assertEqual(kwargs["params"]["lon"], 79.995269)
            self.assertEqual(kwargs["params"]["units"], "metric")
            self.assertEqual(kwargs["params"]["appid"], "private-weather-test-key")
            self.assertEqual(kwargs["timeout"], 12)
        self.assertEqual(result["current"]["temperature_c"], 27.5)
        self.assertEqual(result["current"]["description"], "light rain")
        self.assertEqual(result["current"]["humidity"], 78)
        self.assertEqual(result["current"]["wind_speed"], 2.4)
        self.assertEqual(result["current"]["cloud_cover"], 85)
        self.assertEqual(result["current"]["rainfall_mm"], 1.2)
        self.assertEqual(result["periods"][0]["temperature_c"], 28.1)
        self.assertEqual(result["periods"][0]["rainfall_mm"], None)
        self.assertIn("UTC", result["periods"][0]["time"])
        self.assertTrue(result["current"]["icon_url"].endswith("10d@2x.png"))
        self.assertIn("forecast_source_url", result)
        self.assertNotIn("private-weather-test-key", repr(result))

    def test_weather_provider_missing_key_bad_coordinates_and_missing_fields(self):
        class ForbiddenClient:
            def get(self, *_args, **_kwargs):
                self.fail("Invalid setup must not trigger a weather request")

        with self.assertRaises(WeatherUnavailable) as missing_key:
            fetch_forecast(13.0, 79.0, http=ForbiddenClient(), api_key="")
        self.assertIn("AGROMONITORING_API_KEY", str(missing_key.exception))
        with self.assertRaises(WeatherUnavailable) as bad_coordinate:
            fetch_forecast(13.0, 181.0, http=ForbiddenClient(), api_key="test-key")
        self.assertIn("Longitude", str(bad_coordinate.exception))

        class MissingTemperatureResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"weather": [{"description": "clouds"}], "main": {"humidity": 80}}

        class OneResponseClient:
            def get(self, *_args, **_kwargs):
                return MissingTemperatureResponse()

        with self.assertLogs("services.providers.weather", level="WARNING"):
            with self.assertRaises(WeatherUnavailable) as missing_field:
                fetch_forecast(13.0, 79.0, http=OneResponseClient(), api_key="test-key")
        self.assertIn("temperature", str(missing_field.exception).lower())

    def test_weather_key_loads_from_dotenv_and_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            provider_file = project / "services" / "providers" / "weather.py"
            provider_file.parent.mkdir(parents=True)
            (project / ".env").write_text("AGROMONITORING_API_KEY='dotenv-test-key'\n", encoding="utf-8")
            with patch.object(weather_provider, "__file__", str(provider_file)), \
                    patch.dict(os.environ, {"AGROMONITORING_API_KEY": ""}):
                self.assertEqual(weather_provider._api_key(), "dotenv-test-key")
            with patch.object(weather_provider, "__file__", str(provider_file)), \
                    patch.dict(os.environ, {"AGROMONITORING_API_KEY": "process-test-key"}):
                self.assertEqual(weather_provider._api_key(), "process-test-key")

        example = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")
        gitignore = (PROJECT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("AGROMONITORING_API_KEY=", example)
        self.assertIn(".env", gitignore)

    def test_weather_provider_handles_http_network_and_json_errors_without_secrets(self):
        secret = "private-weather-test-key"

        class HttpErrorResponse:
            def __init__(self, status_code):
                self.status_code = status_code

            def raise_for_status(self):
                raise requests.HTTPError(f"failed URL appid={secret}", response=self)

        class ErrorClient:
            def __init__(self, error):
                self.error = error

            def get(self, *_args, **_kwargs):
                raise self.error

        for status, message in ((401, "rejected"), (403, "denied"), (429, "rate limit"), (503, "HTTP 503")):
            with self.subTest(status=status):
                response = HttpErrorResponse(status)
                class StatusClient:
                    def get(self, *_args, **_kwargs):
                        return response

                with self.assertLogs("services.providers.weather", level="WARNING") as logs:
                    with self.assertRaises(WeatherUnavailable) as failed:
                        fetch_forecast(13.0, 79.0, http=StatusClient(), api_key=secret)
                self.assertIn(message, str(failed.exception))
                self.assertNotIn(secret, str(failed.exception))
                self.assertNotIn(secret, "\n".join(logs.output))

        for error in (requests.Timeout(f"appid={secret}"), requests.ConnectionError(f"appid={secret}")):
            with self.subTest(error=type(error).__name__):
                with self.assertLogs("services.providers.weather", level="WARNING") as logs:
                    with self.assertRaises(WeatherUnavailable) as failed:
                        fetch_forecast(13.0, 79.0, http=ErrorClient(error), api_key=secret)
                self.assertNotIn(secret, str(failed.exception))
                self.assertNotIn(secret, "\n".join(logs.output))

        class InvalidJsonResponse:
            def raise_for_status(self):
                return None

            def json(self):
                raise ValueError(f"invalid JSON appid={secret}")

        class InvalidJsonClient:
            def get(self, *_args, **_kwargs):
                return InvalidJsonResponse()

        with self.assertLogs("services.providers.weather", level="WARNING") as logs:
            with self.assertRaises(WeatherUnavailable) as invalid_json:
                fetch_forecast(13.0, 79.0, http=InvalidJsonClient(), api_key=secret)
        self.assertIn("unreadable response", str(invalid_json.exception))
        self.assertNotIn(secret, "\n".join(logs.output))

    def test_weather_route_decimal_validation_loading_state_and_successful_fetch(self):
        page = self.client.get("/weather")
        self.assertIn(b'data-weather-form', page.data)
        self.assertIn(b'data-weather-submit', page.data)
        script = self.client.get("/static/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn(b"Loading current weather and forecast from AgroMonitoring", script.data)
        script.close()

        with patch.object(farm_app, "fetch_forecast") as fetch:
            invalid = self.post("/weather", {"latitude": "13 007696", "longitude": "79.995269"})
        self.assertEqual(invalid.status_code, 200)
        self.assertIn(b"Latitude must use decimal notation", invalid.data)
        fetch.assert_not_called()

        result = {
            "current": {"time": "2026-10-25 12:00 UTC", "temperature_c": 29.5,
                        "description": "scattered clouds", "icon_url": "https://openweathermap.org/img/wn/03d@2x.png",
                        "humidity": 62, "wind_speed": 1.8, "cloud_cover": 40,
                        "rainfall_mm": None, "rainfall_period": None},
            "periods": [{"time": "2026-10-25 15:00 UTC", "temperature_c": 30.0,
                         "description": "scattered clouds", "icon_url": None,
                         "humidity": 55, "wind_speed": 2.1, "cloud_cover": 35,
                         "rainfall_mm": 0.4, "rainfall_period": "3h"}],
            "fetched_at": "2026-10-25T12:00:00+00:00",
            "source": "AgroMonitoring Weather API",
            "source_url": "https://agromonitoring.com/api/current-weather",
            "forecast_source_url": "https://agromonitoring.com/api/forecast-weather",
        }
        self.post("/farm", {
            "consent": "yes", "district": "Salem", "village": "Weather test field",
            "crop": "Maize", "irrigation_method": "rainfed",
            "latitude": "13.007696", "longitude": "79.995269",
        })
        with patch.object(farm_app, "fetch_forecast", return_value=result) as fetch:
            response = self.post("/weather", {"latitude": "13.007696", "longitude": "79.995269"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"29.5&deg;C", response.data)
        self.assertIn(b"scattered clouds", response.data)
        self.assertIn(b"Forecast readings", response.data)
        self.assertIn(b"0.4 mm (3h)", response.data)
        self.assertIn(b"Stored forecast", response.data)
        fetch.assert_called_once_with(13.007696, 79.995269)
        connection = database.connect(farm_app.DB_PATH)
        try:
            stored = connection.execute(
                "SELECT forecast_date,temp_min,precipitation_mm,precipitation_probability,source FROM weather_records WHERE user_id=?",
                (self.user_id,),
            ).fetchone()
            self.assertEqual(stored["forecast_date"], "2026-10-25 15:00 UTC")
            self.assertEqual(stored["temp_min"], 30.0)
            self.assertEqual(stored["precipitation_mm"], 0.4)
            self.assertIsNone(stored["precipitation_probability"])
            self.assertEqual(stored["source"], "AgroMonitoring Weather API")
        finally:
            connection.close()

    def test_weather_route_shows_missing_key_and_retry_for_provider_failure(self):
        with patch.object(farm_app, "weather_api_configured", return_value=False):
            page = self.client.get("/weather")
        self.assertIn(b"AGROMONITORING_API_KEY", page.data)
        self.assertIn(b"Retry weather request", page.data)

        with patch.object(farm_app, "weather_api_configured", return_value=True), \
                patch.object(farm_app, "fetch_forecast", side_effect=WeatherUnavailable("AgroMonitoring rate limit reached. Wait briefly, then retry.")):
            failed = self.post("/weather", {"latitude": "13.007696", "longitude": "79.995269"})
        self.assertIn(b"rate limit reached", failed.data)
        self.assertIn(b"Retry weather request", failed.data)

    def test_market_provider_missing_key_and_network_errors_are_safe(self):
        class ForbiddenClient:
            def get(self, *_args, **_kwargs):
                self.fail("A missing API key must not trigger a network request")

        with self.assertRaises(MarketUnavailable) as missing:
            fetch_prices("Paddy", api_key="", http=ForbiddenClient())
        self.assertIn("not configured", str(missing.exception))

        class FailingClient:
            def get(self, *_args, **_kwargs):
                raise requests.ConnectionError("request failed with api-key=private-test-value")

        with self.assertRaises(MarketUnavailable) as failed:
            fetch_prices("Paddy", api_key="private-test-value", http=FailingClient())
        self.assertNotIn("private-test-value", str(failed.exception))

    def test_legacy_market_csv_import_is_labeled_and_requires_price(self):
        valid = self.post("/market", {
            "action": "import_csv",
            "price_csv": (io.BytesIO(b"date,crop,market,variety,modal_price\n2026-10-24,Paddy,Salem,Common,2350\n"), "prices.csv"),
        })
        self.assertEqual(valid.status_code, 200)
        self.assertIn(b"User-imported CSV; not live", valid.data)
        connection = database.connect(farm_app.DB_PATH)
        try:
            row = connection.execute("SELECT source,unit,modal_price FROM market_prices").fetchone()
            self.assertEqual(row["source"], "User-imported CSV; not live")
            self.assertEqual(row["modal_price"], 2350)
        finally:
            connection.close()
        invalid = self.post("/market", {
            "action": "import_csv",
            "price_csv": (io.BytesIO(b"date,crop,market,variety,modal_price\n2026-10-24,Paddy,Salem,Common,\n"), "blank-price.csv"),
        })
        self.assertEqual(invalid.status_code, 200)
        connection = database.connect(farm_app.DB_PATH)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM market_prices").fetchone()[0], 1)
        finally:
            connection.close()

    def test_post_requires_csrf_token(self):
        response = self.client.post("/farm", data={"district": "Salem", "village": "Example"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
