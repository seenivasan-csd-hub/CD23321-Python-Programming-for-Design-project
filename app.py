import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, abort, flash, g, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from services import db as database
from services.crop_advisory import KNOWLEDGE_PATH, load_knowledge, recommend as recommend_regional_crops
from services.disease_model import (
    LEGACY_EXPECTED_CROP, SUPPORTED_CROPS, ModelCompatibilityError, ModelUnavailable,
    assess_image_quality,
    model_status, predict as predict_disease,
)
from services.disease_guidance import guidance_for_prediction
from services.providers.market import MarketUnavailable, SOURCE_NAME as MARKET_SOURCE, SOURCE_URL as MARKET_SOURCE_URL, fetch_prices
from services.providers.soil import UnconfiguredRegionalSoilProvider
from services.providers.weather import (
    SOURCE_NAME as WEATHER_SOURCE, SOURCE_URL as WEATHER_SOURCE_URL,
    FORECAST_SOURCE_URL as WEATHER_FORECAST_SOURCE_URL, WeatherUnavailable,
    fetch_forecast, is_configured as weather_api_configured,
)
from services.soil import (
    FIELD_RULES, SoilValidationError, interpret_soil_record, missing_soil_fields,
    source_label, validate_soil_test,
)
from services.translations import TRANSLATIONS


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "farm_assistant.db"
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = INSTANCE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DISTRICTS = json.loads((BASE_DIR / "data" / "districts.json").read_text(encoding="utf-8"))
CROP_KNOWLEDGE = load_knowledge()
SOIL_PROVIDER = UnconfiguredRegionalSoilProvider()
PROFILE_CROPS = [item["name"] for item in CROP_KNOWLEDGE["crops"]]
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=MAX_IMAGE_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SESSION_REFRESH_EACH_REQUEST=False,
)
DUMMY_PASSWORD_HASH = generate_password_hash("FarmIQ dummy password used for timing parity")
PUBLIC_ENDPOINTS = {"static", "login", "register", "health"}
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
DECIMAL_COORDINATE_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    if "db" not in g:
        g.db = database.connect(DB_PATH)
    return g.db


def close_db(_error=None):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


app.teardown_appcontext(close_db)


def init_db():
    database.initialize(DB_PATH, BASE_DIR / "schema.sql")


def _csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _claim_legacy_data(user_id):
    """Give previously unowned local records to the first account, never later signups."""
    for table in (
        "farms", "soil_records", "crop_recommendations", "disease_scans",
        "weather_records", "market_prices", "farm_profiles", "soil_test_records",
        "crop_advice_history",
    ):
        get_db().execute(f"UPDATE {table} SET user_id=? WHERE user_id IS NULL", (user_id,))
    preference = get_db().execute(
        "SELECT language FROM user_preferences WHERE preference_id=1"
    ).fetchone()
    if preference and preference["language"] in TRANSLATIONS:
        get_db().execute("UPDATE users SET language=? WHERE user_id=?", (preference["language"], user_id))


def _safe_next_url(value):
    if not value:
        return url_for("index")
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or "\\" in value or not value.startswith("/") or value.startswith("//"):
        return url_for("index")
    return value


@app.before_request
def protect_requests_and_set_language():
    if request.path.startswith("/static/uploads/"):
        abort(404)

    g.current_user = None
    user_id = session.get("user_id")
    if user_id is not None:
        row = get_db().execute(
            "SELECT user_id,full_name,email,language FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if row:
            g.current_user = dict(row)
        else:
            session.clear()

    if request.endpoint not in PUBLIC_ENDPOINTS and g.current_user is None:
        next_path = request.full_path if request.query_string else request.path
        return redirect(url_for("login", next=next_path))

    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        expected = session.get("_csrf_token", "")
        if not expected or not secrets.compare_digest(submitted, expected):
            abort(400, description="The form expired. Reload the page and try again.")

    lang = request.args.get("lang")
    if lang in TRANSLATIONS:
        session["farm_lang"] = lang
        if g.current_user:
            get_db().execute(
                "UPDATE users SET language=? WHERE user_id=?", (lang, g.current_user["user_id"])
            )
            get_db().commit()
            g.current_user["language"] = lang
    default_language = g.current_user["language"] if g.current_user else request.cookies.get("farm_lang", "en")
    g.lang = session.get("farm_lang", default_language)
    if g.lang not in TRANSLATIONS:
        g.lang = "en"


@app.context_processor
def inject_template_values():
    return {
        "t": TRANSLATIONS.get(getattr(g, "lang", "en"), TRANSLATIONS["en"]),
        "lang": getattr(g, "lang", "en"),
        "csrf_token": _csrf_token,
        "districts": DISTRICTS,
        "today": date.today().isoformat(),
        "current_user": getattr(g, "current_user", None),
    }


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "geolocation=(self), camera=(self)")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'",
    )
    response.set_cookie("farm_lang", getattr(g, "lang", "en"), httponly=True, samesite="Lax")
    return response


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return render_template("error.html", code=413, message="The upload is larger than the 5 MB limit."), 413


@app.errorhandler(400)
def bad_request(error):
    message = getattr(error, "description", "The request could not be processed.")
    return render_template("error.html", code=400, message=message), 400


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user:
        return redirect(url_for("index"))
    next_url = request.form.get("next") if request.method == "POST" else request.args.get("next")
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().casefold()
        password = request.form.get("password") or ""
        user = get_db().execute(
            "SELECT user_id,password_hash,language FROM users WHERE email=? COLLATE NOCASE", (email,)
        ).fetchone() if email and len(email) <= 254 else None
        password_ok = check_password_hash(user["password_hash"], password) if user and len(password) <= 256 else False
        if not user:
            check_password_hash(DUMMY_PASSWORD_HASH, password[:256])
        if not password_ok:
            flash("Email or password is incorrect.", "error")
        else:
            session.clear()
            session.permanent = True
            session["user_id"] = user["user_id"]
            session["farm_lang"] = user["language"] if user["language"] in TRANSLATIONS else "en"
            session["_csrf_token"] = secrets.token_urlsafe(32)
            return redirect(_safe_next_url(next_url))
    return render_template("login.html", next_url=next_url or "")


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.current_user:
        return redirect(url_for("index"))
    first_account = get_db().execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    form_data = {"full_name": "", "email": ""}
    if request.method == "POST":
        form_data = {
            "full_name": (request.form.get("full_name") or "").strip(),
            "email": (request.form.get("email") or "").strip().casefold(),
        }
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        try:
            if not 2 <= len(form_data["full_name"]) <= 100:
                raise ValueError("Enter your full name (2–100 characters).")
            if len(form_data["email"]) > 254 or not EMAIL_PATTERN.fullmatch(form_data["email"]):
                raise ValueError("Enter a valid email address.")
            if len(password) < 12 or len(password) > 256:
                raise ValueError("Choose a password between 12 and 256 characters.")
            if password != confirm_password:
                raise ValueError("The password and confirmation do not match.")
            if first_account and request.form.get("legacy_data_consent") != "yes":
                raise ValueError("Confirm that this first account should own existing local farm records.")

            connection = get_db()
            cursor = connection.execute(
                "INSERT INTO users(full_name,email,password_hash,language,created_at) VALUES (?,?,?,?,?)",
                (form_data["full_name"], form_data["email"], generate_password_hash(password), g.lang, utc_now()),
            )
            user_id = cursor.lastrowid
            if connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1:
                _claim_legacy_data(user_id)
            connection.commit()
            session.clear()
            session.permanent = True
            session["user_id"] = user_id
            session["farm_lang"] = g.lang
            session["_csrf_token"] = secrets.token_urlsafe(32)
            flash("Account created. Your FarmIQ workspace is ready.", "success")
            return redirect(url_for("index"))
        except sqlite3.IntegrityError:
            get_db().rollback()
            flash("An account with that email address already exists.", "error")
        except ValueError as exc:
            flash(str(exc), "error")
    return render_template("register.html", first_account=first_account, form_data=form_data)


@app.post("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


def get_profile():
    row = get_db().execute(
        "SELECT p.*, f.location AS farm_location FROM farm_profiles p "
        "LEFT JOIN farms f ON f.farm_id=p.farm_id WHERE p.user_id=?",
        (g.current_user["user_id"],),
    ).fetchone()
    return dict(row) if row else None


def parse_optional_float(value, label, minimum, maximum):
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
    return parsed


def parse_weather_coordinate(value, label, minimum, maximum):
    value = (value or "").strip()
    if not value:
        return None
    if not DECIMAL_COORDINATE_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must use decimal notation, for example 13.007696.")
    return parse_optional_float(value, label, minimum, maximum)


def validate_district(value):
    match = next((name for name in DISTRICTS if name.casefold() == (value or "").strip().casefold()), None)
    if not match:
        raise ValueError("Choose a district from the Tamil Nadu district list.")
    return match


def _water_mode(irrigation):
    if irrigation == "rainfed":
        return "rainfed"
    if irrigation in {"canal", "well", "borewell", "drip", "reliable"}:
        return "irrigated"
    return "unknown"


@app.route("/")
def index():
    user_id = g.current_user["user_id"]
    profile = get_profile()
    latest_soil = get_db().execute(
        "SELECT * FROM soil_test_records WHERE user_id=? ORDER BY recorded_at DESC, soil_test_id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    recent_advice = get_db().execute(
        "SELECT * FROM crop_advice_history WHERE user_id=? ORDER BY created_at DESC LIMIT 3", (user_id,)
    ).fetchall()
    legacy_recent = get_db().execute(
        "SELECT crop_name, score, created_at FROM crop_recommendations WHERE user_id=? ORDER BY recommendation_id DESC LIMIT 3",
        (user_id,),
    ).fetchall()
    market_rows = get_db().execute(
        "SELECT * FROM market_price_cache ORDER BY fetched_at DESC LIMIT 3"
    ).fetchall()
    weather_rows = _latest_weather_rows()
    return render_template(
        "index.html", profile=profile, latest_soil=dict(latest_soil) if latest_soil else None,
        soil_summary=interpret_soil_record(dict(latest_soil)) if latest_soil else None,
        recent_advice=[dict(row) for row in recent_advice], legacy_recent=[dict(row) for row in legacy_recent],
        market_rows=[dict(row) for row in market_rows], weather_rows=weather_rows,
        model_status=model_status(),
    )


@app.route("/farm", methods=["GET", "POST"])
def farm_profile():
    profile = get_profile()
    if request.method == "POST":
        try:
            if request.form.get("consent") != "yes":
                raise ValueError("Confirm that you consent to saving this farm profile on this device.")
            district = validate_district(request.form.get("district"))
            village = (request.form.get("village") or "").strip()
            if not village or len(village) > 120:
                raise ValueError("Enter a village or town name (up to 120 characters).")
            area = parse_optional_float(request.form.get("area_acres"), "Farm area", 0.01, 100000)
            lat = parse_optional_float(request.form.get("latitude"), "Latitude", -90, 90)
            lon = parse_optional_float(request.form.get("longitude"), "Longitude", -180, 180)
            if (lat is None) != (lon is None):
                raise ValueError("Enter both coordinates or leave both blank.")
            soil_type = (request.form.get("soil_type") or "").strip()
            allowed_soils = {"red", "black", "alluvial", "sandy", "clay", "loam", "laterite", "other", ""}
            if soil_type not in allowed_soils:
                raise ValueError("Choose a soil type from the list.")
            irrigation = request.form.get("irrigation_method", "unknown")
            if irrigation not in {"canal", "well", "borewell", "drip", "rainfed", "seasonal", "unknown"}:
                raise ValueError("Choose an irrigation option from the list.")
            crop = (request.form.get("crop") or "").strip()
            if crop and crop not in PROFILE_CROPS:
                raise ValueError("Choose a crop from the supported list.")
            sowing_date = (request.form.get("sowing_date") or "").strip() or None
            if sowing_date:
                parsed_sowing = date.fromisoformat(sowing_date)
                if parsed_sowing > date.today():
                    raise ValueError("Sowing date cannot be in the future.")
            location = f"{village}, {district}, Tamil Nadu"
            connection = get_db()
            if profile and profile.get("farm_id"):
                farm_id = profile["farm_id"]
                connection.execute(
                    "UPDATE farms SET location=?, area_acres=?, soil_type=? WHERE farm_id=? AND user_id=?",
                    (location, area, soil_type or None, farm_id, g.current_user["user_id"]),
                )
            else:
                cursor = connection.execute(
                    "INSERT INTO farms(location, area_acres, soil_type, user_id) VALUES (?, ?, ?, ?)",
                    (location, area, soil_type or None, g.current_user["user_id"]),
                )
                farm_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO farm_profiles(user_id, farm_id, district, village, latitude, longitude, crop, area_acres, soil_type, irrigation_method, sowing_date, consented_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET farm_id=excluded.farm_id, district=excluded.district, village=excluded.village, "
                "latitude=excluded.latitude, longitude=excluded.longitude, crop=excluded.crop, area_acres=excluded.area_acres, "
                "soil_type=excluded.soil_type, irrigation_method=excluded.irrigation_method, sowing_date=excluded.sowing_date, updated_at=excluded.updated_at",
                (g.current_user["user_id"], farm_id, district, village, lat, lon, crop or None, area,
                 soil_type or None, irrigation, sowing_date, utc_now(), utc_now()),
            )
            connection.commit()
            flash("Farm profile saved on this device with your consent.", "success")
            return redirect(url_for("farm_profile"))
        except (ValueError, OverflowError) as exc:
            flash(str(exc), "error")
    return render_template("farm.html", profile=profile, crops=PROFILE_CROPS)


@app.route("/farm/delete", methods=["POST"])
def delete_farm_profile():
    profile = get_profile()
    if not profile:
        flash("There is no saved farm profile to remove.", "info")
        return redirect(url_for("farm_profile"))
    connection = get_db()
    farm_id = profile.get("farm_id")
    if profile.get("latitude") is not None and profile.get("longitude") is not None:
        connection.execute(
            "DELETE FROM weather_records WHERE user_id=? AND latitude=? AND longitude=?",
            (g.current_user["user_id"], profile["latitude"], profile["longitude"]),
        )
    connection.execute("DELETE FROM farm_profiles WHERE profile_id=? AND user_id=?",
                       (profile["profile_id"], g.current_user["user_id"]))
    if farm_id:
        connection.execute("DELETE FROM farms WHERE farm_id=? AND user_id=?", (farm_id, g.current_user["user_id"]))
    connection.commit()
    flash("The saved profile and profile-linked history were removed from this device.", "success")
    return redirect(url_for("farm_profile"))


@app.route("/soil", methods=["GET", "POST"])
def soil_health():
    profile = get_profile()
    if request.method == "POST":
        try:
            if request.form.get("store_soil") != "yes":
                raise SoilValidationError("Confirm that you consent to storing this soil test on this device.")
            record = validate_soil_test(request.form)
            connection = get_db()
            connection.execute(
                "INSERT INTO soil_test_records(profile_id, nitrogen, phosphorus, potassium, soil_ph, electrical_conductivity, organic_carbon, moisture, phosphorus_method, source, lab_name, confidence_label, sample_date, notes, recorded_at, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile["profile_id"] if profile else None, record["nitrogen"], record["phosphorus"], record["potassium"], record["soil_ph"],
                 record["electrical_conductivity"], record["organic_carbon"], record["moisture"], record["phosphorus_method"],
                 record["source"], record["lab_name"], record["confidence_label"], record["sample_date"], record["notes"],
                 utc_now(), g.current_user["user_id"]),
            )
            connection.commit()
            flash("Soil test saved. The app will not generate fertilizer dosages from this record.", "success")
            return redirect(url_for("soil_health"))
        except SoilValidationError as exc:
            flash(str(exc), "error")
    rows = get_db().execute(
        "SELECT * FROM soil_test_records WHERE user_id=? ORDER BY recorded_at DESC, soil_test_id DESC LIMIT 30",
        (g.current_user["user_id"],),
    ).fetchall()
    records = [dict(row) for row in rows]
    latest = records[0] if records else None
    return render_template(
        "soil.html", profile=profile, records=records,
        latest_summary=interpret_soil_record(latest) if latest else None,
        latest_missing=missing_soil_fields(latest) if latest else list(label for key, (_a, _b, label) in FIELD_RULES.items()),
        source_label=source_label, regional_status=SOIL_PROVIDER.status_message,
    )


@app.route("/soil/<int:record_id>/delete", methods=["POST"])
def delete_soil_record(record_id):
    cursor = get_db().execute("DELETE FROM soil_test_records WHERE soil_test_id=? AND user_id=?",
                              (record_id, g.current_user["user_id"]))
    get_db().commit()
    flash("Soil-test record removed." if cursor.rowcount else "Soil-test record was not found.", "success")
    return redirect(url_for("soil_health"))


@app.route("/crops", methods=["GET", "POST"])
def crop_advice():
    profile = get_profile()
    query = {"district": profile["district"] if profile else "", "month": date.today().month,
             "water_availability": _water_mode(profile["irrigation_method"]) if profile else "unknown"}
    advice = None
    if request.method == "POST":
        try:
            query["district"] = validate_district(request.form.get("district"))
            query["month"] = int(request.form.get("month", ""))
            if not 1 <= query["month"] <= 12:
                raise ValueError("Choose a valid sowing month.")
            query["water_availability"] = request.form.get("water_availability", "unknown")
            if query["water_availability"] not in {"irrigated", "rainfed", "unknown"}:
                raise ValueError("Choose a valid water option.")
            query["soil_type"] = profile.get("soil_type") if profile else request.form.get("soil_type")
            advice = recommend_regional_crops(**query)
            if request.form.get("save_history") == "yes":
                if not profile:
                    flash("Save a farm profile first to store recommendations in farm history.", "info")
                else:
                    get_db().execute(
                        "INSERT INTO crop_advice_history(profile_id, district, sowing_month, water_availability, results_json, knowledge_version, created_at, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (profile["profile_id"], query["district"], query["month"], query["water_availability"],
                         json.dumps(advice["results"], ensure_ascii=False), CROP_KNOWLEDGE["version"],
                         utc_now(), g.current_user["user_id"]),
                    )
                    get_db().commit()
                    flash("This advisory was saved to the local farm history.", "success")
        except (ValueError, TypeError) as exc:
            flash(str(exc), "error")
    return render_template("crops.html", profile=profile, crops=PROFILE_CROPS, advice=advice, query=query)


# Keep the original heuristic page available, clearly labeled as a software demo.
LEGACY_CROP_PROFILES = {
    "Paddy": {"N": (40, 100), "P": (20, 60), "K": (20, 60), "ph": (5.0, 7.0), "temp": (22, 32), "humidity": (60, 95), "rain": (100, 300), "note": "Check irrigation access, variety, season, soil test and local paddy guidance."},
    "Groundnut": {"N": (10, 50), "P": (20, 80), "K": (20, 80), "ph": (5.5, 7.5), "temp": (22, 32), "humidity": (40, 75), "rain": (50, 150), "note": "Check soil drainage, sowing window, seed variety and local groundnut guidance."},
    "Tomato": {"N": (20, 80), "P": (20, 80), "K": (30, 100), "ph": (5.5, 7.5), "temp": (18, 30), "humidity": (40, 75), "rain": (30, 120), "note": "Check variety, disease pressure, drainage, staking and local tomato guidance."},
}


def range_fit(value, low, high):
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1 - distance / max(high - low, 1))


def recommend_crops(values):
    keys = [("N", "N"), ("P", "P"), ("K", "K"), ("ph", "ph"), ("temp", "temp"), ("humidity", "humidity"), ("rain", "rain")]
    results = []
    for crop, profile in LEGACY_CROP_PROFILES.items():
        fit = [range_fit(values[input_key], *profile[profile_key]) for input_key, profile_key in keys]
        results.append({"crop": crop, "score": round(100 * sum(fit) / len(fit)), "note": profile["note"],
                        "reason": "Relative range-fit to a demo profile; not validated crop suitability."})
    return sorted(results, key=lambda item: item["score"], reverse=True)


@app.route("/crops/demo", methods=["GET", "POST"])
def legacy_crop_demo():
    results = None
    if request.method == "POST":
        try:
            values = {"N": float(request.form["nitrogen"]), "P": float(request.form["phosphorus"]),
                      "K": float(request.form["potassium"]), "ph": float(request.form["soil_ph"]),
                      "temp": float(request.form["temperature"]), "humidity": float(request.form["humidity"]),
                      "rain": float(request.form["rainfall"])}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Enter finite numeric values.")
            if not 0 <= values["humidity"] <= 100 or not 0 <= values["ph"] <= 14:
                raise ValueError("Humidity must be 0–100 and pH must be 0–14.")
            results = recommend_crops(values)
            for item in results:
                get_db().execute(
                    "INSERT INTO crop_recommendations(farm_location, crop_name, score, explanation, created_at, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                    ((request.form.get("location") or "").strip()[:120], item["crop"], item["score"],
                     item["reason"] + " " + item["note"], utc_now(), g.current_user["user_id"]),
                )
            get_db().commit()
        except (KeyError, ValueError) as exc:
            flash(f"Please enter valid demo values. {exc}", "error")
    return render_template("crop_demo.html", results=results)


def _parse_market_date(raw):
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw.strip(), pattern).date().isoformat()
        except (ValueError, AttributeError):
            continue
    raise ValueError("Market date must be YYYY-MM-DD or DD/MM/YYYY.")


def _import_market_csv(upload, user_id):
    raw = upload.stream.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("CSV upload exceeds the 5 MB limit.")
    try:
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    except UnicodeDecodeError as exc:
        raise ValueError("Use a UTF-8 CSV file.") from exc
    required = {"date", "crop", "market", "variety", "modal_price"}
    if not rows.fieldnames or not required.issubset({name.strip() for name in rows.fieldnames if name}):
        raise ValueError("CSV needs date,crop,market,variety,modal_price headers.")
    cleaned = []
    for index, row in enumerate(rows):
        if index >= 1000:
            raise ValueError("CSV import is limited to 1,000 rows at a time.")
        crop = (row.get("crop") or "").strip()[:100]
        market = (row.get("market") or "").strip()[:100]
        variety = (row.get("variety") or "").strip()[:100]
        if not crop or not market:
            continue
        price = parse_optional_float(row.get("modal_price"), "Modal price", 0.01, 1000000000)
        if price is None:
            raise ValueError("Each imported market row must include a modal price.")
        cleaned.append((_parse_market_date(row.get("date") or ""), crop, market, variety, price))
    if not cleaned:
        raise ValueError("The CSV did not contain any valid price rows.")
    get_db().executemany(
        "INSERT INTO market_prices(date, crop_name, market_name, variety, modal_price, source, unit, source_url, fetched_at, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(d, crop, market, variety, price, "User-imported CSV; not live", "Unit not specified in CSV", None, utc_now())
         + (user_id,) for d, crop, market, variety, price in cleaned],
    )
    get_db().commit()
    return len(cleaned)


def _latest_cached_market(limit=50):
    rows = get_db().execute("SELECT * FROM market_price_cache ORDER BY fetched_at DESC, price_date DESC LIMIT ?", (limit,)).fetchall()
    output = [dict(row) for row in rows]
    for item in output:
        item["stale"] = _is_stale(item.get("fetched_at"), hours=24)
    return output


def _is_stale(timestamp, hours):
    if not timestamp:
        return True
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() > hours * 3600
    except (ValueError, TypeError):
        return True


@app.route("/market", methods=["GET", "POST"])
def market():
    live_requested = False
    live_prices = []
    live_fetched_at = None
    live_status = "No live request has been made."
    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "import_csv":
                uploaded = request.files.get("price_csv")
                if not uploaded or not uploaded.filename:
                    raise ValueError("Choose a CSV file first.")
                count = _import_market_csv(uploaded, g.current_user["user_id"])
                flash(f"Imported {count} user-supplied market rows. They are not labeled live.", "success")
            elif action == "fetch_live":
                live_requested = True
                commodity = (request.form.get("commodity") or "").strip()[:100]
                district = validate_district(request.form.get("district")) if request.form.get("district") else None
                market_name = (request.form.get("market") or "").strip()[:100] or None
                if not commodity:
                    raise ValueError("Enter a commodity name to query the official price provider.")
                payload = fetch_prices(commodity, district=district, market=market_name)
                live_prices = payload["prices"]
                live_fetched_at = payload["fetched_at"]
                if live_prices:
                    get_db().executemany(
                        "INSERT INTO market_price_cache(commodity,district,market,variety,price_date,min_price,max_price,modal_price,unit,source,source_url,fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(row["commodity"], row["district"], row["market"], row["variety"], row["date"], row["min_price"],
                          row["max_price"], row["modal_price"], row["unit"], MARKET_SOURCE, MARKET_SOURCE_URL, payload["fetched_at"])
                         for row in live_prices],
                    )
                    get_db().commit()
                    live_status = f"Fetched {len(live_prices)} records from the official source."
                else:
                    live_status = "The source returned no records for these filters. No sample prices were substituted."
            else:
                raise ValueError("Choose a market-data action.")
        except (ValueError, MarketUnavailable) as exc:
            live_status = str(exc)
            flash(str(exc), "error")
    imported = get_db().execute(
        "SELECT * FROM market_prices WHERE user_id=? ORDER BY date DESC, price_id DESC LIMIT 30",
        (g.current_user["user_id"],),
    ).fetchall()
    return render_template(
        "market.html", cached_prices=_latest_cached_market(), imported_prices=[dict(row) for row in imported],
        live_prices=live_prices, live_fetched_at=live_fetched_at, live_status=live_status, live_requested=live_requested,
        api_configured=bool(os.environ.get("DATA_GOV_IN_API_KEY", "").strip()),
        source_url=MARKET_SOURCE_URL,
    )


def _latest_weather_rows(latitude=None, longitude=None):
    clauses = ["fetched_at IS NOT NULL", "user_id=?"]
    params = [g.current_user["user_id"]]
    if latitude is not None and longitude is not None:
        clauses.extend(["latitude=?", "longitude=?"])
        params.extend([latitude, longitude])
    where = " AND ".join(clauses)
    latest = get_db().execute(f"SELECT MAX(fetched_at) AS fetched_at FROM weather_records WHERE {where}", params).fetchone()
    if not latest or not latest["fetched_at"]:
        return []
    rows = get_db().execute(
        f"SELECT * FROM weather_records WHERE {where} AND fetched_at=? ORDER BY forecast_date LIMIT 40",
        [*params, latest["fetched_at"]],
    ).fetchall()
    output = [dict(row) for row in rows]
    for item in output:
        item["stale"] = _is_stale(item.get("fetched_at"), hours=6)
    return output


@app.route("/weather", methods=["GET", "POST"])
def weather():
    profile = get_profile()
    forecast = None
    error_message = None
    latitude_value = profile.get("latitude") if profile else None
    longitude_value = profile.get("longitude") if profile else None
    cached = _latest_weather_rows(latitude_value, longitude_value) if latitude_value is not None else []
    if request.method == "GET" and not weather_api_configured():
        error_message = "AgroMonitoring is not configured. Add AGROMONITORING_API_KEY to the project's .env file and restart FarmIQ."
    if request.method == "POST":
        error_message = None
        try:
            latitude = parse_weather_coordinate(request.form.get("latitude"), "Latitude", -90, 90)
            longitude = parse_weather_coordinate(request.form.get("longitude"), "Longitude", -180, 180)
            if latitude is None or longitude is None:
                raise ValueError("Enter both decimal coordinates or use the location button before fetching weather.")
            latitude_value, longitude_value = latitude, longitude
            forecast = fetch_forecast(latitude, longitude)
            # Coordinates and forecasts are persisted only when they match the profile the user explicitly saved.
            if profile and profile.get("latitude") is not None and math.isclose(latitude, profile["latitude"], abs_tol=0.00001) and math.isclose(longitude, profile["longitude"], abs_tol=0.00001):
                current = forecast["current"]
                get_db().executemany(
                    "INSERT INTO weather_records(latitude,longitude,forecast_date,temp_max,temp_min,precipitation_mm,precipitation_probability,current_temperature,relative_humidity,wind_speed,weather_code,source,fetched_at,user_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(latitude, longitude, row["time"], row["temperature_c"], row["temperature_c"], row["rainfall_mm"], None,
                      current["temperature_c"], current.get("humidity"), current.get("wind_speed"),
                      row.get("weather_id"), WEATHER_SOURCE, forecast["fetched_at"], g.current_user["user_id"])
                     for row in forecast["periods"]],
                )
                get_db().commit()
                cached = _latest_weather_rows(latitude, longitude)
        except ValueError as exc:
            error_message = str(exc)
        except WeatherUnavailable as exc:
            error_message = str(exc)
            if profile and profile.get("latitude") is not None:
                cached = _latest_weather_rows(profile["latitude"], profile["longitude"])
    return render_template(
        "weather.html", profile=profile, forecast=forecast, cached=cached, error_message=error_message,
        latitude=latitude_value, longitude=longitude_value, source_url=WEATHER_SOURCE_URL,
        forecast_source_url=WEATHER_FORECAST_SOURCE_URL, weather_source=WEATHER_SOURCE,
    )



def _normalized_private_image(upload):
    if not upload or not upload.filename:
        raise ValueError("Choose or capture a leaf image first.")
    if Path(upload.filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Choose a JPG, PNG, or WEBP image.")
    raw = upload.stream.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 5 MB limit.")
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
        with Image.open(io.BytesIO(raw)) as check:
            if check.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("The file content is not a supported image format.")
            check.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            if min(opened.size) < 128:
                raise ValueError("Use a photo at least 128 pixels wide and high.")
            if opened.width * opened.height > 20_000_000:
                raise ValueError("Image dimensions exceed the 20 megapixel safety limit.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            normalized = output.getvalue()
    except ValueError:
        raise
    except Exception as exc:
        # Pillow's decoder errors vary by format; present one useful message and no parser details.
        raise ValueError("This image could not be decoded. Try a clear JPG, PNG, or WEBP photo.") from exc
    if not normalized or len(normalized) > MAX_IMAGE_BYTES:
        raise ValueError("Processed image exceeds the 5 MB limit.")
    filename = f"{secrets.token_hex(16)}.jpg"
    target = UPLOAD_DIR / filename
    target.write_bytes(normalized)
    return filename, target, hashlib.sha256(normalized).hexdigest()


@app.route("/disease", methods=["GET", "POST"])
def disease_scanner():
    message = None
    result = None
    status = model_status()
    if request.method == "POST":
        filename = None
        try:
            crop = (request.form.get("crop", "") or "").casefold()
            if status["kind"] == "legacy" and crop not in SUPPORTED_CROPS:
                raise ValueError("Choose a crop supported by the currently installed demo model.")
            filename, image_path, image_hash = _normalized_private_image(request.files.get("leaf_image"))
            quality = assess_image_quality(image_path)
            scan_status = "saved_no_model"
            prediction = None
            try:
                prediction = predict_disease(image_path, crop if status["kind"] == "legacy" else None)
                if quality["warning"]:
                    prediction = dict(prediction)
                    reasons = list(prediction.get("uncertainty_reasons") or [])
                    reasons.append(quality["warning"])
                    prediction.update({
                        "uncertain": True,
                        "label": "uncertain",
                        "class_name": "Uncertain result",
                        "uncertainty_reasons": reasons,
                    })
                scan_status = "uncertain" if prediction["uncertain"] else "predicted"
            except (ModelUnavailable, ModelCompatibilityError) as exc:
                message = str(exc)
            guidance = guidance_for_prediction(prediction)
            try:
                stored_path = str(image_path.relative_to(BASE_DIR))
            except ValueError:
                # In tests and packaged deployments the private upload directory can be external.
                # The deletion route still checks that every resolved path remains inside UPLOAD_DIR.
                stored_path = str(image_path.resolve())
            cursor = get_db().execute(
                "INSERT INTO disease_scans(image_filename,predicted_label,confidence,created_at,crop_name,image_path,model_version,scan_status,uncertain,input_sha256,source_url,user_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (filename, prediction["raw_label"] if prediction else None,
                 prediction["confidence"] if prediction else None, utc_now(),
                 (prediction.get("selected_crop") or prediction.get("predicted_crop") or prediction.get("crop")) if prediction else crop or None, stored_path,
                 prediction["model_version"] if prediction else None, scan_status,
                 int(prediction["uncertain"]) if prediction else 1, image_hash,
                 prediction["model_card_url"] if prediction else status["model_card_url"], g.current_user["user_id"]),
            )
            get_db().commit()
            result = {
                "prediction": prediction, "guidance": guidance, "scan_status": scan_status,
                "image_url": url_for("disease_scan_image", scan_id=cursor.lastrowid),
                "selected_crop": crop.title() if crop else None,
                "quality_warning": quality["warning"],
            }
            if prediction:
                message = "The model output is a candidate class, not a field diagnosis. Confirm uncertain results with an agricultural expert."
            else:
                message = message or "Image stored privately on this device. No disease prediction was made."
            flash(message, "success" if prediction and not prediction["uncertain"] else "info")
        except ValueError as exc:
            flash(str(exc), "error")
            if filename:
                (UPLOAD_DIR / filename).unlink(missing_ok=True)

    scans = get_db().execute(
        "SELECT scan_id,crop_name,predicted_label,confidence,created_at,model_version,scan_status,uncertain FROM disease_scans WHERE user_id=? ORDER BY scan_id DESC LIMIT 30",
        (g.current_user["user_id"],),
    ).fetchall()
    scan_history = []
    for row in scans:
        scan = dict(row)
        expected_crop = LEGACY_EXPECTED_CROP.get(scan.get("predicted_label"))
        selected_crop = str(scan.get("crop_name") or "").casefold()
        scan["predicted_crop"] = expected_crop.title() if expected_crop else None
        scan["crop_mismatch"] = bool(expected_crop and selected_crop and selected_crop != expected_crop)
        scan_history.append(scan)
    return render_template(
        "disease.html", status=status, result=result, scans=scan_history,
        supported_crops=status["supported_crops"], model_url=status["model_url"],
        model_license=status["license"], model_weight_size=f"{status['weight_size'] / (1024 * 1024):.1f} MB",
    )


@app.get("/disease/image/<int:scan_id>")
def disease_scan_image(scan_id):
    row = get_db().execute(
        "SELECT image_path FROM disease_scans WHERE scan_id=? AND user_id=?",
        (scan_id, g.current_user["user_id"]),
    ).fetchone()
    if not row or not row["image_path"]:
        abort(404)
    candidate = Path(row["image_path"])
    candidate = (candidate if candidate.is_absolute() else BASE_DIR / candidate).resolve()
    try:
        candidate.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        abort(404)
    if not candidate.is_file():
        abort(404)
    response = send_file(candidate, mimetype="image/jpeg", conditional=True)
    response.headers["Cache-Control"] = "private, no-store"
    return response


@app.route("/disease/<int:scan_id>/delete", methods=["POST"])
def delete_disease_scan(scan_id):
    row = get_db().execute("SELECT image_path,image_filename FROM disease_scans WHERE scan_id=? AND user_id=?",
                           (scan_id, g.current_user["user_id"])).fetchone()
    if not row:
        flash("Scan record was not found.", "error")
        return redirect(url_for("disease_scanner"))
    stored = row["image_path"]
    target = None
    if stored:
        candidate = (BASE_DIR / stored).resolve()
        try:
            candidate.relative_to(UPLOAD_DIR.resolve())
            target = candidate
        except ValueError:
            abort(400, description="The stored image path is invalid.")
    elif row["image_filename"]:
        # Remove legacy files only when the database value is a plain filename.
        safe_name = secure_filename(row["image_filename"])
        if safe_name == row["image_filename"] and Path(safe_name).name == safe_name:
            legacy = (BASE_DIR / "static" / "uploads" / safe_name).resolve()
            if legacy.parent == (BASE_DIR / "static" / "uploads").resolve():
                target = legacy
    if target and target.exists():
        target.unlink()
    get_db().execute("DELETE FROM disease_scans WHERE scan_id=? AND user_id=?",
                     (scan_id, g.current_user["user_id"]))
    get_db().commit()
    flash("Scan and its local image were deleted.", "success")
    return redirect(url_for("disease_scanner"))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "disease_model": model_status()["available"],
                    "market_provider_configured": bool(os.environ.get("DATA_GOV_IN_API_KEY", "").strip())})


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=os.environ.get("FLASK_DEBUG", "0") == "1")
