PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS farms (
    farm_id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer_name TEXT,
    location TEXT NOT NULL,
    area_acres REAL,
    soil_type TEXT,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS soil_records (
    soil_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    farm_id INTEGER,
    soil_ph REAL NOT NULL,
    nitrogen REAL NOT NULL,
    phosphorus REAL NOT NULL,
    potassium REAL NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (farm_id) REFERENCES farms(farm_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS crop_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    farm_location TEXT,
    crop_name TEXT NOT NULL,
    score INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS disease_scans (
    scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    predicted_label TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    crop_name TEXT,
    image_path TEXT,
    model_version TEXT,
    scan_status TEXT NOT NULL DEFAULT 'unanalysed',
    uncertain INTEGER NOT NULL DEFAULT 1,
    input_sha256 TEXT,
    source_url TEXT,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS weather_records (
    weather_id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL,
    longitude REAL,
    forecast_date TEXT,
    temp_max REAL,
    temp_min REAL,
    precipitation_mm REAL,
    precipitation_probability REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    current_temperature REAL,
    relative_humidity REAL,
    wind_speed REAL,
    weather_code INTEGER,
    source TEXT,
    fetched_at TEXT,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS market_prices (
    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    crop_name TEXT NOT NULL,
    market_name TEXT NOT NULL,
    variety TEXT,
    modal_price REAL NOT NULL,
    source TEXT DEFAULT 'user_import',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    unit TEXT,
    source_url TEXT,
    fetched_at TEXT,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS farm_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE REFERENCES users(user_id) ON DELETE CASCADE,
    farm_id INTEGER UNIQUE,
    district TEXT NOT NULL,
    village TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    crop TEXT,
    area_acres REAL,
    soil_type TEXT,
    irrigation_method TEXT,
    sowing_date TEXT,
    consented_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (farm_id) REFERENCES farms(farm_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS soil_test_records (
    soil_test_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER,
    nitrogen REAL,
    phosphorus REAL,
    potassium REAL,
    soil_ph REAL,
    electrical_conductivity REAL,
    organic_carbon REAL,
    moisture REAL,
    phosphorus_method TEXT,
    source TEXT NOT NULL,
    lab_name TEXT,
    confidence_label TEXT NOT NULL,
    sample_date TEXT,
    notes TEXT,
    recorded_at TEXT NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES farm_profiles(profile_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS crop_advice_history (
    advice_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER,
    district TEXT NOT NULL,
    sowing_month INTEGER NOT NULL,
    water_availability TEXT NOT NULL,
    results_json TEXT NOT NULL,
    knowledge_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES farm_profiles(profile_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS market_price_cache (
    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity TEXT NOT NULL,
    district TEXT,
    market TEXT,
    variety TEXT,
    price_date TEXT,
    min_price REAL,
    max_price REAL,
    modal_price REAL,
    unit TEXT,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id INTEGER PRIMARY KEY CHECK (preference_id = 1),
    language TEXT NOT NULL DEFAULT 'en',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_crop_recommendations_created
ON crop_recommendations(created_at);
CREATE INDEX IF NOT EXISTS idx_market_prices_crop_date
ON market_prices(crop_name, date);
CREATE INDEX IF NOT EXISTS idx_soil_test_records_recorded
ON soil_test_records(recorded_at);
CREATE INDEX IF NOT EXISTS idx_disease_scans_created
ON disease_scans(created_at);
CREATE INDEX IF NOT EXISTS idx_market_cache_fetched
ON market_price_cache(fetched_at);
