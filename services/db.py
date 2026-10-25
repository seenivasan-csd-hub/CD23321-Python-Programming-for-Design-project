import re
import sqlite3
from pathlib import Path


MIGRATIONS = {
    "farms": {"user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE"},
    "soil_records": {"user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE"},
    "disease_scans": {
        "crop_name": "TEXT",
        "image_path": "TEXT",
        "model_version": "TEXT",
        "scan_status": "TEXT NOT NULL DEFAULT 'unanalysed'",
        "uncertain": "INTEGER NOT NULL DEFAULT 1",
        "input_sha256": "TEXT",
        "source_url": "TEXT",
        "user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE",
    },
    "weather_records": {
        "current_temperature": "REAL",
        "relative_humidity": "REAL",
        "wind_speed": "REAL",
        "weather_code": "INTEGER",
        "source": "TEXT",
        "fetched_at": "TEXT",
        "user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE",
    },
    "market_prices": {
        "unit": "TEXT",
        "source_url": "TEXT",
        "fetched_at": "TEXT",
        "user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE",
    },
    "farm_profiles": {"user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE"},
    "soil_test_records": {"user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE"},
    "crop_advice_history": {"user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE"},
    "crop_recommendations": {
        "district": "TEXT",
        "season": "TEXT",
        "evidence_status": "TEXT",
        "source_url": "TEXT",
        "user_id": "INTEGER REFERENCES users(user_id) ON DELETE CASCADE",
    },
}


def _rebuild_singleton_farm_tables(connection):
    table = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='farm_profiles'"
    ).fetchone()
    if not table or not re.search(r"CHECK\s*\(\s*profile_id\s*=\s*1\s*\)", table[0], re.IGNORECASE):
        return

    old_names = {
        "farm_profiles": "farm_profiles_legacy_auth",
        "soil_test_records": "soil_test_records_legacy_auth",
        "crop_advice_history": "crop_advice_history_legacy_auth",
    }
    for name in old_names.values():
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
            raise RuntimeError("A previous account migration backup table already exists; database was left unchanged.")

    connection.commit()
    connection.execute("PRAGMA foreign_keys=OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        for current_name, legacy_name in old_names.items():
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (current_name,)).fetchone():
                connection.execute(f'ALTER TABLE "{current_name}" RENAME TO "{legacy_name}"')

        connection.execute("""
            CREATE TABLE farm_profiles (
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
            )
        """)
        connection.execute("""
            CREATE TABLE soil_test_records (
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
            )
        """)
        connection.execute("""
            CREATE TABLE crop_advice_history (
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
            )
        """)

        connection.execute("""
            INSERT INTO farm_profiles(profile_id,user_id,farm_id,district,village,latitude,longitude,crop,area_acres,soil_type,irrigation_method,sowing_date,consented_at,updated_at)
            SELECT profile_id,user_id,farm_id,district,village,latitude,longitude,crop,area_acres,soil_type,irrigation_method,sowing_date,consented_at,updated_at
            FROM farm_profiles_legacy_auth
        """)
        connection.execute("""
            INSERT INTO soil_test_records(soil_test_id,profile_id,nitrogen,phosphorus,potassium,soil_ph,electrical_conductivity,organic_carbon,moisture,phosphorus_method,source,lab_name,confidence_label,sample_date,notes,recorded_at,user_id)
            SELECT soil_test_id,profile_id,nitrogen,phosphorus,potassium,soil_ph,electrical_conductivity,organic_carbon,moisture,phosphorus_method,source,lab_name,confidence_label,sample_date,notes,recorded_at,user_id
            FROM soil_test_records_legacy_auth
        """)
        connection.execute("""
            INSERT INTO crop_advice_history(advice_id,profile_id,district,sowing_month,water_availability,results_json,knowledge_version,created_at,user_id)
            SELECT advice_id,profile_id,district,sowing_month,water_availability,results_json,knowledge_version,created_at,user_id
            FROM crop_advice_history_legacy_auth
        """)
        for legacy_name in (old_names["soil_test_records"], old_names["crop_advice_history"], old_names["farm_profiles"]):
            connection.execute(f'DROP TABLE "{legacy_name}"')
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys=ON")

    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("Account migration found invalid foreign-key references; database integrity check failed.")


def initialize(path, schema_path):
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(Path(schema_path).read_text(encoding="utf-8"))
        for table, columns in MIGRATIONS.items():
            present = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in present:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        _rebuild_singleton_farm_tables(connection)
        connection.executescript("""
            CREATE INDEX IF NOT EXISTS idx_farms_user ON farms(user_id);
            CREATE INDEX IF NOT EXISTS idx_soil_records_user ON soil_records(user_id);
            CREATE INDEX IF NOT EXISTS idx_crop_recommendations_user ON crop_recommendations(user_id);
            CREATE INDEX IF NOT EXISTS idx_disease_scans_user ON disease_scans(user_id);
            CREATE INDEX IF NOT EXISTS idx_weather_records_user ON weather_records(user_id);
            CREATE INDEX IF NOT EXISTS idx_market_prices_user ON market_prices(user_id);
            CREATE INDEX IF NOT EXISTS idx_farm_profiles_user ON farm_profiles(user_id);
            CREATE INDEX IF NOT EXISTS idx_soil_test_records_user ON soil_test_records(user_id);
            CREATE INDEX IF NOT EXISTS idx_crop_advice_history_user ON crop_advice_history(user_id);
        """)
        connection.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (3)")
        connection.commit()
    finally:
        connection.close()


def connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
