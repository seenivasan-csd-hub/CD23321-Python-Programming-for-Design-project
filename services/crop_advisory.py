import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = BASE_DIR / "data" / "tn_crop_knowledge.json"
ALIASES = {
    "tiruchirapalli": "tiruchirappalli", "trichy": "tiruchirappalli",
    "tiruchy": "tiruchirappalli", "pudukottai": "pudukkottai",
    "thirunelveli": "tirunelveli", "tuticorin": "thoothukudi",
    "thiruvallur": "thiruvallur", "thiruvannamalai": "tiruvannamalai",
    "viluppuram": "viluppuram", "villupuram": "viluppuram",
    "namakal": "namakkal", "sivagangai": "sivaganga",
    "sivagangai": "sivaganga", "kanniyakumari": "kanyakumari",
    "thiruvarur": "thiruvarur", "tiruvarur": "thiruvarur",
    "tirupathur": "tirupathur", "tiruppur": "tiruppur",
}


def _key(value):
    key = " ".join((value or "").strip().lower().replace("&", "and").split())
    return ALIASES.get(key, key)


def load_knowledge(path=KNOWLEDGE_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recommend(district, month, water_availability="unknown", soil_type=None, knowledge=None):
    knowledge = knowledge or load_knowledge()
    district_key = _key(district)
    month = int(month)
    results = []
    for crop in knowledge["crops"]:
        matching = []
        district_month_windows = []
        for window in crop.get("windows", []):
            district_keys = {_key(name) for name in window.get("districts", [])}
            excluded = {_key(name) for name in window.get("exclude_districts", [])}
            district_match = (window.get("all_districts", False) or district_key in district_keys) and district_key not in excluded
            if not district_match or month not in window.get("months", []):
                continue
            district_month_windows.append(window)
            allowed_water = window.get("water_modes")
            if water_availability in {"rainfed", "irrigated"} and allowed_water:
                if water_availability not in allowed_water:
                    continue
            matching.append(window)

        if matching:
            first = matching[0]
            results.append({
                "crop": crop["name"], "group": crop.get("group", "Other"),
                "matched": True, "status": "Documented district and sowing-window match",
                "season": first["season"], "duration": first.get("duration"),
                "water_note": first.get("water_note") or crop.get("water_note"),
                "soil_note": crop.get("soil_note"),
                "reason": first.get("note") or crop.get("note", "Check local conditions."),
                "source_url": first.get("source_url") or crop.get("source_url"),
                "source_title": first.get("source_title") or crop.get("source_title"),
                "evidence": "TNAU district/season guidance",
            })
        elif crop.get("preliminary"):
            results.append({
                "crop": crop["name"], "group": crop.get("group", "Other"),
                "matched": False, "preliminary": True,
                "status": "Preliminary; no exact district/month match in this data set",
                "season": None, "duration": crop.get("duration"),
                "water_note": crop.get("water_note"), "soil_note": crop.get("soil_note"),
                "reason": crop.get("note", "Confirm suitability with local extension staff."),
                "source_url": crop.get("source_url"), "source_title": crop.get("source_title"),
                "evidence": "Background source only; not ranked",
            })
        else:
            water_limited = bool(district_month_windows)
            limiting_window = district_month_windows[0] if water_limited else None
            allowed_modes = (limiting_window or {}).get("water_modes", [])
            if water_limited:
                reason = (
                    "A TNAU district and month window exists, but it is listed for "
                    f"{', '.join(allowed_modes)} water conditions. The selected water category did not match."
                )
            else:
                reason = (
                    "No included district and month window matched this query. This limited calendar is not proof that the crop is unsuitable."
                )
            results.append({
                "crop": crop["name"], "group": crop.get("group", "Other"),
                "matched": False, "preliminary": False,
                "status": "No documented calendar match",
                "season": limiting_window.get("season") if limiting_window else None,
                "duration": limiting_window.get("duration") if limiting_window else None,
                "water_note": crop.get("water_note"),
                "soil_note": crop.get("soil_note"),
                "reason": reason,
                "source_url": limiting_window.get("source_url", crop.get("source_url")) if limiting_window else crop.get("source_url"),
                "source_title": limiting_window.get("source_title", crop.get("source_title")) if limiting_window else crop.get("source_title"),
                "evidence": "Included calendar had no matching window",
            })

    results.sort(key=lambda item: (not item["matched"], item.get("preliminary", False), item["crop"]))
    rotations = []
    for rotation in knowledge.get("rotations", []):
        if district_key in {_key(name) for name in rotation["districts"]}:
            rotations.append(rotation)
    return {"results": results, "rotations": rotations, "month": month, "district": district,
            "soil_type": soil_type, "water_availability": water_availability}
