from datetime import date
import math


class SoilValidationError(ValueError):
    pass


FIELD_RULES = {
    "nitrogen": (0, 10000, "kg/ha"),
    "phosphorus": (0, 5000, "kg/ha"),
    "potassium": (0, 10000, "kg/ha"),
    "soil_ph": (0, 14, "pH"),
    "electrical_conductivity": (0, 100, "dS/m"),
    "organic_carbon": (0, 100, "%"),
    "moisture": (0, 100, "%"),
}


def _optional_number(raw, field, low, high):
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SoilValidationError(f"{field.replace('_', ' ').title()} must be numeric.") from exc
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise SoilValidationError(
            f"{field.replace('_', ' ').title()} must be between {low:g} and {high:g}."
        )
    return parsed


def validate_soil_test(form, today=None):
    values = {
        key: _optional_number(form.get(key), key, low, high)
        for key, (low, high, _unit) in FIELD_RULES.items()
    }
    if not any(value is not None for value in values.values()):
        raise SoilValidationError("Enter at least one measurement from a soil test.")

    phosphorus_method = (form.get("phosphorus_method") or "").strip().lower()
    if phosphorus_method not in {"", "olsen", "bray"}:
        raise SoilValidationError("Choose Olsen, Bray, or no phosphorus method.")

    sample_date = (form.get("sample_date") or "").strip()
    if sample_date:
        try:
            parsed_date = date.fromisoformat(sample_date)
        except ValueError as exc:
            raise SoilValidationError("Sample date must be a valid date.") from exc
        if parsed_date > (today or date.today()):
            raise SoilValidationError("Sample date cannot be in the future.")

    lab_name = (form.get("lab_name") or "").strip()[:120]
    lab_confirmed = form.get("lab_confirmed") == "yes"
    if lab_confirmed and not lab_name:
        raise SoilValidationError("Enter the laboratory name shown on the report.")

    values.update({
        "phosphorus_method": phosphorus_method or None,
        "sample_date": sample_date or None,
        "source": "laboratory" if lab_confirmed else "user_entered",
        "lab_name": lab_name if lab_confirmed else None,
        "confidence_label": (
            "Reported by laboratory; not independently verified by this app"
            if lab_confirmed else "User-entered; not independently verified"
        ),
        "notes": (form.get("notes") or "").strip()[:500] or None,
    })
    return values


def source_label(source):
    return {
        "laboratory": "Laboratory-measured (as reported)",
        "user_entered": "User-entered soil test",
        "regional": "Regional dataset estimate",
    }.get(source, "Unavailable")


def interpret_soil_record(record):
    """Return TNAU soil-test bands; these are not fertilizer prescriptions."""
    output = {}
    for field, label, unit, cutoffs in (
        ("organic_carbon", "Organic carbon", "%", (0.5, 0.75)),
        ("nitrogen", "Available nitrogen", "kg/ha", (280, 450)),
        ("potassium", "Available potassium", "kg/ha", (118, 280)),
    ):
        value = record.get(field)
        if value is None:
            output[field] = {"label": label, "value": None, "unit": unit, "band": "Unavailable"}
            continue
        low, medium = cutoffs
        band = "Low" if value < low else "Medium" if value <= medium else "High"
        output[field] = {"label": label, "value": value, "unit": unit, "band": band}

    p_value = record.get("phosphorus")
    method = (record.get("phosphorus_method") or "").lower()
    if p_value is None:
        p_band = "Unavailable"
    elif method == "olsen":
        p_band = "Low" if p_value < 11 else "Medium" if p_value <= 22 else "High"
    elif method == "bray":
        p_band = "Low" if p_value < 24.2 else "Medium" if p_value <= 49.7 else "High"
    else:
        p_band = "Method required for interpretation"
    output["phosphorus"] = {
        "label": "Available phosphorus", "value": p_value, "unit": "kg/ha",
        "band": p_band, "method": method or None,
    }

    for field, label, unit in (
        ("soil_ph", "Soil pH", "pH"),
        ("electrical_conductivity", "Electrical conductivity", "dS/m"),
        ("moisture", "Soil moisture", "%"),
    ):
        output[field] = {
            "label": label, "value": record.get(field), "unit": unit,
            "band": "Reported; crop- and method-specific interpretation not configured"
            if record.get(field) is not None else "Unavailable",
        }
    return output


def missing_soil_fields(record):
    return [
        label for key, label in (
            ("nitrogen", "available nitrogen"), ("phosphorus", "available phosphorus"),
            ("potassium", "available potassium"), ("soil_ph", "pH"),
            ("electrical_conductivity", "electrical conductivity"),
            ("organic_carbon", "organic carbon"), ("moisture", "moisture"),
        ) if record.get(key) is None
    ]
