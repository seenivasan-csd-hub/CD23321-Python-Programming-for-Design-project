from datetime import datetime, timezone
import logging
import math
import os
import re
from pathlib import Path

import requests


CURRENT_URL = "https://api.agromonitoring.com/agro/1.0/weather"
FORECAST_URL = "https://api.agromonitoring.com/agro/1.0/weather/forecast"
SOURCE_NAME = "AgroMonitoring Weather API"
SOURCE_URL = "https://agromonitoring.com/api/current-weather"
FORECAST_SOURCE_URL = "https://agromonitoring.com/api/forecast-weather"
ICON_PATTERN = re.compile(r"^[0-9]{2}[dn]$")
LOGGER = logging.getLogger(__name__)


class WeatherUnavailable(Exception):
    pass


def _api_key():
    configured = os.environ.get("AGROMONITORING_API_KEY", "").strip()
    if configured:
        return configured

    env_path = Path(__file__).resolve().parents[2] / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or key.strip() != "AGROMONITORING_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split("#", 1)[0].strip()
        return value.strip()
    return ""


def is_configured():
    return bool(_api_key())


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _icon_url(value):
    if isinstance(value, str) and ICON_PATTERN.fullmatch(value):
        return f"https://openweathermap.org/img/wn/{value}@2x.png"
    return None


def _weather_fields(payload):
    entries = payload.get("weather") if isinstance(payload, dict) else None
    condition = entries[0] if isinstance(entries, list) and entries and isinstance(entries[0], dict) else {}
    description = condition.get("description")
    if not isinstance(description, str) or not description.strip():
        description = None
    icon = condition.get("icon")
    icon = icon if isinstance(icon, str) and ICON_PATTERN.fullmatch(icon) else None
    return {
        "description": description,
        "icon": icon,
        "icon_url": _icon_url(icon),
        "weather_id": _finite_number(condition.get("id")),
    }


def _rainfall(payload):
    rain = payload.get("rain") if isinstance(payload, dict) else None
    if not isinstance(rain, dict):
        return None, None
    for period in ("3h", "1h"):
        value = _finite_number(rain.get(period))
        if value is not None:
            return value, period
    return None, None


def _format_timestamp(value):
    timestamp = _finite_number(value)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="minutes").replace("+00:00", " UTC")
    except (OverflowError, OSError, ValueError):
        return None


def _raise_request_error(endpoint_name, error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        LOGGER.warning("AgroMonitoring %s request returned HTTP %s", endpoint_name, status)
        if status == 401:
            message = "AgroMonitoring rejected the API key (HTTP 401). Check AGROMONITORING_API_KEY."
        elif status == 403:
            message = "AgroMonitoring denied this API key (HTTP 403). Check its access or subscription."
        elif status == 429:
            message = "AgroMonitoring rate limit reached. Wait briefly, then retry."
        elif status == 400:
            message = "AgroMonitoring rejected the coordinates or request. Check the decimal latitude and longitude, then retry."
        else:
            message = f"AgroMonitoring is temporarily unavailable (HTTP {status}). Retry shortly."
        raise WeatherUnavailable(message) from None

    if isinstance(error, requests.Timeout):
        LOGGER.warning("AgroMonitoring %s request timed out", endpoint_name)
        message = "AgroMonitoring did not respond before the request timed out. Retry shortly."
    elif isinstance(error, requests.ConnectionError):
        LOGGER.warning("AgroMonitoring %s connection failed", endpoint_name)
        message = "FarmIQ could not connect to AgroMonitoring. Check your internet connection and retry."
    else:
        LOGGER.warning("AgroMonitoring %s request failed (%s)", endpoint_name, type(error).__name__)
        message = "FarmIQ could not retrieve weather from AgroMonitoring. Retry shortly."
    raise WeatherUnavailable(message) from None


def _get_json(http, url, params, endpoint_name):
    try:
        response = http.get(url, params=params, timeout=12)
        response.raise_for_status()
    except requests.RequestException as exc:
        _raise_request_error(endpoint_name, exc)

    try:
        payload = response.json()
    except (requests.RequestException, ValueError, TypeError) as exc:
        LOGGER.warning("AgroMonitoring %s returned invalid JSON (%s)", endpoint_name, type(exc).__name__)
        raise WeatherUnavailable("AgroMonitoring returned an unreadable response. Retry shortly.") from None
    return payload


def _normalize_current(payload):
    if not isinstance(payload, dict):
        LOGGER.warning("AgroMonitoring current response was not a JSON object")
        raise WeatherUnavailable("AgroMonitoring returned an unsupported current-weather response.")
    main = payload.get("main")
    temperature = _finite_number(main.get("temp")) if isinstance(main, dict) else None
    if temperature is None:
        LOGGER.warning("AgroMonitoring current response is missing main.temp")
        raise WeatherUnavailable("AgroMonitoring did not return a current temperature. Retry shortly.")

    fields = _weather_fields(payload)
    if fields["description"] is None:
        LOGGER.warning("AgroMonitoring current response is missing weather[0].description")
        raise WeatherUnavailable("AgroMonitoring did not return a current weather description. Retry shortly.")
    rainfall, rainfall_period = _rainfall(payload)
    wind = payload.get("wind") if isinstance(payload.get("wind"), dict) else {}
    clouds = payload.get("clouds") if isinstance(payload.get("clouds"), dict) else {}
    return {
        **fields,
        "time": _format_timestamp(payload.get("dt")),
        "temperature_c": temperature,
        "humidity": _finite_number(main.get("humidity")),
        "wind_speed": _finite_number(wind.get("speed")),
        "cloud_cover": _finite_number(clouds.get("all")),
        "rainfall_mm": rainfall,
        "rainfall_period": rainfall_period,
    }


def _normalize_forecast(payload):
    if not isinstance(payload, list) or not payload:
        LOGGER.warning("AgroMonitoring forecast response was not a non-empty JSON array")
        raise WeatherUnavailable("AgroMonitoring returned no forecast entries for these coordinates.")

    periods = []
    invalid_entries = 0
    for item in payload:
        if not isinstance(item, dict):
            invalid_entries += 1
            continue
        main = item.get("main")
        temperature = _finite_number(main.get("temp")) if isinstance(main, dict) else None
        time_label = _format_timestamp(item.get("dt"))
        if temperature is None or time_label is None:
            invalid_entries += 1
            continue
        fields = _weather_fields(item)
        rainfall, rainfall_period = _rainfall(item)
        wind = item.get("wind") if isinstance(item.get("wind"), dict) else {}
        clouds = item.get("clouds") if isinstance(item.get("clouds"), dict) else {}
        periods.append({
            **fields,
            "time": time_label,
            "temperature_c": temperature,
            "humidity": _finite_number(main.get("humidity")),
            "wind_speed": _finite_number(wind.get("speed")),
            "cloud_cover": _finite_number(clouds.get("all")),
            "rainfall_mm": rainfall,
            "rainfall_period": rainfall_period,
        })
    if invalid_entries:
        LOGGER.warning("AgroMonitoring forecast contained %s entries missing dt or main.temp", invalid_entries)
    if not periods:
        raise WeatherUnavailable("AgroMonitoring forecast entries were missing timestamps or temperatures.")
    return periods


def fetch_forecast(latitude, longitude, http=requests, api_key=None):
    """Fetch and normalize AgroMonitoring current conditions and forecast intervals."""
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise WeatherUnavailable("Enter decimal latitude and longitude coordinates.")
    latitude = _finite_number(latitude)
    longitude = _finite_number(longitude)
    if latitude is None or not -90 <= latitude <= 90:
        raise WeatherUnavailable("Latitude must be a decimal number between -90 and 90.")
    if longitude is None or not -180 <= longitude <= 180:
        raise WeatherUnavailable("Longitude must be a decimal number between -180 and 180.")

    api_key = api_key if api_key is not None else _api_key()
    api_key = api_key.strip()
    if not api_key:
        raise WeatherUnavailable("AgroMonitoring is not configured. Add AGROMONITORING_API_KEY to the project's .env file and restart FarmIQ.")

    params = {"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}
    current_payload = _get_json(http, CURRENT_URL, params, "current-weather")
    forecast_payload = _get_json(http, FORECAST_URL, params, "forecast")
    return {
        "current": _normalize_current(current_payload),
        "periods": _normalize_forecast(forecast_payload),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "forecast_source_url": FORECAST_SOURCE_URL,
    }
