from datetime import datetime, timezone
import os

import requests


SOURCE_NAME = "AGMARKNET / Open Government Data Platform India"
SOURCE_URL = "https://www.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi"
RESOURCE_ID = "6141ea17-a69d-4713-b600-0a43c8fd9a6c"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"


class MarketUnavailable(Exception):
    pass


def fetch_prices(commodity, district=None, market=None, http=requests, api_key=None):
    api_key = api_key if api_key is not None else os.environ.get("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        raise MarketUnavailable("Live market data is not configured. Add a data.gov.in API key to the environment.")
    params = {"api-key": api_key, "format": "json", "limit": 100}
    filters = {"state": "Tamil Nadu", "commodity": commodity}
    if district:
        filters["district"] = district
    if market:
        filters["market"] = market
    params.update({f"filters[{key}]": value for key, value in filters.items()})
    try:
        response = http.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        body = response.json()
        records = body.get("records")
        if not isinstance(records, list):
            raise MarketUnavailable("The market-data service returned an unsupported response.")
        prices = []
        for row in records:
            try:
                prices.append({
                    "commodity": str(row.get("commodity") or commodity)[:100],
                    "district": str(row.get("district") or "")[:100],
                    "market": str(row.get("market") or "")[:100],
                    "variety": str(row.get("variety") or "")[:100],
                    "date": str(row.get("arrival_date") or row.get("date") or "")[:32],
                    "min_price": float(row["min_price"]) if row.get("min_price") not in (None, "") else None,
                    "max_price": float(row["max_price"]) if row.get("max_price") not in (None, "") else None,
                    "modal_price": float(row["modal_price"]) if row.get("modal_price") not in (None, "") else None,
                    "unit": str(row.get("price_unit") or row.get("unit") or "Unit not supplied by source")[:80],
                    "source": SOURCE_NAME,
                    "source_url": SOURCE_URL,
                })
            except (TypeError, ValueError, KeyError):
                continue
        return {"prices": prices, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": SOURCE_NAME, "source_url": SOURCE_URL}
    except MarketUnavailable:
        raise
    except (requests.RequestException, ValueError, TypeError) as exc:
        # In particular, never put an API key or full request URL in an error message.
        raise MarketUnavailable("Live market data could not be reached. Showing stored/imported data separately.") from exc
