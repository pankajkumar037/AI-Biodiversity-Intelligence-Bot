"""Optional geo enrichment. Every call is cached, timed out and allowed to fail."""
from __future__ import annotations

from typing import Any

import httpx

TIMEOUT_SECONDS = 8.0
USER_AGENT = "darukaa-biodiversity-intelligence/0.1 (hackathon prototype)"

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_cache: dict[tuple, Any] = {}


def _cell(lat: float, lon: float) -> tuple[float, float]:
    """Round to roughly 110 m so nearby requests share a cache entry."""
    return (round(lat, 3), round(lon, 3))


def _get(url: str, params: dict) -> dict | None:
    try:
        response = httpx.get(
            url, params=params, timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        # NOTE: a failed lookup leaves the slot empty for the next step of the
        # waterfall. It must never take down a turn.
        return None


def geocode(place: str) -> dict | None:
    """Place name to coordinates via Nominatim. One request, no grid queries."""
    key = ("geocode", place.lower())
    if key in _cache:
        return _cache[key]
    data = _get(NOMINATIM_URL, {"q": place, "format": "json", "limit": 1})
    result = None
    if data:
        first = data[0]
        result = {
            "lat": float(first["lat"]),
            "lon": float(first["lon"]),
            "display_name": first.get("display_name"),
        }
    _cache[key] = result
    return result


def soilgrids(lat: float, lon: float) -> dict | None:
    """Modelled SOC and pH at 0-5 cm. These are 250 m predictions, not soil tests."""
    key = ("soilgrids", *_cell(lat, lon))
    if key in _cache:
        return _cache[key]
    data = _get(SOILGRIDS_URL, {
        "lat": lat, "lon": lon,
        "property": ["soc", "phh2o"], "depth": "0-5cm", "value": "mean",
    })
    result = None
    if data:
        result = {}
        for layer in data.get("properties", {}).get("layers", []):
            name = layer.get("name")
            depths = layer.get("depths") or []
            if not depths:
                continue
            mean = (depths[0].get("values") or {}).get("mean")
            if mean is None:
                continue
            factor = (layer.get("unit_measure") or {}).get("d_factor", 1) or 1
            if name == "soc":
                # SoilGrids reports soc in dg/kg; convert to percent.
                result["soc_percent"] = round(float(mean) / factor / 10.0, 3)
            elif name == "phh2o":
                result["ph"] = round(float(mean) / factor, 2)
    _cache[key] = result
    return result


def nasa_power_climatology(lat: float, lon: float) -> dict | None:
    """Long-term annual rainfall and temperature from the AG community climatology."""
    key = ("power", *_cell(lat, lon))
    if key in _cache:
        return _cache[key]
    data = _get(NASA_POWER_URL, {
        "parameters": "PRECTOTCORR,T2M", "community": "AG",
        "longitude": lon, "latitude": lat, "format": "JSON",
    })
    result = None
    if data:
        parameters = data.get("properties", {}).get("parameter", {})
        rainfall = parameters.get("PRECTOTCORR", {}).get("ANN")
        temperature = parameters.get("T2M", {}).get("ANN")
        result = {}
        if rainfall is not None:
            # PRECTOTCORR climatology is mm/day; annual total is roughly x365.
            result["rainfall_mm"] = round(float(rainfall) * 365.0, 1)
        if temperature is not None:
            result["temperature_c"] = round(float(temperature), 1)
    _cache[key] = result
    return result


def enrich(lat: float, lon: float) -> dict[str, Any]:
    """Every source we can reach for this point. Missing sources return None."""
    return {
        "soil": soilgrids(lat, lon),
        "climate": nasa_power_climatology(lat, lon),
    }


def clear_cache() -> None:
    _cache.clear()
