"""Unit conversion, validation and cautious inference. Pure Python, no LLM."""
from __future__ import annotations

from typing import Any

from core.schemas import SiteProfileDraft

# Conventional van Bemmelen factor. Flagged as approximate wherever it is used.
ORGANIC_MATTER_TO_CARBON = 1.724

PLAUSIBLE_RANGE = {
    "soc_percent": (0.0, 20.0),
    "ph": (2.0, 12.0),
    "rainfall_mm": (0.0, 12000.0),
    "natural_cover_percent": (0.0, 100.0),
    "habitat_diversity_index": (0.0, 4.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
}

RAINFALL_CATEGORY_MM = {"low": 400.0, "medium": 800.0, "high": 1500.0}

LAND_USE_VALUES = {"cropland", "grassland", "pasture", "orchard", "forest", "fallow"}


def soc_from_g_per_kg(value: float) -> float:
    """SoilGrids and lab reports often give g/kg; percent is 10x smaller."""
    return value / 10.0


def soc_from_organic_matter(percent: float) -> float:
    """Approximate: organic matter to organic carbon."""
    return percent / ORGANIC_MATTER_TO_CARBON


def soc_from_stock(stock_t_ha: float, bulk_density: float, depth_cm: float) -> float:
    """SOC% from a stock in t/ha, given bulk density (g/cm3) and depth (cm)."""
    if bulk_density <= 0 or depth_cm <= 0:
        raise ValueError("bulk density and depth must be positive")
    return stock_t_ha / (bulk_density * depth_cm)


def validate(name: str, value: Any) -> str | None:
    """Return a complaint if the value cannot be real, otherwise None."""
    bounds = PLAUSIBLE_RANGE.get(name)
    if bounds is None or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} should be a number"
    low, high = bounds
    if not low <= float(value) <= high:
        return f"{name} of {value} is outside the plausible range {low}-{high}"
    return None


def climate_zone_from_rainfall(rainfall_mm: float) -> str | None:
    """Coarse zone from rainfall alone. Only ever recorded as inferred."""
    if rainfall_mm < 300:
        return "arid"
    if rainfall_mm < 600:
        return "semi-arid"
    if rainfall_mm < 1000:
        return "dry_sub_humid"
    return "humid"


def normalise_draft(
    draft: SiteProfileDraft,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    """Draft to flat profile values.

    Returns (stated values, values derived from a category the user gave, warnings
    about conversions, rejections that need re-checking). Derived values are kept
    apart so they can be recorded as inference rather than as something the user said.
    """
    values: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    warnings: list[str] = []
    rejected: list[str] = []

    if draft.soc_percent is not None:
        values["soc_percent"] = float(draft.soc_percent)
    elif draft.soc_g_per_kg is not None:
        values["soc_percent"] = round(soc_from_g_per_kg(float(draft.soc_g_per_kg)), 3)
        warnings.append("Converted organic carbon from g/kg to percent")
    elif draft.soil_organic_matter_percent is not None:
        values["soc_percent"] = round(
            soc_from_organic_matter(float(draft.soil_organic_matter_percent)), 3
        )
        warnings.append(
            "Converted soil organic matter to organic carbon with the 1.724 factor; "
            "treat it as approximate"
        )

    if draft.ph is not None:
        values["ph"] = float(draft.ph)

    if draft.rainfall_mm is not None:
        values["rainfall_mm"] = float(draft.rainfall_mm)
    elif draft.rainfall_category is not None:
        derived["rainfall_mm"] = RAINFALL_CATEGORY_MM[draft.rainfall_category]
        warnings.append(
            f"Used a representative {draft.rainfall_category} rainfall figure; "
            f"give millimetres if you know them"
        )

    if draft.climate_zone is not None:
        values["climate_zone"] = draft.climate_zone.value
    if draft.land_use is not None:
        land_use = draft.land_use.strip().lower()
        if land_use in LAND_USE_VALUES:
            values["land_use"] = land_use
    if draft.crop is not None:
        values["crop"] = draft.crop.strip().lower()
    if draft.cropping_system is not None:
        values["cropping_system"] = draft.cropping_system
    if draft.natural_cover_percent is not None:
        values["natural_cover_percent"] = float(draft.natural_cover_percent)
    if draft.biodiversity_trend is not None:
        values["biodiversity_trend"] = draft.biodiversity_trend
    if draft.pollinator_trend is not None:
        values["pollinator_trend"] = draft.pollinator_trend
    if draft.pesticide_use is not None:
        values["pesticide_use"] = draft.pesticide_use
    if draft.residue_burning is not None:
        values["residue_burning"] = bool(draft.residue_burning)
    if draft.recent_clearing is not None:
        values["recent_clearing"] = bool(draft.recent_clearing)
    if draft.nearby_pollution_source is not None:
        values["nearby_pollution_source"] = draft.nearby_pollution_source.strip().lower()
    if draft.erosion_observed is not None:
        values["erosion_observed"] = bool(draft.erosion_observed)
    if draft.soil_moisture_status is not None:
        values["soil_moisture_status"] = draft.soil_moisture_status
    if draft.irrigation is not None:
        values["irrigation"] = draft.irrigation.strip().lower()
    if draft.place_name is not None:
        values["place_name"] = draft.place_name.strip()
    if draft.lat is not None:
        values["lat"] = float(draft.lat)
    if draft.lon is not None:
        values["lon"] = float(draft.lon)

    for name in list(values):
        problem = validate(name, values[name])
        if problem:
            rejected.append(problem)
            values.pop(name)

    return values, derived, warnings, rejected


def infer_missing(values: dict[str, Any]) -> dict[str, Any]:
    """Cheap inferences worth making, returned separately so they stay marked inferred."""
    inferred: dict[str, Any] = {}
    if "climate_zone" not in values and isinstance(values.get("rainfall_mm"), (int, float)):
        zone = climate_zone_from_rainfall(float(values["rainfall_mm"]))
        if zone:
            inferred["climate_zone"] = zone
    if "land_use" not in values and values.get("crop"):
        inferred["land_use"] = "cropland"
    return inferred
