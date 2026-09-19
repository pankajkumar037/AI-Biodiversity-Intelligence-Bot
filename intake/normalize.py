"""Unit conversion, validation and cautious inference. Pure Python, no LLM."""
from __future__ import annotations

import re
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

# A value is kept only if the message contains a word about that topic. The
# extraction model fills fields it was never told about; this is the mechanical
# guard against a value being stamped "user" when the user never said it.
FIELD_KEYWORDS: dict[str, str] = {
    "soc_percent": r"carbon|\bsoc\b|organic",
    "soc_g_per_kg": r"carbon|\bsoc\b|organic",
    "soil_organic_matter_percent": r"organic matter|\bsom\b",
    "ph": r"\bph\b|acidic|alkaline|sodic",
    "rainfall_mm": r"rain|precip|\bmm\b",
    "rainfall_category": r"rain|precip",
    "climate_zone": r"arid|humid|temperate|tropical|dryland|desert",
    "land_use": r"cropland|farm|field|grassland|pasture|orchard|forest|fallow|grazing|crop|wheat|rice|maize|grow",
    "cropping_system": r"mono|rotation|rotate|mixed|fallow|only|single|intercrop",
    "natural_cover_percent": r"cover|vegetation|trees|scrub|margin|natural",
    "biodiversity_trend": r"biodivers|species|wildlife|bird|insect|bee|pollinat|fewer|declin|disappear",
    "pollinator_trend": r"pollinat|\bbees?\b|butterfl|insect",
    "pesticide_use": r"pesticid|insecticid|herbicid|fungicid|spray|agrochem|chemical",
    "residue_burning": r"burn|fire|stubble",
    "recent_clearing": r"clear|cut|fell|deforest|removed trees|no trees",
    "nearby_pollution_source": r"pollut|factory|effluent|mine|industr|sewage|dump",
    "erosion_observed": r"eros|gull|wash|runoff|topsoil|soil loss|rill",
    "soil_moisture_status": r"moist|dry soil|soil is dry|wet soil|waterlogg|damp",
    "overgrazed": r"overgraz|grazing pressure|too many (cattle|goats|sheep|animals)|heavily grazed",
    "trees_nearby": r"tree|wood|shrub|scrub|vegetation|bare|treeless",
    "irrigation": r"irrigat|drip|sprinkler|canal|borewell|tube ?well|watered",
}

ZONE_WORDS = [
    ("semi-arid", r"semi[- ]?arid"),
    ("arid", r"\barid\b|desert|hot arid"),
    ("dry_sub_humid", r"dry sub[- ]?humid"),
    ("humid", r"\bhumid\b"),
    ("temperate", r"temperate"),
    ("tropical", r"tropical"),
]


CROP_WORDS = {
    "wheat": "wheat", "rice": "rice", "paddy": "rice", "maize": "maize", "corn": "maize",
    "millet": "millet", "bajra": "millet", "sorghum": "sorghum", "jowar": "sorghum",
    "cotton": "cotton", "soy": "soybean", "mustard": "mustard", "barley": "barley",
    "chickpea": "chickpea", "gram": "chickpea", "lentil": "lentil", "pulse": "pulses",
    "groundnut": "groundnut", "sugarcane": "sugarcane", "potato": "potato",
}
LAND_USE_WORDS = [
    ("orchard", r"orchard|almond|mango|olive|citrus|apple|plantation"),
    ("grassland", r"grassland|pasture|rangeland|grazing|overgraz|meadow|livestock"),
    ("forest", r"\bforest\b|woodland"),
    ("fallow", r"\bfallow\b"),
    ("cropland", r"cropland|arable|farm|field|monoculture|intercrop|rotation|crop"),
]

# Only these mean something to the rule engine; anything else the extractor puts
# in "constraints" is a paraphrase of the message, not a constraint.
CONSTRAINT_SLUGS = {"leased_land_no_trees", "no_irrigation", "no_livestock",
                    "no_machinery", "residue_needed_for_fodder"}


def crop_in_text(message: str) -> str | None:
    """A crop the user named, from a fixed word list."""
    text = message.lower()
    for word, crop in CROP_WORDS.items():
        if re.search(rf"\b{word}", text):
            return crop
    return None


def land_use_in_text(message: str) -> str | None:
    """Land use implied by the words used. A named crop means cropland."""
    text = message.lower()
    for land_use, pattern in LAND_USE_WORDS:
        if re.search(pattern, text):
            return land_use
    return "cropland" if crop_in_text(message) else None


def zone_in_text(message: str) -> str | None:
    """The climate zone the user named, if they named one. Semi-arid before arid."""
    text = message.lower()
    for zone, pattern in ZONE_WORDS:
        if re.search(pattern, text):
            return zone
    return None


def unsupported_fields(values: dict[str, Any], message: str) -> list[str]:
    """Extracted fields the message gives no word for."""
    text = message.lower()
    return [
        name for name in values
        if name in FIELD_KEYWORDS and not re.search(FIELD_KEYWORDS[name], text)
    ]


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
    if draft.overgrazed is not None:
        values["overgrazed"] = bool(draft.overgrazed)
    if draft.trees_nearby is not None:
        values["trees_nearby"] = bool(draft.trees_nearby)
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
