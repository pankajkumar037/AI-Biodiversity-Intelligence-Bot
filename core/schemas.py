"""All enums and Pydantic models. Every other module imports from here."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── enums ──────────────────────────────────────────────────────────────
class MetricEnum(str, Enum):
    SOC = "SOC"
    soil_organic_matter = "soil_organic_matter"
    pH = "pH"
    soil_moisture = "soil_moisture"
    water_holding_capacity = "water_holding_capacity"
    bulk_density = "bulk_density"
    erosion = "erosion"
    nutrient_availability = "nutrient_availability"
    nitrogen = "nitrogen"
    microbial_biomass = "microbial_biomass"
    soil_biota = "soil_biota"
    species_richness = "species_richness"
    habitat_diversity = "habitat_diversity"
    fragmentation = "fragmentation"
    pollinators = "pollinators"
    tree_cover = "tree_cover"
    biomass = "biomass"
    crop_yield = "crop_yield"
    temperature = "temperature"
    rainfall = "rainfall"
    salinity = "salinity"
    pollution_load = "pollution_load"
    carbon_sequestration_rate = "carbon_sequestration_rate"


class PracticeEnum(str, Enum):
    cover_crops = "cover_crops"
    mulching = "mulching"
    crop_rotation = "crop_rotation"
    intercropping = "intercropping"
    no_till = "no_till"
    reduced_tillage = "reduced_tillage"
    manure = "manure"
    compost = "compost"
    integrated_nutrient_mgmt = "integrated_nutrient_mgmt"
    irrigation = "irrigation"
    terracing = "terracing"
    check_dams = "check_dams"
    shelterbelts = "shelterbelts"
    hedges_buffer_strips = "hedges_buffer_strips"
    grassland_restoration = "grassland_restoration"
    rotational_grazing = "rotational_grazing"
    crop_livestock_integration = "crop_livestock_integration"
    conservation_agriculture = "conservation_agriculture"
    agroforestry_agrisilvicultural = "agroforestry_agrisilvicultural"
    agroforestry_silvopastoral = "agroforestry_silvopastoral"
    agroforestry_agrosilvopastoral = "agroforestry_agrosilvopastoral"
    gypsum_amendment = "gypsum_amendment"
    water_harvesting = "water_harvesting"
    other = "other"


class ClimateZone(str, Enum):
    arid = "arid"
    semi_arid = "semi-arid"
    dry_sub_humid = "dry_sub_humid"
    humid = "humid"
    temperate = "temperate"
    tropical = "tropical"
    global_ = "global"
    unstated = "unstated"


class FieldSource(str, Enum):
    user = "user"
    soilgrids = "soilgrids"
    nasa_power = "nasa_power"
    nominatim = "nominatim"
    inferred = "inferred"


class Intent(str, Enum):
    new_info = "new_info"
    explain = "explain"
    constraint = "constraint"
    what_if = "what_if"
    concept = "concept"
    out_of_scope = "out_of_scope"


class Audience(str, Enum):
    farmer = "farmer"
    ngo = "ngo"
    researcher = "researcher"


Direction = Literal["increase", "decrease", "no_change", "mixed"]
TimeHorizon = Literal["short", "medium", "long"]
QueryPurpose = Literal["support", "risk", "path", "background"]

# Source priority for the profile merge rule: user beats APIs beats inference.
SOURCE_PRIORITY: dict[str, int] = {
    "user": 3,
    "soilgrids": 2,
    "nasa_power": 2,
    "nominatim": 2,
    "inferred": 1,
}


# ── site profile ───────────────────────────────────────────────────────
class SiteField(BaseModel):
    """One profile value with where it came from and how much to trust it."""

    value: Any
    source: FieldSource
    turn: int = 0
    uncertainty: float | None = None


class SiteProfile(BaseModel):
    """Flat named fields plus user constraints. The source of truth for a site."""

    fields: dict[str, SiteField] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    audience: Audience | None = None

    def value(self, name: str) -> Any:
        """Return the bare value of a field, or None when unset."""
        field = self.fields.get(name)
        return field.value if field else None

    def values(self) -> dict[str, Any]:
        """Return all fields as a plain name -> value dict."""
        return {name: field.value for name, field in self.fields.items()}

    def set(self, name: str, value: Any, source: FieldSource, turn: int = 0,
            uncertainty: float | None = None) -> None:
        """Set a field if the new source outranks the existing one."""
        existing = self.fields.get(name)
        if existing is not None:
            new_rank = SOURCE_PRIORITY[source.value]
            old_rank = SOURCE_PRIORITY[existing.source.value]
            if new_rank < old_rank:
                return
        self.fields[name] = SiteField(
            value=value, source=source, turn=turn, uncertainty=uncertainty
        )


class SiteProfileDraft(BaseModel):
    """What the extraction LLM is allowed to return. Never written to state directly."""

    soc_percent: float | None = None
    soc_g_per_kg: float | None = None
    soil_organic_matter_percent: float | None = None
    ph: float | None = None
    rainfall_mm: float | None = None
    rainfall_category: Literal["low", "medium", "high"] | None = None
    climate_zone: ClimateZone | None = None
    land_use: str | None = None
    crop: str | None = None
    cropping_system: Literal["monoculture", "rotation", "mixed", "fallow"] | None = None
    natural_cover_percent: float | None = None
    place_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    biodiversity_trend: Literal["declining", "stable", "improving"] | None = None
    irrigation: str | None = None
    constraints: list[str] = Field(default_factory=list)
    audience: Audience | None = None


# ── diagnosis ──────────────────────────────────────────────────────────
class Flag(BaseModel):
    """A problem detected by a threshold band or a compound rule."""

    flag: str
    rule_id: str
    metric: str | None = None
    value: Any = None
    band: str | None = None
    note: str | None = None


class Pattern(BaseModel):
    """An interacting problem pattern from compound_rules.json."""

    rule_id: str
    name: str
    implies: str
    loop: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


# ── knowledge ──────────────────────────────────────────────────────────
class SoftRisk(BaseModel):
    condition: str = Field(alias="if")
    risk: str
    penalty: float = 0.0
    mitigation: str | None = None

    model_config = {"populate_by_name": True}


class HardConstraints(BaseModel):
    land_use_in: list[str] = Field(default_factory=list)
    min_rainfall_mm: float | None = None
    max_rainfall_mm: float | None = None
    incompatible_constraints: list[str] = Field(default_factory=list)


class RegionVariant(BaseModel):
    region: str
    note: str
    source_unit: str | None = None


class PracticeCard(BaseModel):
    """One intervention. Carries no effect sizes — numbers come from claims."""

    practice_id: PracticeEnum
    name: str
    addresses_flags: list[str] = Field(default_factory=list)
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_risks: list[SoftRisk] = Field(default_factory=list)
    requires_first: list[str] = Field(default_factory=list)
    impacted_metrics: list[MetricEnum] = Field(default_factory=list)
    mechanism: str
    time_horizon: TimeHorizon
    region_variants: list[RegionVariant] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


# ── LLM output ─────────────────────────────────────────────────────────
class Estimate(BaseModel):
    """A number that must trace to a claim in the cited chunk."""

    metric: MetricEnum
    direction: Direction
    value: float | None = Field(
        default=None,
        description="A number copied verbatim from a CLAIMS line of the cited chunk. "
                    "Leave null if no CLAIMS line gives one. Never a score from the dossier.",
    )
    low: float | None = Field(default=None, description="Range low, only from a CLAIMS line.")
    high: float | None = Field(default=None, description="Range high, only from a CLAIMS line.")
    unit: str | None = Field(default=None, description="The unit exactly as the CLAIMS line gives it.")
    timeframe_years: float | None = None
    source: str = Field(
        description="An evidence label from the block, like S3. Never a path id like P3.",
    )


class Risk(BaseModel):
    description: str
    mitigation: str
    source: str = Field(description="An evidence label like S4 whose text states this risk.")


class Recommendation(BaseModel):
    practice_id: PracticeEnum = Field(description="Exactly as given in the dossier's candidate list.")
    action: str = Field(description="What to do on this site, naming its crop, season or constraint.")
    mechanism: str
    mechanism_sources: list[str] = Field(
        default_factory=list,
        description="One or more evidence labels (S1, S2...) whose text supports the mechanism. "
                    "Pick from the candidate's evidence_labels in the dossier. Never empty.",
    )
    mechanism_paths: list[str] = Field(
        default_factory=list,
        description="Path ids (P1, P2...) from the dossier whose causal chain this mechanism follows.",
    )
    impacted_metrics: list[MetricEnum] = Field(default_factory=list)
    variables_considered: list[MetricEnum] = Field(
        default_factory=list, description="At least three distinct metrics.",
    )
    estimates: list[Estimate] = Field(
        default_factory=list,
        description="Empty unless a CLAIMS line in a cited chunk gives a number for this practice.",
    )
    time_horizon: TimeHorizon
    risks: list[Risk] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    claim: str
    type: Literal["cause", "effect", "tradeoff", "synergy", "conflict"]
    path_id: str | None = Field(
        default=None, description="A path id from the dossier's paths list, like P4, for a causal claim.",
    )
    sources: list[str] = Field(
        default_factory=list, description="Evidence labels like S2 for an evidence claim. Never path ids.",
    )


class Adjudication(BaseModel):
    """The only structured output of the reasoning call."""

    reasoning_chain: list[ReasoningStep] = Field(default_factory=list)
    agrees_with_ranking: bool = True
    disagreement_reason: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)


# ── verification and confidence ────────────────────────────────────────
class VerificationReport(BaseModel):
    attempts: int = 0
    checks: dict[str, str] = Field(default_factory=dict)
    stripped_numbers: int = 0
    dropped_steps: int = 0
    dropped_recommendations: int = 0
    passed: bool = False


class ConfidenceBreakdown(BaseModel):
    evidence: float
    context: float
    agreement: float
    data_quality: float


class ConfidenceReport(BaseModel):
    value: float
    band: Literal["high", "medium", "low"]
    breakdown: ConfidenceBreakdown


# ── retrieval ──────────────────────────────────────────────────────────
class PlannedQuery(BaseModel):
    q: str
    purpose: QueryPurpose
    practice_id: str | None = None
    climate_zone: str | None = None
    path_id: str | None = None


class EvidenceItem(BaseModel):
    """One labelled chunk offered to the reasoning call."""

    label: str
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.0
    evidence_level: str | None = None
    content_role: str | None = None
    climate_zones: list[str] = Field(default_factory=list)
    practices: list[str] = Field(default_factory=list)
    claims: list[dict] = Field(default_factory=list)


# ── API ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str
    profile_patch: dict[str, Any] | None = None


class AnalyzeRequest(BaseModel):
    profile: dict[str, Any]
    audience: Audience | None = None
    constraints: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    q: str
    climate_zone: str | None = None
    practice: str | None = None
    limit: int = 10


class ChatResponse(BaseModel):
    trace_id: str
    session_id: str
    turn: int
    intent: Intent | None = None
    answer: str
    question: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    confidence: ConfidenceReport | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
