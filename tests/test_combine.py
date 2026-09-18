"""Hard constraints exclude, soft risks downgrade, and both must say why."""
from __future__ import annotations

from core.schemas import PracticeCard
from reasoning import combine, sequence

SITE = {"climate_zone": "semi-arid", "land_use": "cropland", "rainfall_mm": 400, "ph": 8.2}


def test_always_and_constraint_rules():
    assert combine.risk_applies("always", SITE, []) is True
    assert combine.risk_applies("constraint no_machinery", SITE, ["no_machinery"]) is True
    assert combine.risk_applies("constraint no_machinery", SITE, []) is False


def test_zone_membership_rule():
    assert combine.risk_applies("climate_zone in [arid, semi-arid]", SITE, []) is True
    assert combine.risk_applies("climate_zone in [humid]", SITE, []) is False


def test_numeric_comparison_rule():
    assert combine.risk_applies("rainfall_mm < 500", SITE, []) is True
    assert combine.risk_applies("rainfall_mm > 1000", SITE, []) is False
    assert combine.risk_applies("ph > 8.0", SITE, []) is True


def test_unparseable_rule_does_not_fire():
    assert combine.risk_applies("if the moon is full", SITE, []) is False
    assert combine.risk_applies("rainfall_mm < abc", SITE, []) is False


def _card(**overrides) -> PracticeCard:
    base = {
        "practice_id": "mulching",
        "name": "Test practice",
        "addresses_flags": ["low_SOC"],
        "hard_constraints": {"land_use_in": ["cropland"], "incompatible_constraints": []},
        "soft_risks": [],
        "requires_first": [],
        "impacted_metrics": ["SOC"],
        "mechanism": "test",
        "time_horizon": "short",
        "region_variants": [],
        "evidence_chunk_ids": [],
    }
    base.update(overrides)
    return PracticeCard.model_validate(base)


def test_user_constraint_blocks_a_practice():
    card = _card(hard_constraints={
        "land_use_in": ["cropland"], "incompatible_constraints": ["leased_land_no_trees"]
    })
    assert combine.blocking_constraint(card, SITE, ["leased_land_no_trees"]) == "leased_land_no_trees"
    assert combine.blocking_constraint(card, SITE, []) is None


def test_wrong_land_use_blocks_a_practice():
    card = _card(hard_constraints={"land_use_in": ["grassland"], "incompatible_constraints": []})
    assert combine.blocking_constraint(card, SITE, []) == "land use cropland not suitable"


def test_rainfall_floor_blocks_a_practice():
    card = _card(hard_constraints={
        "land_use_in": ["cropland"], "min_rainfall_mm": 600, "incompatible_constraints": []
    })
    assert "600" in combine.blocking_constraint(card, SITE, [])


def test_unknown_land_use_does_not_block():
    card = _card(hard_constraints={"land_use_in": ["grassland"], "incompatible_constraints": []})
    assert combine.blocking_constraint(card, {"climate_zone": "semi-arid"}, []) is None


def test_sequence_puts_a_prerequisite_first_and_pulls_it_in():
    cards = {
        "mulching": _card(practice_id="mulching", name="Mulching", time_horizon="short"),
        "intercropping": _card(practice_id="intercropping", name="Intercropping",
                               requires_first=["mulching"], time_horizon="medium"),
    }
    plan = sequence.sequence(["intercropping"], cards)
    assert [step["practice_id"] for step in plan] == ["mulching", "intercropping"]
    assert plan[0]["added_as_prerequisite"] is True
    assert plan[1]["added_as_prerequisite"] is False


def test_sequence_pushes_a_horizon_out_past_its_prerequisite():
    cards = {
        "mulching": _card(practice_id="mulching", name="Mulching", time_horizon="medium"),
        "intercropping": _card(practice_id="intercropping", name="Intercropping",
                               requires_first=["mulching"], time_horizon="medium"),
    }
    plan = sequence.sequence(["mulching", "intercropping"], cards)
    assert plan[0]["time_horizon"] == "medium"
    assert plan[1]["time_horizon"] == "long"
