"""Ablation study: does the machinery around the model actually change the answer?

Configs
  A  LLM only               profile in, no retrieval, no rules, no graph
  B  RAG only               retrieval + the model, no rules, no graph, no verification
  C  RAG + rules + graph    the full dossier, but nothing is verified
  D  Full system            C plus V1-V10 and the degrade path

Run:  python evals/run_eval.py [--cases TC-02 TC-03] [--configs A B C D]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core import llm  # noqa: E402
from core.schemas import Adjudication, EvidenceItem  # noqa: E402
from generation import adjudicate as adjudicate_mod  # noqa: E402
from graph_flow import build  # noqa: E402
from reasoning import (  # noqa: E402
    causal,  # noqa: E402
    combine,
    diagnose,
    sequence,
)
from reasoning import graph as graph_mod  # noqa: E402
from retrieval import planner  # noqa: E402
from verification import checks  # noqa: E402

CASES_DIR = Path(__file__).parent / "test_cases"
RESULTS_PATH = Path(__file__).parent / "results.json"
REPORT_PATH = Path(__file__).parent / "report.md"

LLM_ONLY_SYSTEM = """\
You are an environmental scientist. Given a site profile, recommend land management
practices. Fill the schema. Use practice_id values from the enum, list the variables
you considered, and give estimates with units where you can.
"""


def load_cases(wanted: list[str] | None) -> list[dict]:
    cases = []
    for path in sorted(CASES_DIR.glob("TC-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if "profile" not in case:
            continue
        if wanted and case["id"] not in wanted:
            continue
        cases.append(case)
    return cases


# ── the four configurations ────────────────────────────────────────────
def run_config_a(case: dict) -> dict:
    """No retrieval, no rules, no graph. A strong prompt and nothing else."""
    result = llm.generate_json(
        system=LLM_ONLY_SYSTEM,
        user=json.dumps({"site": case["profile"]}, indent=2),
        schema=Adjudication,
        model=config.REASONING_MODEL,
        temperature=config.ADJUDICATE_TEMPERATURE,
    )
    return {"adjudication": result, "evidence": [], "excluded": [], "flags": []}


def _retrieve_for(case: dict, practices: list[str], flags: list[str]):
    queries = planner.plan(flags, practices, [], case["profile"].get("climate_zone"))
    items, traces, _ = planner.run_plan(queries, case["profile"].get("climate_zone"))
    return items, traces


def run_config_b(case: dict) -> dict:
    """Retrieval and the model, with no engine output to reason over."""
    items, _ = _retrieve_for(case, [], [])
    dossier = {"site": case["profile"],
               "note": "No diagnosis or causal analysis was supplied."}
    result = adjudicate_mod.adjudicate(dossier, items)
    return {"adjudication": result, "evidence": items, "excluded": [], "flags": []}


def _full_dossier(case: dict):
    profile = case["profile"]
    constraints = case.get("constraints", [])
    flags, _ = diagnose.diagnose(profile)
    names = diagnose.flag_names(flags)
    patterns = [p.model_dump() for p in diagnose.detect_patterns(names)]

    graph = graph_mod.load_graph()
    causes = causal.root_causes(graph, list(causal.flag_targets(names)), profile)[:8]
    ranking = causal.leverage(graph, names, profile)
    ranked, excluded = combine.candidates(names, profile, constraints)
    combos = combine.pair_candidates(ranked, names)
    plan = sequence.sequence([c["practice_id"] for c in ranked[:4]])

    items, traces = _retrieve_for(case, [c["practice_id"] for c in ranked[:4]], names)
    dossier = adjudicate_mod.build_dossier(profile, names, patterns, causes, ranking,
                                           ranked, combos, excluded, plan)
    return dossier, items, traces, names, ranked, excluded, causes


def run_config_c(case: dict) -> dict:
    """The whole dossier, but whatever the model says is published unchecked."""
    dossier, items, _, names, ranked, excluded, _ = _full_dossier(case)
    result = adjudicate_mod.adjudicate(dossier, items)
    return {"adjudication": result, "evidence": items, "excluded": excluded,
            "flags": names, "candidates": ranked}


def run_config_d(case: dict) -> dict:
    """The shipped system, verification and all."""
    state = build.run_analysis(case["profile"], case.get("constraints", []))
    raw = state.get("adjudication")
    return {
        "adjudication": Adjudication.model_validate(raw) if raw else Adjudication(),
        "evidence": [EvidenceItem.model_validate(item) for item in state.get("evidence", [])],
        "excluded": state.get("excluded", []),
        "flags": state.get("flags", []),
        "candidates": state.get("candidates", []),
        "verification": state.get("verification", {}),
        "confidence": state.get("confidence"),
    }


RUNNERS = {"A": run_config_a, "B": run_config_b, "C": run_config_c, "D": run_config_d}
LABELS = {
    "A": "LLM only",
    "B": "RAG only",
    "C": "RAG + rules + graph",
    "D": "Full system",
}


# ── metrics ────────────────────────────────────────────────────────────
def score(case: dict, outcome: dict) -> dict:
    """Grounding, citation validity, generic language, exclusions and risk detection."""
    adjudication: Adjudication = outcome["adjudication"]
    items: list[EvidenceItem] = outcome["evidence"]
    evidence = {item.label: item for item in items}

    numbers_total = numbers_grounded = 0
    citations_total = citations_valid = 0
    variables_ok = 0

    for recommendation in adjudication.recommendations:
        variables = {metric.value for metric in recommendation.variables_considered}
        if len(variables) >= config.MIN_VARIABLES_PER_RECOMMENDATION:
            variables_ok += 1

        labels = (list(recommendation.mechanism_sources)
                  + [estimate.source for estimate in recommendation.estimates]
                  + [risk.source for risk in recommendation.risks])
        for label in labels:
            citations_total += 1
            if label in evidence:
                citations_valid += 1

        for estimate in recommendation.estimates:
            if estimate.value is None and estimate.low is None and estimate.high is None:
                continue
            numbers_total += 1
            item = evidence.get(estimate.source)
            if item and any(
                checks._claim_supports(claim, estimate.metric.value,
                                       checks._unit(estimate.unit), estimate.value,
                                       estimate.low, estimate.high)
                for claim in item.claims
            ):
                numbers_grounded += 1

    text = " ".join(
        f"{recommendation.action} {recommendation.mechanism}"
        for recommendation in adjudication.recommendations
    ).lower()
    generic_hits = sum(1 for phrase in config.GENERIC_PHRASES if phrase in text)

    recommended = {
        recommendation.practice_id.value
        for recommendation in adjudication.recommendations
    }
    should_exclude = set(case["expected"].get("must_exclude", []))
    violations = sorted(recommended & should_exclude)

    expected_downgrade = set(case["expected"].get("must_downgrade", []))
    risk_detected = (
        not (recommended & expected_downgrade) if expected_downgrade else None
    )

    wanted = set(case["expected"].get("must_recommend_any", []))
    hit_wanted = bool(recommended & wanted) if wanted else None

    n = len(adjudication.recommendations)
    return {
        "recommendations": n,
        "numbers_total": numbers_total,
        "grounding_rate": round(numbers_grounded / numbers_total, 3) if numbers_total else None,
        "citation_validity": round(citations_valid / citations_total, 3) if citations_total else None,
        "generic_phrase_hits": generic_hits,
        "must_exclude_violations": violations,
        "risk_detected": risk_detected,
        "recommended_expected_practice": hit_wanted,
        "three_variable_compliance": round(variables_ok / n, 3) if n else None,
        "verification": outcome.get("verification", {}).get("checks"),
        "confidence": (outcome.get("confidence") or {}).get("value"),
    }


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(value) if value else "none"
    return str(value)


def write_report(results: list[dict]) -> None:
    """A table per case, plus the honest notes about what this does and does not show."""
    lines = ["# Ablation study", "",
             "Each configuration answers the same cases. Config D is the shipped system.",
             ""]
    for label, description in LABELS.items():
        lines.append(f"- **{label}** - {description}")
    lines.append("")

    columns = [
        ("grounding_rate", "numbers traced to a claim"),
        ("citation_validity", "citations that exist"),
        ("three_variable_compliance", "recs with >=3 variables"),
        ("generic_phrase_hits", "generic phrases"),
        ("must_exclude_violations", "excluded practice recommended"),
        ("risk_detected", "risky practice kept out of the top set"),
        ("confidence", "reported confidence"),
    ]

    for entry in results:
        lines.append(f"## {entry['case']} - {entry['description']}")
        lines.append("")
        header = "| metric | " + " | ".join(LABELS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(LABELS) + 1))
        for key, human in columns:
            row = [human]
            for name in LABELS:
                row.append(_cell((entry["configs"].get(name) or {}).get(key)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for name in LABELS:
            failure = (entry["configs"].get(name) or {}).get("error")
            if failure:
                lines.append(f"- config {name} failed: {failure}")
        lines.append("")

    lines += [
        "## Reading this honestly", "",
        "- Config A has no evidence block, so its citations and numbers cannot be",
        "  checked at all; a 0 there means unverifiable, not necessarily wrong.",
        "- Configs B and C see the same evidence as D. A citation validity below 1.0",
        "  there means the model cited a label that was never given to it, which is",
        "  exactly what verification exists to catch.",
        "- n/a for the risky-practice row means the case has no downgraded practice",
        "  to check.",
        "- `risk_detected` is coarse: it only asks whether a practice the engine",
        "  downgrades stayed out of the recommendation set.",
        "- Sample sizes here are small. Treat the table as a direction, not a",
        "  measurement, and say so in the submission.",
        "- Judging Gemini output with Gemini inflates agreement. The checks counted",
        "  here are mechanical except V7.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--configs", nargs="*", default=list(RUNNERS))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        sys.exit("no test cases with a profile matched")

    results = []
    for case in cases:
        entry = {"case": case["id"], "description": case["description"], "configs": {}}
        for name in args.configs:
            started = time.perf_counter()
            print(f"  {case['id']} config {name} ...", flush=True)
            try:
                outcome = RUNNERS[name](case)
                measured = score(case, outcome)
            except Exception as exc:  # noqa: BLE001 - a failed config is a result
                measured = {"error": f"{type(exc).__name__}: {exc}"}
            measured["seconds"] = round(time.perf_counter() - started, 1)
            entry["configs"][name] = measured
        results.append(entry)

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    write_report(results)
    print(f"\nwrote {RESULTS_PATH.name} and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
