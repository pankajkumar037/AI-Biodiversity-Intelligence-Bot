# Ablation study

Each configuration answers the same cases. Config D is the shipped system.

- **A** - LLM only
- **B** - RAG only
- **C** - RAG + rules + graph
- **D** - Full system

## TC-02 - A fully specified semi-arid site needs no clarifying question.

| metric | A | B | C | D |
|---|---|---|---|---|
| numbers traced to a claim | 0.0 | n/a | 0.0 | 1.0 |
| citations that exist | 0.0 | 1.0 | 0.0 | 1.0 |
| recs with >=3 variables | 1.0 | 1.0 | 1.0 | 1.0 |
| generic phrases | 0 | 0 | 0 | 0 |
| excluded practice recommended | none | none | none | none |
| risky practice kept out of the top set | no | yes | yes | yes |
| reported confidence | n/a | n/a | n/a | 0.68 |


## TC-03 - A humid site must not inherit the dryland water-competition reasoning.

| metric | A | B | C | D |
|---|---|---|---|---|
| numbers traced to a claim | 0.0 | n/a | n/a | 1.0 |
| citations that exist | 0.0 | 1.0 | 0.111 | 1.0 |
| recs with >=3 variables | 1.0 | 1.0 | 1.0 | 1.0 |
| generic phrases | 0 | 1 | 1 | 0 |
| excluded practice recommended | none | none | none | none |
| risky practice kept out of the top set | n/a | n/a | n/a | n/a |
| reported confidence | n/a | n/a | n/a | 0.565 |


## Reading this honestly

- Config A has no evidence block, so its citations and numbers cannot be
  checked at all; a 0 there means unverifiable, not necessarily wrong.
- Configs B and C see the same evidence as D. A citation validity below 1.0
  there means the model cited a label that was never given to it, which is
  exactly what verification exists to catch.
- n/a for the risky-practice row means the case has no downgraded practice
  to check.
- `risk_detected` is coarse: it only asks whether a practice the engine
  downgrades stayed out of the recommendation set.
- Sample sizes here are small. Treat the table as a direction, not a
  measurement, and say so in the submission.
- Judging Gemini output with Gemini inflates agreement. The checks counted
  here are mechanical except V7.
