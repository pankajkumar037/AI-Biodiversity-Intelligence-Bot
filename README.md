# Darukaa — AI Biodiversity Intelligence System

A land-management advisor that behaves like an environmental scientist rather than a chatbot.
The diagnosis, the causal simulation, the ranking and the confidence score are all computed in
Python. Gemini explains the result and challenges it. Nothing reaches the user until it has been
checked against a retrieved source.

The design rule this repository follows: **what the code decides is separate from what the
language model explains.**

**Live demo:** https://ai-biodiversity-intelligence-bot.onrender.com/ (free Render tier — the
first request after idle takes ~50 s to wake). 

---

## Architecture

```mermaid
flowchart TD
    START([user message]) --> intent{intent<br/>Flash-Lite classifier}

    intent -->|out_of_scope| oos[polite redirect] --> END([answer])
    intent -->|concept| concept[quote the corpus,<br/>no site reasoning] --> END
    intent -->|new_info / constraint / what_if| intake[intake<br/>Flash-Lite extraction]
    intake --> normalize[normalize<br/>units, validation]
    normalize --> geo[geo_enrich<br/>SoilGrids, NASA POWER, Nominatim]
    geo --> slot{slot_check<br/>value of information}

    slot -->|one question worth asking| ask[ask exactly one question] --> END
    slot -->|enough is known| diagnose[diagnose<br/>thresholds + compound rules]

    diagnose --> root[root_cause<br/>reverse graph traversal]
    root --> lev[leverage<br/>reach over flagged metrics]
    lev --> cand[candidates<br/>hard filters + risk penalties]
    cand --> comb[combine<br/>synergy pairing]
    comb --> seq[sequence<br/>prerequisite ordering]
    seq --> plan[plan_queries<br/>support + risk + path]
    plan --> ret[retrieve<br/>vector + text, RRF, rerank]
    ret --> adj[adjudicate<br/>one Flash call]
    adj --> ver{verify<br/>V1–V10}
    ver -->|failures, retries left| adj
    ver -->|passed or degraded| render[render<br/>pure Python template]
    render --> END
```

Every box above is a LangGraph node. The conditional edges — not the model — decide whether the
system asks a question, runs the full pipeline, or answers from memory.

Intent is classified before anything is read into the profile, so a general question is
never mined for site values it does not contain. One route deliberately bypasses the reasoning
engine: `concept` answers a general question by
quoting the corpus with citations and no LLM in the loop at all, then says plainly that this is
general evidence rather than advice for the user's site.

### Division of responsibility

| Component | Responsible for | Never responsible for |
|---|---|---|
| LangGraph | control flow, state, retries, checkpoints | content decisions |
| Rule engine + causal graph | diagnosis, filtering, simulation, sequencing | natural language |
| Retrieval | finding evidence, scoring, labelling it | deciding what is true |
| Gemini | extraction, explanation, critique of the ranking | profile values, confidence, effect sizes |
| Verification | guaranteeing every claim is grounded | blocking the response entirely |

---

## Data model

MongoDB Atlas, database `darukaa`.

| Collection | Count | Contents |
|---|---|---|
| `chunks` | 1008 | passages, 768-dim embeddings, extracted claims |
| `variable_graph` | 505 | causal edges (342 traversable) |
| `practices_draft` | 16 | extracted practice profiles, source material for the cards |
| `sessions` | per user | site memory, constraints, per-turn traces |

Two Atlas indexes on `chunks`:

- `vector_index` — 768 dims, cosine, filters on `climate_zones`, `practices`, `metrics`,
  `evidence_level`, `content_role`, `context_only`, `doc_id`
- `text_index` — Atlas Search over `text`, `topic`, `doc_title`

A chunk carries its own provenance (`doc_id`, `page_start`, `evidence_level`) and a `claims`
array. **A claim is the only thing that licenses a number in the output.**

```json
{
  "metric": "SOC", "direction": "increase", "value": 7.3, "unit": "percent",
  "basis": "mean across 61 cover-crop studies",
  "evidence_span": "verbatim sentence containing the number"
}
```

### The knowledge layer

RAG answers *what does the literature say*. These files answer *what does this site need*:

| File | Holds |
|---|---|
| `knowledge/thresholds.json` | metric bands and categorical rules, each with its source |
| `knowledge/compound_rules.json` | interacting problem patterns such as the carbon–water trap |
| `knowledge/practices/*.json` | 19 practice cards: what each addresses, blocks on, risks |
| `knowledge/regional_defaults.json` | zone-level fallbacks, each with its source |

Practice cards carry **no effect sizes**. Numbers come only from retrieved claims, so every
figure shown to a user carries a citation.

---

## How the knowledge base was built

The extraction pipeline lives in `data/notebook/Biodiversity_data_extraction.ipynb`; its
output is in `data/extracted`, `data/graph` and `data/manifests`, and `db_ingestion/ingest.py`
loads it into Atlas.

### The corpus: 14 documents, chosen by role

| Role | Documents | Why |
|---|---|---|
| **Practice evidence** — what to do | FAO *Recarbonizing Global Soils* Vol. 3 and Vol. 4 | Vol. 3 documents 49 soil practices in a fixed format including drawbacks and constraints, which is why it maps almost directly onto practice cards. Vol. 4 adds 51 field case studies with the conditions attached |
| **Authority** — what is established | IPCC SRCCL 2019; IPBES Land Degradation and Restoration 2018; IPBES Pollinators 2016; FAO SoWBFA 2019 | SRCCL gives the land–climate–carbon links; IPBES gives the degradation–biodiversity links SRCCL is thin on; the pollinator and SoWBFA reports cover species, pollination and soil biota |
| **Quantified evidence** | Joshi et al. 2023 (open-access cover-crop meta-analysis); Keerthika et al. 2026 (semi-arid Rajasthan field trial) | The numbers the system is allowed to quote |
| **Indian grounding** | Handa et al. 2019 (agroforestry models by agro-ecological region); Tewari et al. (hot arid systems) | Regional evidence outranks global when the site matches |
| **Human impact** | FAO Soil Pollution 2018; India State of Forest Report 2023 | Pollution and land-cover change |

Two selection rules drove everything: every practice needs both a supporting source and a
source describing its risks, and regional evidence outranks global when the site matches.

### Why not index whole PDFs

The documents total ~3,500 pages. Most of it is front matter, methodology, references and
off-topic chapters. We read each table of contents, computed the printed-page-to-PDF-page
offset, and indexed only the useful ranges — about 380 pages. Chapter 7 of ISFR (agroforestry)
went in; 45 pages of forest-fire monitoring did not.

### The extraction strategy

**The core problem.** The naive approach is: send a PDF to an LLM, get JSON back. That works for
a 30-page chapter. These documents are 150–870 pages, and a single call over 870 pages either
exceeds limits or skims — the model returns a shallow summary of an enormous input. So the unit
of extraction had to shrink. The insight was that you can find those units without an LLM.

**Stage 0 — structure first, deterministically.** Every document has a table of contents. We read
it, computed the offset between printed page numbers and PDF page indices (IPCC +10, IPBES +58,
FAO Vol. 3 +18), and derived exact page ranges for each section. Zero tokens spent. That gave a
natural unit boundary and the ability to drop 85% of each document before extraction: IPCC has
seven chapters and three are relevant; ISFR has a 45-page forest-fire chapter that will never
answer a question about soil carbon. Every slice was verified by checking that its first page
contains the heading its name claims — which caught a real error in the agroforestry manual,
where the first offset was wrong. 3,500 pages → 139 unit PDFs → ~380 pages actually extracted.

**Stage 1 — the unit is the section, not the document.**

| Document | Unit | Why |
|---|---|---|
| FAO Vol. 3 | one practice entry | already organised the way the knowledge base needs |
| FAO Vol. 4 | one case study | self-contained, with conditions stated |
| Handa 2019 | one agroforestry model | fixed fields: species, region, rainfall |
| IPCC / IPBES | 12-page overlapping window | continuous prose with no useful boundaries |
| Journal papers | whole file | already short |

One page of overlap between windows means a finding split across a boundary is captured by one
side or the other.

**Stage 2 — one prompt per document type.** This was the decision that mattered most. A single
prompt produced 2.0 passages per page from FAO manuals and 0.33 from IPCC/IPBES assessments —
not because assessments are less useful, but because they are written differently. FAO states
practices; IPCC states synthesised findings buried in citation clutter and confidence language.
Three prompts resulted:

- **Practice prompt** — description, mechanism, evidence, constraints; fill a practice profile
  with species, regions, rainfall bands and every stated drawback.
- **Assessment prompt** — target 10–20 passages per window; strip inline citations from passage
  text (IPCC is unreadable otherwise) but keep confidence language, because "high confidence" is
  the finding; prioritise statements linking two or more variables; map confidence wording onto
  edge strength.
- **Inventory prompt** (ISFR, GSOCseq) — read tables and charts as content, never estimate a
  value from a bar height, and return zero claims and zero links, always. A statistical inventory
  reports areas, not practice effects. Letting "forest cover increased by 1,445 km²" become a
  claim would poison the recommendation engine.

The re-run with the assessment prompt: 145 → 402 passages, 99 → 238 links, conditional links
19 → 62. Same documents, same model.

**Stage 3 — extraction with receipts.** Two rules made the output verifiable. Passage text is
copied, never paraphrased: the model chooses boundaries and labels, it does not rewrite, so
citations point at real sentences. And every claim carries `evidence_span`, the exact sentence
containing the number — the mechanism that makes downstream verification possible at all. The
sliced PDF pages went to Gemini rather than extracted text, because text extraction scrambles
multi-column layouts and destroys tables; the FAO practice tables and ISFR land-use tables
survived intact.

**Stage 4 — validation.** Every claim's span was checked as a substring of pypdf's text
extraction. Initial failure rate: 13%. Investigating the worst file (Joshi, 16 of 38 failing)
showed the failures were encoding artefacts, not fabrication: `Mg ha⁻¹` rendered differently by
the two readers, `ln(R)` extracted as `In(R)`. After Unicode normalisation, zero genuine failures
on that file and 2% across the assessment corpus. That distinction mattered — a naive reading
would have called it a 13% hallucination rate and triggered a pointless re-extraction.

**Stage 5 — normalisation into a graph.** Raw extraction produced 646 links across 353 distinct
node names — `cover_crops`, `legume cover crops`, `soil organic carbon`, `SOC`, `soil carbon
stocks` all appearing separately. The same edge counted three times fragments the graph. We
built an alias map (exact matches), regex families (anything matching `fertili[sz]` →
`fertilizer_use`), and promoted genuinely new concepts into the vocabulary rather than forcing
them into existing buckets. Nodes outside the reasoning scope — REDD+, farm subsidies,
migration — were tagged peripheral and kept but excluded from traversal, so "what about policy
drivers?" can be answered with "we have that knowledge and deliberately exclude it from
site-level advice." 353 → 165 nodes, 646 raw links → 505 unique edges.

Then the filter. The first instinct was `support_count ≥ 2`, which would have kept 61 of 477
core edges — discarding single-source findings from IPCC with "high confidence" attached. The
right filter is evidence quality:

```
keep if support_count >= 2
     or strength in {strong, moderate}
     or source is an assessment or meta-analysis
```

342 traversable edges, 130 conditional.

**What the strategy produced** — engineering choices, not luck: `vegetation_cover → SOC`
positive above 336 mm rainfall and negative below (same driver, opposite sign, numeric
threshold); `agroforestry → SOC (decrease)` in dry climates, on the most-supported practice in
the corpus; `irrigation → salinity` in drylands, appearing twice. None of those survive a
pipeline that summarises instead of extracting, or that discards conditions to keep edges simple.

The general principle: deterministic work stays deterministic (finding sections, slicing pages,
validating spans, deduplicating nodes), and the LLM is used only where judgment is genuinely
required — choosing passage boundaries, labelling roles, and identifying what a document asserts.

### Why the JSON schema has four levels

```
UnitExtraction
├── passages[]        verbatim text        → embedded, retrieved, cited
├── claims[]          numbers + receipts   → verification
├── links[]           cause → effect       → causal graph
└── practice_profile                       → practice cards
```

`passages` alone would be an ordinary RAG. The other three exist for specific reasons:

- **`claims` carry `evidence_span`**, the exact sentence containing the number. That is what
  lets Python check the model's output: if it writes "SOC +25% [S4]" and S4's claims say 7.3%,
  the check fails and the number is stripped. Without claims you are trusting the model.
- **`links` carry a `condition`.** `cover_crops → soil_moisture (decrease, condition:
  arid|semi-arid)` is what lets the system downgrade a practice for one site and recommend it
  for another. That conditionality is the difference between reasoning and cheerleading.
- **`content_role` on each passage** (evidence, mechanism, constraint, risk) means a "what could
  go wrong" query filters directly to trade-off text instead of hoping it ranks well.

### Result

104 units → 1,008 passages, 649 claims (144 with numbers), 646 links → 505 deduplicated edges,
342 traversable, 130 conditional.

The graph found things worth having: `vegetation_cover → SOC` is positive above 336 mm rainfall
and negative below it. Agroforestry, the best-supported practice, carries a documented SOC
penalty in dry climates. Irrigation raises salinity in drylands.

### Why MongoDB Atlas

| Need | Why Mongo |
|---|---|
| Vector search | Atlas has it built in, on the free M0 tier |
| Keyword search | Atlas Search in the same database; hybrid via reciprocal rank fusion |
| Nested metadata | Claims live inside their chunk as a sub-document. In a relational store that is a join; here it arrives with the retrieval result, so verification is free |
| Graph + practices + memory | Separate collections, one connection |
| Conversation state | LangGraph's MongoDB checkpointer writes to the same cluster |

The alternative was a vector DB plus Postgres plus a separate session store — three services
for a hackathon. One database holding vectors, text index, structured knowledge and memory is
simpler to explain and simpler to deploy.

Embeddings are `gemini-embedding-001` at 768 dimensions rather than the 3072 default: it fits
the free tier comfortably and avoids loading a local embedding model into the web process,
which would exhaust free hosting memory during the live demo.

---

## The reasoning that makes this non-obvious

For the brief's own semi-arid, low-rainfall test case the system **downgrades cover crops on its
own**, even though cover cropping is the textbook answer to low soil carbon:

```
CANDIDATES (semi-arid, SOC 0.3%, 400 mm, wheat monoculture)
  0.769  intercropping             coverage 0.75  benefit +3.34
  0.732  mulching                  coverage 0.50  benefit +2.00
  ...
  0.562  cover_crops               coverage 0.75  benefit +0.97  penalty 0.25
         downgraded: cover crop water use competes with the cash crop in dry conditions
```

Three independent mechanisms produce that result:

- a condition-aware graph edge, `cover_crops → soil_moisture (decrease)` under `semiarid regions`
- a `soft_risk` on the practice card, firing on `climate_zone in [arid, semi-arid]`
- a **risk query** issued for every candidate, which retrieves FAO Vol. 4 p227: *"Competition for
  water with the main crop can be promoted, especially in dry years"*

The same practice scores differently on different land, because `condition_holds()` evaluates each
edge against the site rather than applying the evidence globally.

---

## Verification

| # | Check | On failure |
|---|---|---|
| V1 | structure valid, practice is a real candidate, and it addresses something the user said is wrong | retry |
| V2 | every `S#` exists in the evidence block | retry |
| V3 | every number matches a claim: same metric, same unit, value in range | retry, then strip the number |
| V4 | direction agrees with the claim or a graph edge | retry |
| V5 | cited chunk's climate fits the site | lower confidence |
| V6 | mechanism has a source and a risk query ran for that practice | retry |
| V7 | mechanism is supported by the cited text (Flash-Lite judge) | one retry, then soften |
| V8 | no generic filler | retry |
| V9 | every reasoning step anchors to a real path id or `S#`; a cause/effect step anchored only to evidence is judged against that text | strip the step |
| V10 | each recommendation connects ≥3 environmental variables | retry, drop if impossible |

Verification never blocks a response. When retries are spent the answer degrades: unverifiable
numbers are stripped, unanchored reasoning steps are removed, confidence is multiplied down, and
the trace records exactly what failed.

The model is never shown a number it could parrot. The dossier it reasons over is qualitative —
rank order, `raises`/`lowers` lists, path directions — and the only figures anywhere in its
input are on `CLAIMS` lines in the evidence block. Path ids and evidence labels are separate
families with separate fields (`mechanism_paths`, `mechanism_sources`), so a citation can always
be checked against the right thing.

`confidence = evidence × context × agreement × data_quality`, always reported with its breakdown.

---

## What an answer looks like

Sections run in a fixed order, and the question decides the lead:

```
SITE         every value with its source (user / soilgrids / nasa_power / inferred)
DIAGNOSIS    flags with the rule ids that fired, compound patterns
CAUSES       "You reported: ..." then root-cause chains, observed drivers first
WHAT CHANGED baseline vs hypothetical, what moved, what stayed, what still binds  (what-if only, leads)
EVIDENCE     the cited chunks with document, page and a line of text
RECOMMENDED  an order note when the top-ranked practice is not first, then cards
TRADE-OFFS   downgraded practices with the penalty applied, and the model's trade-off steps
EXCLUDED     what was ruled out and why
CONFIDENCE   the four factors
```

A "why" question leads with CAUSES. A what-if runs on a hypothetical copy of the profile and
the baseline is restored on the next turn, so a hypothetical never leaks forward. A message that
names a different land use or place starts a fresh profile rather than merging into the old one.

Anything the user says is wrong on their land — pesticide use, a pollinator decline, residue
burning, recent clearing, a nearby pollution source, visible erosion — becomes a profile field,
a flag, a target in the causal graph and the first retrieval queries. These *problem flags* count
double in scoring, and a recommendation that addresses none of them fails verification.

---

## Local setup

Python 3.13 (the pins in `requirements.txt` need ≥3.12).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

cp .env.example .env            # then fill in the three values
uvicorn main:app
```

Then open http://127.0.0.1:8000/ — the UI is served from `frontend/` on the same origin.
`--reload` is unreliable on Windows; restart the process after backend edits.

`.env` needs `GEMINI_API_KEY`, `db_password`, and `MONGODB_URI` containing the literal
`<db_password>` placeholder. `.env` is gitignored and must stay that way.

```bash
pytest tests/ evals/ -q         # 118 tests, no database or API key needed
ruff check .
python evals/run_eval.py        # the A/B/C/D ablation (makes live API calls)
```

### Deploying

One service serves both the API and the UI. `render.yaml` describes it for Render (free plan
works; add the three env vars in the dashboard and set `PYTHON_VERSION` to 3.13). `Dockerfile`
covers Cloud Run, Railway or anything else. In Atlas → Network Access, allow the host's egress
IPs or `0.0.0.0/0`. The stream sends a heartbeat every 15 s so proxies do not drop a turn while
the model is thinking.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | database ping, collection counts, Atlas index status |
| POST | `/chat` | one conversational turn: `{session_id, message, profile_patch?}` |
| POST | `/chat/stream` | the same turn as server-sent events, one per pipeline node, then the result |
| POST | `/analyze` | structured profile in, full analysis out, no conversation |
| POST | `/analyze/stream` | streamed twin of `/analyze` |
| GET | `/session/{id}` | stored profile, constraints, recommendation history |
| POST | `/session/{id}/reset` | clear the thread and the site memory |
| GET | `/trace/{session_id}/{turn}` | the full reasoning trace for one turn |
| GET | `/knowledge/stats` | corpus size and indexed documents |
| POST | `/search` | debug: raw hybrid retrieval for one query |
| GET | `/ui/` | the frontend (`/` redirects here) |

Every response carries a `trace_id`. Errors return a JSON body, never a stack trace.

### The trace

`/chat` and `/analyze` return a `trace` built up as the pipeline runs — profile with per-field
provenance, flags and the rule ids that fired, root-cause paths, leverage ranking, candidate
scores with the risks applied, exclusions with reasons, every query with its filter level and
what it kept, the evidence with scores, the verification result, and the confidence breakdown.

It is the part of the system that shows its working, and it is built incrementally in graph
state rather than reconstructed at the end.

---

## Frontend

`frontend/` is three files — HTML, CSS, JS — with no build step, served by FastAPI. One centred
conversation. Each answer opens with a thinking block that streams the pipeline's steps as they
finish (from `/chat/stream`) and collapses to "Reasoned for 38s · 4 flags · 12 sources" when
done; the answer then reveals section by section. Evidence, reasoning, profile, verification and
the traversed causal graph are inline toggles under each answer. Input modes: free text,
structured JSON (goes to `/analyze`), or coordinates (reverse-geocoded, then SoilGrids and
NASA POWER). Send becomes Stop while a turn runs.

Only the pipeline progress is truly streamed; the answer text is assembled by the Python
renderer after verification and revealed client-side.

---

## Known limitations

- **This is a qualitative causal reasoner, not a process model.** It is strong on direction,
  interactions and trade-offs; it does not predict magnitudes. Coupling something like RothC
  would be the honest way to get quantitative projections.
- **Engine scores are for ranking only.** `suitability` and `net_effects` order options; they are
  not real-world effect sizes, and they are labelled that way in the dossier and the API.
- **Threshold bands need sourcing.** Several rules in `thresholds.json` carry `"verify": true` —
  they are defensible starting points, not values traced to a cited table. The SOC bands follow
  Indian Soil Health Card classes; the natural-cover and habitat-diversity bands are estimates.
- **`condition_holds()` is deliberately conservative.** A condition it cannot interpret returns
  false, so edges qualified by things like "legume cover crops" or "duration <5 years" are
  skipped rather than assumed. This loses some real evidence to avoid asserting effects that may
  not apply here.
- **Risk edges read the full graph, not just traversable ones.** Several well-evidenced
  trade-offs are marked non-traversable in the extracted data, and dropping them would hide the
  caveats a recommendation needs to carry.
- **Latency is 25–170 s for a full reasoning turn**, depending on the reasoning model and how
  many verification passes it takes. A first-pass-clean turn on Flash-Lite is about 25 s.
- **Some corpus chunks are bibliography fragments.** The ingestion split a few reference-list
  passages as if they were prose, and one can surface as evidence. A cleaning pass over
  `data/extracted` is the fix.
- **Landscape metrics are not computed.** Habitat-diversity and fragmentation thresholds exist
  and fire when the user supplies the value; ESA WorldCover and GBIF are not wired.
- **Regional defaults are indicative.** `regional_defaults.json` is not yet traced to an ICAR
  table and is marked for verification.
- **The ablation study is small.** Treat `evals/report.md` as a direction, not a measurement.
- **Judging Gemini with Gemini inflates agreement.** Only V7 is model-judged; every other check
  is mechanical. A different model family would be the better judge.
