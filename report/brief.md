# Comparability Breaks as Headroom in Multi-Step Financial Reasoning

**Author:** Sam Zaheer · **Prepared for:** Verita AI · **Date:** 2026-09-23
**Subfield:** SEC-filing analysis · **Route:** Construct (with a condensed Review and a Dissect pass on our own transcripts)

This document covers the research question, the design, the findings and the recommendation. Companion documents:
- [`docs/architecture.md`](../docs/architecture.md): the evaluation architecture, i.e. how both stages ran, grading and judge calibration, answer-key isolation and reward-hacking controls, Docker, and the path to production scale.
- [README](../README.md): setup, and reproducing every result.

## Executive summary

We tested whether frontier LLMs correctly handle **accounting comparability breaks** when they compute financial metrics from real SEC filings. The breaks tested were segment spinoffs, discontinued-operations restatements, fiscal-year misalignment and an acquisition. The models were Claude Sonnet 5, Claude Opus 5 and GPT-5.1, each tested with and without realistic tools. There were 30 items in 9 buckets.

The evaluation ran in two independent stages:
1. **Stage 1, local harness:** 1 iteration, then 3 iterations; 540 runs.
2. **Stage 2, Gymnasium environment:** the same design; 540 newly generated episodes.

**The pre-registered hypothesis is refuted.**
- **Multi-sector comparability breaks** were solved by every model, in both conditions, on every iteration, in both stages: **72/72 runs per stage**. This held under adversarial construction: neutral phrasing, evidence split across separate documents, and a genuine distractor figure.
- **The airline acquisition break** (Alaska/Hawaiian) was *detected* on every Stage 1 run (36/36). Claude and Opus also adjusted for it correctly on every run. GPT-5.1 detected it but sometimes failed to act on it.

| Model | Stage 1 · 3 iterations | Stage 2 · 3 iterations | 95% range (Stage 1) |
|---|---|---|---|
| Claude Opus 5 | **179/180** (99.4%) | 179/180 | 97–100% |
| Claude Sonnet 5 | **174/180** (96.7%) | 174/180 | 93–98% |
| GPT-5.1 | **154/180** (85.6%) | 156/180 | 80–90% |

GPT-5.1 is clearly behind the other two models. Opus and Sonnet 5 are statistically indistinguishable. The residual failures are narrow, reproduce across both stages, and are specific to particular models:
- margin-vs-profit confusion (stable in GPT-5.1, and partly a wording confound);
- judgment collapse on one open-ended comparison (stable in GPT-5.1);
- an acquisition detected but not adjusted for (GPT-5.1).

**Recommendation: do not pursue accounting-comparability detection as a training target.** None of the residual findings justifies a dedicated RL environment by itself.

## Research question

> Do frontier LLMs with realistic document-search and calculator tools recognize and correct for accounting comparability breaks — segment redefinitions, discontinued-operations restatements, fiscal-year misalignment — when computing and comparing financial metrics across periods and entities? Or do they compute a superficially plausible number that silently ignores the break?

## Subfield and route

**SEC-filing analysis.**
- **Trading and portfolio agents were ruled out first.** Their only ground truth is future market movement, which a computer cannot check against a fixed key and which is dominated by noise.
- **Agentic financial research was ruled out next.** It is already covered by three recent, well-resourced benchmarks.
- **SEC-filing analysis had a specific, evidenced sub-gap:** comparability-break mechanisms, which no existing benchmark tests as a controlled variable.

**Construct.** It is the only route that makes the hypothesized mechanism a controlled variable, with matched items with and without a break and every answer computed from raw filings. We borrowed from two other routes:
- a condensed **Review** of the literature establishes the gap;
- the failure taxonomy is a **Dissect** pass over our own transcripts.

We rejected two routes outright:
- **Pure Review** would repeat better-resourced surveys.
- **Convert** would rebuild FinQA-style arithmetic chains, which are already near-saturated once a calculator is available.

## Prior benchmark landscape

| Paper | Setup | Finding | Source |
|---|---|---|---|
| **Finance Agent Benchmark** (Vals AI + Stanford, [2508.00828](https://arxiv.org/abs/2508.00828)) | 537 expert questions, full agentic EDGAR harness | No model exceeds 50% (o3 best, 46.8%). Failures stem from tool misuse, not missing information | Table 2, §4.2.1–2 |
| **FinSearchComp** (ByteDance Seed + Columbia, [2509.13160](https://arxiv.org/abs/2509.13160)) | T1 single-fact → T3 multi-period, tool-ablated | Tools add +40.8 pp on T1 but only +8.1 pp on T3 | Fig. 1, 5, 6; §4.1 |
| **Fin-RATE** (Yale/UCSD/Tongji + Goldman Sachs AI Research, [2602.07294](https://arxiv.org/abs/2602.07294)) | Single-document, cross-entity and longitudinal QA, with ground-truth context | A 35–42 pp gap to human experts persists with perfect retrieval. Finance-tuned Fin-R1 falls from 57.5% (single-document) to 3.3% (cross-entity) | Table 2, abstract |

**Gap.** All three papers show that models degrade on cross-period and cross-entity synthesis, and that tools do not close the gap. None isolates *why*: entity confusion, period misalignment, or a specific accounting mechanism such as a spinoff reclassified as discontinued operations. This study isolates that mechanism, using an original dataset with a fully machine-checkable answer key.

## Dataset

**30 items in 9 buckets, built from real filings.**
- Every number was verified against raw SEC XBRL JSON or the filing text. A summarizer pass produced one hallucinated number during construction; it was caught and discarded.
- Each item has a frozen evidence corpus and an answer key. The key records source accession numbers, the relevant evidence, the expected intermediate steps, and either a numeric tolerance or a rubric written before any model run.

| Bucket | N | Basis |
|---|---|---|
| `tech_control` | 4 | Apple, Microsoft, Amazon, Meta: single-document FY2024 margin and revenue |
| `tech_clean_comparison` | 6 | Alphabet, Meta, Amazon, Microsoft, Apple: cross-period and cross-entity, no accounting break |
| `multi_sector_comparability_break` | 4 | Spinoffs reported as discontinued operations (IBM/Kyndryl, GE's compounding dual spinoff, 3M/Solventum), and Walmart/Oracle fiscal-year misalignment |
| `tech_gaap_adjustment` | 4 | Real 8-K exhibits: Workday (5-line add-back), ServiceNow (two tables, one requiring a subtraction), Datadog (GAAP loss, reported in thousands) |
| `tech_guidance_verification` | 2 | Forward guidance from an earlier 8-K vs. the reported quarter (Workday, ServiceNow), with the company's own "beat" framing removed |
| `airline_cross_entity` | 2 | Five US airlines, FY2023–24: open-ended vs. structured framing of identical data |
| `airline_comparability_break` | 2 | Alaska's revenue growth inflated by the Hawaiian acquisition, disclosed in a separate document |
| `airline_control` | 4 | Delta leads on every profitability and cost measure; catches reflexive hedging |
| `airline_tradeoff` | 2 | American vs. Southwest: profitability and balance sheet favor different airlines |

**Adversarial construction of the break items.**
- **Neutral phrasing.** Questions never mention restatement, spinoff or comparability.
- **Split evidence.** The note that resolves the break sits in a different document from the headline figures.
- **No editorializing.** Excerpts carry no explanatory commentary.
- **A genuine distractor.** The GE item includes a real but non-comparable figure.
- **A built-in trap.** In each break item, the originally reported prior-period figure implies a decline of −8% to −50%. The restated continuing-operations basis shows flat-to-positive growth.

**Anti-cheating check.**
- **Numeric items.** For each of the 12 numeric items, the expected value was searched for in its own corpus as a verbatim string. 11 have no match: the answer exists only as the result of a computation. The one match is the Amazon revenue lookup control (TECH-CTRL-03), where retrievability is the point.
- **Judgment items.** Their rubrics require a named conclusion plus a justification, so no string in the corpus can satisfy them.

## Method

**Models.**
- `claude-sonnet-5` and `claude-opus-5` (Anthropic API);
- `openai/gpt-5.1` (OpenRouter, `temperature=0`).

**Conditions.**
- **No-tool:** the filing excerpts are given inline in the prompt.
- **Tool:** the model gets `search_corpus`, scoped to the item's frozen documents, and `python_eval` as a calculator, with at most 10 tool calls.
- In both conditions, the system prompt requires `[DOC]` citations, an explicit statement when periods or entities are not comparable, and a final `ANSWER:` line.

**Two stages, each at 1 and 3 iterations.**
- **Stage 1, local harness.** 30 items × 3 models × 2 conditions × 3 iterations = 540 runs. Grading ran offline, after generation.
- **Stage 2, Gymnasium environment.** The same matrix was run as 540 new episodes. Every tool call went through `env.step`, at a cost of −0.01 per step, and every answer was rewarded 1 or 0 by the environment's verifier. The verifier reuses Stage 1's scoring code, so agreement between stages validates the reward as well as the models.

**Why 3 iterations.** Claude's API exposes no temperature control, and GPT-5.1 at `temperature=0` is not fully deterministic, so one run is one sample. Three iterations give confidence ranges and a consistency rate (the share of question-conditions answered correctly on every iteration). Together these separate stable habits from one-off errors.

**Grading.**
- **Numeric items (216 runs).** Graded against a tolerance band. An explicitly stated scale ("$768,044 thousand") is normalized before comparison.
- **Judgment items, iteration 1.** Grades were drafted against the pre-written rubrics, each with a verbatim supporting quote, and approved by the experimenter.
- **Judgment items, iterations 2–3.** Graded by a cross-family judge (`google/gemini-3.8-flash`). The judge first re-graded all 108 iteration-1 rows blind and agreed on **107/108 (99.1%)**. Its low-confidence verdicts and a random 5% sample (32 rows) were routed to a second review, which confirmed all of them.

**Evaluation infrastructure.** The pipeline behind both stages is documented in [`docs/architecture.md`](../docs/architecture.md):
- [how the evaluation was run](../docs/architecture.md#how-the-evaluation-was-run): the run lifecycle and the environment's reset/step/reward loop;
- [grading and verification](../docs/architecture.md#grading-and-verification): the calibration gate and review routing;
- [isolation and reward hacking](../docs/architecture.md#isolation-and-reward-hacking): the answer-key boundary, the `python_eval` audit, and verifier red-teaming;
- [scaling to production](../docs/architecture.md#scaling-to-production).

## Pre-registered predictions

**Primary predictions, locked 2026-09-21 before any model run.** They are condensed here without changing thresholds or meaning. At that time the pre-registered set had 16 items, including a 6-item break bucket (IBM, GE, 3M, Kellanova and a fiscal-year pair). That bucket was later hardened into the 4-item bucket above. The original items scored the same 100% under easier conditions.

1. **Control items** (single document, single period): ~80–95% with tools. This replicates FinSearchComp's T1 result and is a sanity check.
2. **Clean cross-period and cross-entity items:** ~55–70% with tools.
3. **Comparability-break items:** markedly worse, ~25–45% with tools, and the control-to-break gap does not close much when tools are added. This mirrors FinSearchComp's T3 result (+8.1 pp from tools, against +40.8 pp on T1).
4. **Dominant failure on break items:** a silent, unsupported conclusion, i.e. a clean-looking growth figure computed from the originally reported basis, with no flag that the restated basis tells a different story.

**Decision rule.**
- **Pursue** only if all three hold:
  - tool-equipped accuracy is below 65% on the pre-registered 16-item set;
  - at least 50% of break-bucket errors are period-basis, unsupported-conclusion or cross-document reconciliation failures, rather than retrieval or arithmetic failures;
  - the control-to-break gap is at least 20 pp with tools.
- **Abandon** if tool-equipped accuracy exceeds 80%, or if most errors trace to retrieval or tool mechanics rather than to judgment once the evidence is in hand.

The GAAP, guidance and airline buckets, and the Opus model, were added after the primary run. They are exploratory with respect to these predictions.

**Airline follow-up, locked 2026-09-23 before those items ran.** This tests whether the judgment collapse observed on AIR-XENT-01 generalizes. Each pair pairs an open-ended question with a structured twin on the same data. The controls stop a model that always answers "it depends" from scoring well.

| Pair | Type | Correct answer |
|---|---|---|
| AIR-BREAK-01/02 | Growth momentum, with the acquisition disclosed separately | United, on a like-for-like basis |
| AIR-CTRL-01/02 | Control: most profitable | Delta, on every measure |
| AIR-TRADE-01/02 | Genuine trade-off | American on operating margin; Southwest on balance sheet |
| AIR-CTRL-03/04 | Control: most efficient cost structure | Delta, on fuel and labor share of revenue |

Predictions:
1. Sonnet 5 and GPT-5.1 collapse the AIR-TRADE-01 trade-off in at least 3 of 4 runs; Opus surfaces it.
2. All models catch the acquisition without tools; at least one tool run misses it.
3. All models name Delta on the controls in at least 5 of 6 runs.
4. The structured twins score at least 90%.

Rule: collapse replicates if Sonnet 5 and GPT-5.1 are graded partial or incorrect on at least 50% of their AIR-TRADE-01 runs while scoring at least 90% on the structured twins.

## Results

### Accuracy by bucket

Each cell is the mean over the bucket's items × 3 iterations. The full matrices at both 1 and 3 iterations are shown as heatmaps in [`docs/architecture.md`](../docs/architecture.md#headline-results).

| Bucket | N | Sonnet 5 no-tool | Sonnet 5 + tool | Opus no-tool | Opus + tool | GPT-5.1 no-tool | GPT-5.1 + tool |
|---|---|---|---|---|---|---|---|
| `tech_control` | 4 | 75% | 100% | 92% | 100% | 50% | 50% |
| `tech_clean_comparison` | 6 | 100% | 100% | 100% | 100% | 100% | 100% |
| `multi_sector_comparability_break` | 4 | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| `tech_gaap_adjustment` | 4 | 100% | 100% | 100% | 100% | 100% | 100% |
| `tech_guidance_verification` | 2 | 100% | 100% | 100% | 100% | 100% | 100% |
| `airline_cross_entity` | 2 | 67% | 83% | 100% | 100% | 33% | 50% |
| `airline_comparability_break` | 2 | 100% | 100% | 100% | 100% | 50% | 67% |
| `airline_control` | 4 | 100% | 100% | 100% | 100% | 92% | 100% |
| `airline_tradeoff` | 2 | 100% | 100% | 100% | 100% | 83% | 100% |
| **Stage 1 total** | 30 | **85/90** | **89/90** | **89/90** | **90/90** | **75/90** | **79/90** |
| **Stage 2 total** | 30 | 85/90 | 89/90 | 89/90 | 90/90 | 77/90 | 79/90 |

### What the stages and iterations add

- **1 → 3 iterations.**
  - At 1 iteration, Opus looked perfect (60/60) and ahead of Sonnet 5 (58/60).
  - At 3 iterations, Opus slipped once (TECH-CTRL-01, no tools), and the two models' 95% ranges overlap (97–100% vs. 93–98%).
  - GPT-5.1's range (80–90%) lies entirely below both.
  - **Consistency** (correct on all 3 iterations): Opus 59/60 question-conditions, Sonnet 5 57/60, GPT-5.1 49/60.
- **Stage 1 → Stage 2.**
  - Independently generated episodes reproduce the Stage 1 totals to within two runs per model.
  - The same stable failures recur item by item: GPT-5.1 is 0/6 on both margin controls in each stage, and 0/6 (Stage 1) vs. 1/6 (Stage 2) on AIR-XENT-01.
  - The differences between stages are confined to the items already identified as unstable.
- **Efficiency signal.** In Stage 2's tool condition, Opus solves every episode but averages 4.2 tool steps, for a mean return of 0.958. Sonnet 5 averages 3.5 steps (0.954) and GPT-5.1 3.7 steps (0.841).

### Primary predictions vs. outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Control ~80–95% | Sonnet 5 and Opus 100% with tools; GPT-5.1 50% in both conditions | Partially wrong |
| 2 | Clean comparison ~55–70% | 100% for all models | Wrong, in the favorable direction |
| 3 | Breaks ~25–45%, gap survives tools | 100% for all models, both conditions | **Wrong** |
| 4 | Silent unsupported conclusion dominates | Not observed. Every break response identified the break and cited the disclosure | **Wrong** |

**Decision rule applied.** Tool-equipped accuracy on the break items is 100%, far above the 80% abandon threshold, for all three models. None of the three "pursue" conditions holds. **Abandon.**

### Airline follow-up vs. outcome (all 3 iterations)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Sonnet 5 and GPT-5.1 collapse AIR-TRADE-01 | Trade-off surfaced on 11/12 runs; Opus 6/6 | Wrong |
| 2 | All catch the acquisition without tools; ≥ 1 tool run misses it | 9/9 without tools, 7/9 with tools. The failed tool runs *had* retrieved the note | Outcome right, mechanism wrong |
| 3 | Controls: Delta in ≥ 5 of 6 runs per model | 36/36; no over-hedging | Right |
| 4 | Structured twins ≥ 90% | 68/72 (94%) | Right |

Collapse does not replicate on the new trade-off item. It remains stable for GPT-5.1 on the original item (AIR-XENT-01). Opus's advantage reflects judgment rather than hedging: it surfaces the trade-off and commits to Delta on every control run.

## Failure analysis

Stage 1 has **33 failed runs out of 540: 26 GPT-5.1, 6 Sonnet 5, 1 Opus.** No run showed the predicted failure: there was no wrong-period basis, no entity conflation, and no silent use of a non-comparable figure. Counts below are Stage 1 / Stage 2 across 3 iterations and both conditions.

| Failure mode | Items | Stage 1 / Stage 2 correct | Stable? |
|---|---|---|---|
| Margin vs. profit | TECH-CTRL-01/02 | GPT-5.1 0/12 · 0/12. Sonnet 5 3/6 · 3/6 (TECH-CTRL-01, all no-tool). Opus 5/6 · 5/6 | Stable for GPT-5.1 |
| Judgment collapse | AIR-XENT-01 | GPT-5.1 0/6 · 1/6. Sonnet 5 3/6 · 4/6. Opus 6/6 · 6/6 | Stable for GPT-5.1 |
| Acquisition flagged, not adjusted | AIR-BREAK-01/02 | GPT-5.1 7/12 · 8/12. Sonnet 5 and Opus 24/24 | GPT-5.1 only |
| Occasional slips | AIR-CTRL-04, AIR-TRADE-01, AIR-XENT-02 | GPT-5.1, about one run each | No |

**Margin vs. profit.** Asked for "gross margin", the model reports the dollar gross profit ("$171,008 million") instead of the percentage (69.76%). In the tool condition the calculator reinforces the wrong stopping point. The subtraction is executed, and the division never is:

```text
→ search_corpus("Microsoft Form 10-K 2024 gross margin fiscal year 2024")
← [msft_10k_fy2024.txt] Fiscal year ended June 30, 2024: Revenue $245,122; Cost of revenue $74,114.
→ python_eval("print(245122-74114)")
← 171008
ANSWER: $171,008 million (gross margin for fiscal year 2024)
```

This is partly a **wording confound**: Apple and Microsoft label the dollar line "Gross margin" in their own income statements, so the question should ask for the percentage explicitly.

**Judgment collapse.** The item asks: *"Based on their fiscal year 2024 results, which of these five airlines was the strongest performer?"* GPT-5.1 extracts every figure and computes every margin correctly. It then names a single winner without noting that another airline led on growth:

```text
- Delta:  Op. margin ≈ 5,995 / 61,643 ≈ 9.7%
- Alaska: Revenue FY2023 $10,426M, FY2024 $11,735M ... Op. margin ≈ 4.9%
On both scale (highest operating income) and efficiency (highest operating margin), Delta edges out ...
ANSWER: Delta
```

Opus reaches the same winner but states the criterion and the trade-off: *"if the criterion is rate of improvement, United (large-cap) or Alaska (percentage terms, subject to the organic-growth caveat) would lead."* The failure is therefore not choosing Delta, but collapsing a multi-attribute decision without stating the basis for it. The structured twin (AIR-XENT-02) scored 17/18, so the arithmetic is not the problem.

**Acquisition flagged, not adjusted.** In every AIR-BREAK run, GPT-5.1 found and flagged the Hawaiian acquisition. In the failing runs it still named Alaska as the growth leader, or ranked by absolute dollar growth. Detection is solved; the follow-through after detection is not. This is the only residual failure that touches the target mechanism.

**Scorer defect found and fixed.** Datadog reports in thousands. GPT-5.1's "$768,044 thousand" is correct and states its unit, but an earlier version of the numeric verifier ignored stated scale and marked it wrong. The verifier now normalizes explicit scale, with a regression test. The case shows how a naive verifier can manufacture apparent headroom.

**Grounding caveat (not scored).** On AIR-XENT-01/02, whose corpus does not mention the acquisition, Sonnet 5 and Opus raised it from parametric knowledge. Opus's tool runs attributed the acquisition, and on AIR-CTRL-04 a refinery fuel detail, to source files that do not contain them. The facts are true, but the citations are fabricated. A training verifier would need to check grounding as well as the conclusion.

## Decision and recommendation

**Do not pursue accounting-comparability-break detection as a training target.** The negative result is robust:
- 100% on multi-sector breaks for every model, condition and iteration;
- reproduced across 1 and 3 iterations and across two independent stages (1,080 answers in total);
- extended to a new acquisition case where detection was universal.

A direction that looked promising in the 2025 literature appears to have closed for late-2026 frontier models, at least for well-documented events.

The residual findings are weaker training targets because they are model-specific: a third model already avoids all of them.

| Finding | Likely cause | Cheapest intervention | Training-target confidence |
|---|---|---|---|
| Margin vs. profit | No unit or ratio check before a %-typed answer | Reword the items; add a unit check to the prompt. If GPT-5.1's error disappears, it was never a training target | Low |
| Judgment collapse | No incentive to state the comparison criteria before concluding | Process reward for enumerating dimensions, with a dominance-control penalty so the policy cannot learn "always hedge" | Low; one item, did not generalize |
| Acquisition not adjusted | Detection not carried through into the computation | Supervised fine-tuning on detect-then-adjust traces | Low; one model |

**What would change the conclusion.** The mechanism should be tested on obscure filers' little-publicized reclassifications. If accuracy drops there, the current result reflects recognition of famous events rather than general comparability reasoning. The [scaling section](../docs/architecture.md#scaling-to-production) of the architecture doc describes an ingestion pipeline built for exactly this test.

**Recommended follow-up.** A properly powered judgment-collapse study:
- about 40 matched triples across sectors, each with an open-ended question, a structured twin and a dominance control;
- a pre-registered threshold of at least 30% collapse for at least two models, with at most 10% hedging on the controls.

The environment, verifier and isolation design are ready for it. Before any training use, the verifier needs strict single-value numeric matching and a separate sandbox for `python_eval` ([Isolation and reward hacking](../docs/architecture.md#isolation-and-reward-hacking)).

## Limitations

- **Memorization confound.** This is the most important caveat on the null result. Every break item uses a famous company (IBM, GE, 3M, Alaska/Hawaiian). Models cite the provided text accurately, but recognizing a well-known spinoff may prime correct handling in a way that would not transfer to an obscure company.
- **Small N per bucket.** Buckets have 2–6 items. Iterations separate stable habits from noise but do not add questions. Per-bucket percentages, especially in the two-item airline buckets, are directional.
- **Sampling control.** GPT-5.1 runs at `temperature=0`. Claude's API exposes no temperature parameter. The 3 iterations and the second stage quantify the variation but do not remove it.
- **LLM-produced grades.** Iteration-1 grades were drafted by Claude and approved by the experimenter. The second review of the 32 flagged rows was also done by Claude, at the experimenter's direction; those rows are tagged `reviewed:claude`. The cross-family agreement of 99.1% makes a Claude-specific bias unlikely. It shows that two model families agree, not that either matches a human expert, so a blinded human sample would be the stronger check.
- **Borderline calls.**
  - AIR-XENT-01 turns on whether the growth leader is surfaced.
  - 4 AIR-TRADE-01 runs acknowledged American's lead only in dollars; they were graded correct with low confidence.
  - TECH-CMP-02 accepted "decelerated" for a 0.84 pp drop.
  - Any of these calls could move a small cell.
- **Incomplete evidence on AIR-XENT-01/02.** Their corpus omits the Hawaiian acquisition. Excluding its $869M, Alaska grew 4.22% rather than 12.56%, and United (6.23%) leads on growth. The items' conclusions still hold, because no single airline leads on both growth and margin. The acquisition itself is tested in AIR-BREAK-01/02, whose corpora include the note.
- **Guidance bucket contains only beats.** Both sourced quarters beat guidance; no miss case was tested.
- **Narrow tool scope.** `search_corpus` is limited to each item's frozen documents, not open EDGAR or web search. This measures reasoning rather than harness quality, but makes the tool/no-tool contrast narrower than in open-web agentic benchmarks.

## References

1. Bigeard, A., Krishnan, R., Nashold, L., Wu, S. "Finance Agent Benchmark." [arXiv:2508.00828](https://arxiv.org/abs/2508.00828)
2. ByteDance Seed + Columbia Business School. "FinSearchComp." [arXiv:2509.13160](https://arxiv.org/abs/2509.13160)
3. Jiang, Y. et al. (Yale, UCSD, Tongji, Goldman Sachs AI Research). "Fin-RATE." [arXiv:2602.07294](https://arxiv.org/abs/2602.07294)
4. SEC EDGAR full-text filings and XBRL company-concept API (data.sec.gov): the primary source for all dataset figures. Accession numbers are recorded per item in [`data/answer_keys.jsonl`](../data/answer_keys.jsonl).
