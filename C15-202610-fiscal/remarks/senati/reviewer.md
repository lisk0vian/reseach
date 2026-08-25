# C21-2026 Reviewer Corrections

**Verification targets — both manuscripts carry the same evidence:**

| Manuscript | Source | Compiled PDF |
|---|---|---|
| A — Springer submission | `papers/drafts/springer/main.tex` | `papers/renders/C15-202610[fiscal].springer.en.pdf` (27 pp.) |
| B — CTVEMF draft | `papers/drafts/latex/main.tex` | `papers/renders/C15-202610[fiscal].latex.en.pdf` (27 pp.) |

**Verified on:** 2026-08-24 · **Status:** 15 verified · 5 partial · 6 pending

Legend — `[x]` verified against both manuscripts · `[~]` partially addressed · `[ ]` pending.
Every verified item carries the section, table or figure where the evidence sits.

Section and table numbers below refer to manuscript A. Manuscript B uses the same
table numbers (2, 4, 7, 8, 9, 10, 11) and the same subsection numbers for the
availability audit (3.2), calibration (4.5), risk stratification (4.6), decision-analytic
utility (4.7) and limitations (6). Two pointers differ: the partition protocol is §3.5.1
in B (§3.6 in A), and consensus feature selection is §3.3.6 in B (§3.5 in A).

---

## Prediction time point and data leakage

- [x] **Define** the exact clinical prediction time point — **specifying what information is available before, during, and after delivery** — **to clearly establish the real-world early prediction scenario**.
  > **Verified.** §3.2 ¶1: the prediction is issued at the start of year *Y*, before any case flow of *Y* is observed. Fig. 2 shows the timeline. Domain-transposed: "before / during / after delivery" → "before / during / after the reporting year closes".

- [~] **Exclude** variables known only during or after delivery — **reviewing the temporal availability of each predictor individually** — **to eliminate potential data leakage**.
  > **Partial.** Twelve variables removed before any fit (§3.2 ¶3): 5 concurrent volumetric, 5 label variants, 1 constant provenance field, 1 collinear year index. **Residual:** three ex-post one-hot indicators (file-provenance ID, download date, cut-off date) survived the consensus selection into the reported 74-feature model. Documented in §3.2 ¶4, Table 2 (last row) and §6 ¶3. They are identically zero across 2024–2026, so they cannot inflate any reported figure, but re-fitting on the ex ante subset remains outstanding.

- [x] **Verify** the clinical availability of every predictor variable — **classifying variables according to whether they are available before delivery, during delivery, or after delivery** — **to ensure that the model uses only information available at the prediction time point**.
  > **Verified.** Table 2 classifies every predictor block as *ex ante* / *concurrent* / *ex post*, with the disposition and rationale for each.

---

## Table consistency and data partitions

- [x] **Correct** the inconsistency identified in Table I — **cross-checking its values against the actual data and procedures used** — **to ensure consistency between the table and the methodology**.
  > **Verified.** Every number now derives from a single source (`src/tables/*.xlsx`); the 10 figures are generated from those same tables by `make_figures.py`. The reported configuration is **XGBoost, τ = 0.65, F1 = 0.409** consistently across Tables 6, 8, 9, 10 and 11. The previous draft mixed LightGBM / F1 0.469 / τ 0.66 with XGBoost / 0.409 / 0.65, and its reliability figure contradicted Table 49 of the pipeline.

- [x] **Detail** the data partitions used for model development — **specifying the datasets assigned to training, validation, and testing** — **to make the experimental procedure reproducible**.
  > **Verified.** §3.6 (five numbered roles), Fig. 2 and Table 4: training 2019–2023 (n = 6,076), threshold selection 2024 (n = 1,195), locked test 2025 (n = 1,199), exploratory external 2026 (n = 1,123). Feature selection confined to 2019–2022 with an internal check on 2023.

- [x] **Specify** the dataset used for classification-threshold selection — **documenting when, how, and with which data the cutoff was determined** — **to prevent information from the test set from influencing threshold selection**.
  > **Verified.** §3.6 item 3 and Table 7: all four threshold criteria computed on the 2024 partition only. §4.4 states the 2024 metrics are optimistically biased for that reason and must not be read as performance.

- [ ] **Specify** the dataset used for recalibration — **documenting which observations were used to calibrate the predicted probabilities** — **to prevent contamination of the final evaluation set**.
  > **Pending — no recalibration was performed.** §4.5 ¶3 designates the 2024 partition as the set to use and flags the caveat that 2024 already serves as the threshold-selection set. §6 ¶2 lists it as a deployment prerequisite. The manuscript does not claim recalibrated performance.

- [x] **Define** the dataset reserved for final evaluation — **keeping it independent from training, threshold selection, and recalibration** — **to obtain an unbiased estimate of model performance**.
  > **Verified.** §3.6 item 4: 2025 consulted once, after model and threshold were frozen. Independence from training (2019–2023) and threshold selection (2024) is explicit in §3.6 ¶2 and the Table 8 footnote.

---

## Decision curve analysis (reviewer Figure 7)

- [x] **Review** Figure 7 — **examining the model's net benefit against the treat-all and treat-none strategies** — **to correctly determine the observed clinical utility**.
  > **Verified.** §4.7, Table 11 and Fig. 8 compute net benefit (Eq. 2) across p_t and compare against flag-all and flag-none.

- [x] **Correct** the interpretation of Figure 7 — **considering that the reported net benefit is negative relative to the treat-none strategy** — **to avoid claiming clinical utility that the figure does not demonstrate**.
  > **Verified.** §4.7 ¶2 states plainly: at τ = 0.65 the net benefit is **−0.035**, below flag-none. Table 11 marks that row "Flag none" as the preferred strategy.

- [x] **Moderate** claims regarding clinical utility — **conditioning such claims on evidence of favorable net benefit** — **to avoid overinterpreting the decision-curve results**.
  > **Verified.** Utility is claimed only for p_t ∈ [0.1, 0.5], where net benefit is positive. §4.7 ¶3, §5.2, §5.4 and the Conclusions all state that the framework is a ranking/triage instrument, not an alerting system, and that no utility is claimed at the F1-optimal operating point.

---

## Calibration (reviewer Figure 9)

- [x] **Review** Figure 9 — **comparing predicted probabilities with observed probabilities** — **to correctly determine the direction and magnitude of calibration deviation**.
  > **Verified.** Table 9 gives all ten reliability bins with the **signed** deviation (predicted − observed) and a direction column; Fig. 6 (right panel) plots the signed deviation.

- [x] **Correct** the interpretation of Figure 9 — **replacing the interpretation of underestimation with overestimation when confirmed by the results** — **to ensure that the text accurately reflects the evidence shown**.
  > **Verified.** §4.5 ¶2: over-estimation in every band above 0.3 (the 0.9–1.0 band predicts 0.949 against 0.527 observed; MCE 0.461 in the 0.7–0.8 band). Under-estimation is confined to the two lowest bands and is reported as such. §5.2 repeats the correction.

- [ ] **Verify** the effect of recalibration — **comparing calibration before and after recalibration** — **to demonstrate whether recalibration actually improves agreement between predicted and observed risk**.
  > **Pending.** No before/after comparison exists because no calibration map was fitted. Declared in §4.5 ¶3 and §6 ¶2.

---

## Risk stratification

- [ ] **Repeat** the risk stratification analysis — **using probabilities after recalibration** — **to construct risk categories from calibrated probabilities**.
  > **Pending.** Table 10 and Fig. 7 are built on the **uncalibrated** scale. Flagged in the Table 10 footnote and §4.6 ¶3 ("the band labels describe rank position, not risk magnitude").

- [~] **Review** the risk-category thresholds — **applying them to calibrated probabilities and documenting their justification** — **to ensure that the proposed categories are methodologically supported**.
  > **Partial.** The boundaries are documented and justified (deciles of predicted probability aggregated into four bands, Table 10 footnote), and the monotone gradient 7.2% → 22.9% → 40.7% → 52.1% against a 14.35% base rate is reported. Not yet applied to calibrated probabilities.

- [ ] **Independently validate** the proposed risk categories — **using data separate from those used to establish the category thresholds** — **to determine whether the categories maintain their behavior outside the development sample**.
  > **Pending.** §4.6 ¶3 states explicitly that the strata are derived from and described on the same 2025 partition, and that until an independent validation exists they must be read as a description of ranking behaviour rather than a validated triage instrument.

- [ ] **Evaluate** the consistency of the risk categories — **comparing predicted and observed risk in the independent validation dataset** — **to determine whether the categories maintain appropriate performance**.
  > **Pending.** Depends on the item above. The predicted-versus-observed comparison per band exists for 2025 only (Table 10).

---

## Originality, citations and academic quality

- [~] **Review** the sections identified in the similarity report — **locating matches and distinguishing citations, technical terminology, and overly similar paraphrasing** — **to correct academically problematic similarities**.
  > **Partial.** The manuscript was written from scratch for this submission; no text was carried over from the previous draft. Cannot be closed without running the similarity report against the new file.

- [x] **Correct** citations and references — **properly attributing ideas, data, and claims derived from external sources** — **to ensure appropriate academic attribution**.
  > **Verified.** 36 references, all cited in text, Vancouver numbered style via `sn-vancouver.bst`, DOIs included. Build reports no undefined citations. 20 methodological references were added for the new content (decision curve analysis, calibration, TRIPOD+AI, DeLong, McNemar, Friedman/Nemenyi, conformal prediction, the gradient boosting libraries).

- [~] **Rephrase** high-similarity passages — **expressing the underlying ideas in original wording without changing their scientific meaning** — **to reduce inappropriate textual overlap while preserving academic accuracy**.
  > **Partial.** Full rewrite, including the Related Work section. Pending confirmation against a similarity report on the new text.

- [~] **Review** sections flagged by the possible AI-generated-content indicator — **improving wording, precision, coherence, and source traceability** — **to strengthen the academic quality and authorship of the manuscript**.
  > **Partial.** Prose rewritten throughout; every quantitative claim is traceable to a named table or figure, and the negative findings (negative net benefit, over-estimation, territorial disparity, the historical block not earning its place) are argued rather than asserted. Pending re-run of the detector.

---

## Journal selection and adaptation

- [ ] **Evaluate** the proposed target journals — **comparing quartile, scope, indexing, thematic fit, editorial requirements, publication timelines, and APCs** — **to identify the most appropriate journal for the manuscript**.
  > **Pending.** No comparative evaluation of candidate journals is documented in the repository. Only the requirements of the selected journal are recorded (`papers/templates/JOURNAL_OF_BIG_DATA_REQUIREMENTS.md`).

- [x] **Select** the target journal — **prioritizing alignment with Computer Science, Computer Engineering, Data Science, Artificial Intelligence, Software, Health Informatics, Predictive Analytics in Healthcare, or Maternal and Perinatal Health** — **to appropriately guide the final manuscript version**.
  > **Verified.** Journal of Big Data (SpringerOpen, ISSN 2196-1115). Fits Computer Science / Data Science / Predictive Analytics; no strict length limit, which suits a methods-heavy paper with extensive validation.

- [x] **Review** the journal template and author guidelines — **analyzing structure, length, table and figure formatting, references, and submission requirements** — **to prepare the manuscript according to the selected journal's requirements**.
  > **Verified.** Requirements recorded in `papers/templates/JOURNAL_OF_BIG_DATA_REQUIREMENTS.md` and satisfied: abstract 248 words (limit 250, no citations), 7 keywords (range 3–10), Vancouver references, all 11 tables inside the main file as LaTeX tabulars, 10 figures at 300 dpi as separate files with captions in the manuscript, and the mandatory Competing Interests / Funding / Data Availability declarations.

- [x] **Adapt** the manuscript to the selected journal — **incorporating all methodological and editorial corrections requested by the reviewer** — **to produce a submission-ready scientific manuscript**.
  > **Verified.** Built with `sn-jnl.cls` + `sn-vancouver.bst`, 27 pp., clean compile (no errors, no undefined references or citations). All 10 figures and 11 tables are cross-referenced in the text.

---

## Outstanding work before submission

The six pending items reduce to two pieces of computation, both of which require re-executing part of the pipeline:

1. **Recalibration and its downstream effects** (items 7, 14, 15, 17, 18). Fit a calibration map on a partition reserved for it, report calibration before and after, re-derive the risk bands on the calibrated scale, and validate them on data not used to set the boundaries. Splitting 2024 into two disjoint halves — one for calibration, one for threshold selection — would keep the roles clean and leave 2025 untouched as the locked evaluation.
2. **Re-fitting on the ex ante subset** (item 2). Re-run the consensus selection and the model over the 70 admissible features, removing the three publication-metadata indicators. This closes the residual leakage exposure and would confirm whether it affects the ranking on which the practical recommendation rests.

Item 23 (comparative journal evaluation) is a documentation task, not a computational one.

Items 19, 21 and 22 need the similarity and AI-detection reports re-run against the new manuscript before they can be closed.
