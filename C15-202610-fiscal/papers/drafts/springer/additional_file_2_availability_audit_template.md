# Additional file 2 — Temporal availability audit template

Companion to: *Prosecutorial congestion in Peru, 2019–2026: a leakage-audited,
calibration-aware framework for predicting proxy signals from open
administrative records.*

This template generalises the audit reported in Section "The prediction time
point and the temporal availability audit". It applies to any longitudinal
administrative registry published in periodic batches (annual, quarterly,
monthly), where each batch is compiled and released after the period it
describes has closed.

---

## Step 1 — Fix the prediction time point

State, in one sentence, the instant at which the prediction must be issued.
Everything else follows from this. Example: *"for reporting period `Y`, the
prediction is issued at the start of `Y`, before any of the activity of `Y` has
been observed."*

Record it here:

> Prediction time point: ______________________________________________

## Step 2 — Assign every candidate variable to one availability class

| Class | Definition | Admissible as a predictor? |
|---|---|---|
| **Ex ante** | Observable strictly before the prediction time point. Structural attributes of the unit; encodings fitted on earlier periods only; calendar indicators fixed in advance; lagged aggregates computed from strictly earlier periods. | Yes |
| **Concurrent** | Accumulates during the period being predicted. Volumes, counts, rates and anything derived from them. | No |
| **Ex post** | Comes into existence only when the batch is compiled and released: reporting-period descriptors, download and cut-off dates, file-provenance and file-origin identifiers, batch version tags. | No |

Two traps worth naming explicitly:

- A variable can be **ex ante in principle but inadmissible for another reason**
  (collinearity with the period index, extrapolation outside the training range).
  Record the reason separately; do not fold it into the availability class, or
  the counts you report will not mean what readers take them to mean.
- The **outcome label is usually built from concurrent variables.** Those
  variables are then doubly inadmissible, and any variant of the label computed
  under alternative thresholds is inadmissible for the same reason.

## Step 3 — Fill the audit table

| Variable block | Class | Disposition | Rationale |
|---|---|---|---|
| | | Retained / Excluded | |
| | | Retained / Excluded | |
| | | Retained / Excluded | |
| | | Retained / Excluded | |
| | | Retained / Excluded | |

## Step 4 — Verify against the fitted feature matrix, not against the rule

**This is the step the present study found to be necessary, and it is the one
most easily skipped.** Declaring a block inadmissible in the protocol does not
remove it from the data. In the study reported here, the ex post block was
declared inadmissible, but only one of its fields was actually dropped before
the selection stage; three one-hot indicators derived from the remaining fields
survived the consensus selection into the final model, where they functioned as
period fixed effects.

Run these four checks against the **final list of features that the reported
model was fitted on** — not against the exclusion list, and not against the
candidate pool:

1. **Name match.** For every excluded source field, grep the final feature names
   for the field name and for every encoding prefix derived from it
   (`onehot_`, `freq_`, `te_`, interaction names). Expected result: zero hits.
2. **Degenerate-support check.** For every retained feature, compute the
   proportion of non-zero values per period. A feature that is non-zero in
   exactly one period is a period fixed effect regardless of what it is called.
3. **Period-predictability check.** Fit a cheap multiclass classifier from the
   final feature matrix to the period index. Near-perfect accuracy means some
   feature encodes the period; inspect the top importances.
4. **Evaluation-period nullity.** For any feature that fails check 2 or 3,
   verify whether it is identically zero across every evaluation period. If it
   is, the reported evaluation figures are unaffected, but training-time
   absorption of period-specific variation still occurred and must be reported.

Record the outcome:

| Check | Result | Features flagged | Effect on reported figures |
|---|---|---|---|
| 1. Name match | | | |
| 2. Degenerate support | | | |
| 3. Period predictability | | | |
| 4. Evaluation-period nullity | | | |

## Step 5 — Report the counts precisely

State separately: the number of variables excluded **for unavailability**, the
number excluded **for other reasons**, and the number of features that **failed
the verification of Step 4**. Reporting a single combined figure obscures which
of the three the reader is looking at.
