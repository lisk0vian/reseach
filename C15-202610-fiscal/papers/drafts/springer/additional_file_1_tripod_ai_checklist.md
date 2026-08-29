# Additional file 1 — TRIPOD+AI reporting checklist

Companion to: *Prosecutorial congestion in Peru, 2019–2026: a leakage-audited,
calibration-aware framework for predicting proxy signals from open
administrative records.*

> **Before submission — action required.** The item numbering and topic
> areas below follow the TRIPOD+AI statement (Collins et al., *BMJ* 2024;385:e078378).
> The official checklist form must be downloaded from the TRIPOD website and the
> "Reported in" column below transferred onto it verbatim; the item wording is
> **not** reproduced here, only the topic area and the location in this manuscript.
> Where an item is marked *Not met*, either address it or carry the entry into the
> Limitations section — do not silently mark it as met.

| # | Topic area | Reported in | Status |
|---|---|---|---|
| 1 | Title identifies the study as developing a prediction model, names the target population and the outcome | Title | Met |
| 2 | Abstract: structured summary of objectives, data, methods, results, limitations | Abstract | Met |
| 3 | Background and rationale; existing models | Introduction; Related work | Met |
| 4 | Objectives, including the intended use and users of the model | Introduction; Discussion → Practical implications | Met |
| 5 | Data source, study design, setting, eligibility of records | Materials and methods → Data source and unit of analysis | Met |
| 6 | Dates of data collection, follow-up, and the temporal split | Materials and methods → Partition protocol; Table 6 | Met |
| 7 | Outcome definition, how and when it was measured, and blinding of the outcome to predictor information | Materials and methods → Proxy label construction and sensitivity | **Partially met.** The outcome is a quantile-rule proxy, not a certified outcome; no expert elicitation was carried out. Stated in Limitations. |
| 8 | Predictors: definition, timing of measurement relative to the prediction time point | Materials and methods → The prediction time point and the temporal availability audit; Table 4 | **Partially met.** Three publication-metadata indicators survived the selection into the reported model; the exposure is reported and analysed in the same section and in Limitations. |
| 9 | Sample size and number of events; events per candidate predictor | Table 2; Table 6; Materials and methods → Models, hyperparameter search and evaluation (EPV ≈ 11.5) | Met |
| 10 | Missing data: how much, and how handled | Materials and methods → Data source and unit of analysis; Proxy label construction (26 imputed label inputs) | Met |
| 11 | Data preparation: encoding, scaling, and the partition on which each was fitted | Materials and methods → Feature engineering | Met |
| 12 | Predictor selection: method and the data on which it was performed | Materials and methods → Consensus feature selection | Met |
| 13 | Handling of class imbalance and the rationale | Related work → Leakage, temporal validation and class imbalance; Materials and methods → Models | Met |
| 14 | Model type(s) and the rationale for the choice | Materials and methods → Models; Results → Model comparison | Met |
| 15 | Hyperparameter tuning: search strategy and the data used | Materials and methods → Models | Met |
| 16 | Model output: what the model predicts and on what scale | Results → Calibration; Risk stratification | Met |
| 17 | Evaluation: discrimination, calibration and clinical/decision utility, each reported separately | Results → Model comparison; Calibration; Decision-analytic utility | Met |
| 18 | Uncertainty quantification | Table 8; Results → Subgroup behaviour and uncertainty quantification | **Partially met.** Bootstrap intervals rest on 300 resamples; noted in Methods and Limitations. |
| 19 | Fairness: subgroup performance and any disparity | Results → Subgroup behaviour and uncertainty quantification | Met |
| 20 | Model updating / recalibration | Results → Calibration; Discussion → Practical implications | **Not met.** Calibration was measured but not corrected. Stated explicitly in Results, Discussion and Limitations; no recalibrated performance is claimed. |
| 21 | Open science: funding, conflicts, protocol, registration, data and code availability | Declarations | **Action required.** Insert the repository name and persistent identifier in *Availability of data and materials* before submission. |
| 22 | Patient and public involvement | Not applicable | Not applicable — no human participants; aggregate administrative records only |
| 23 | Flow of records through the study; characteristics of each partition | Table 2; Table 6 | Met |
| 24 | Model specification sufficient for others to make predictions | Materials and methods; Additional file 2; repository | **Partially met.** Full specification is in the accompanying repository; the manuscript reports the pipeline but not the fitted parameters. |
| 25 | Model performance, with uncertainty, in each partition | Table 8; Table 9; Table 11; Table 12 | Met |
| 26 | Interpretation of results in the context of objectives, and comparison with other evidence | Discussion → What the evidence supports; What the evidence does not support; Interpreting the gap to the single-indicator rules | Met |
| 27 | Limitations, and implications for practice | Limitations; Discussion → Practical implications | Met |

## Deliberate departures from the guideline

Where TRIPOD+AI invites a single summary of predictive performance, this study
reports discrimination, calibration and decision-analytic utility separately and
declines to reconcile them into one figure of merit. The reason is given in the
Introduction and borne out in Results: the three assessments disagree here, and
the most conservative of them governs the conclusion.
