# Cover letter — Journal of Big Data

> **Before sending:** fill the bracketed fields, confirm the AI-use disclosure
> matches what is stated in the manuscript, and insert the repository DOI once
> the reproducibility package is archived.

---

To the Editors of *Journal of Big Data*

Dear Editors,

We submit for your consideration the manuscript **"Prosecutorial congestion in
Peru, 2019–2026: a leakage-audited, calibration-aware framework for predicting
proxy signals from open administrative records"**, for the **Methodology**
article type.

**Why this belongs in *Journal of Big Data*.** The contribution we claim is a
protocol for open administrative registries, not a result about one registry.
The paper documents a leakage mode that the existing leakage literature does not
cover: fields that exist only because a registry is *published in periodic
batches* — reporting-period descriptors, download and cut-off dates,
file-provenance identifiers — are perfectly legitimate descriptors of a closed
record, survive a temporal validation split without any warning sign, and then
act as period fixed effects inside the model. A model built on them validates
cleanly and cannot be run at the moment its prediction is needed. This failure
mode becomes *more* likely, not less, as the number of integrated source files
and derived candidate features grows, which is precisely the regime your
readership works in. We provide the audit that detects it as a reusable
template (Additional file 2), including the verification step — audit the fitted
feature matrix, not the exclusion rule — that our own pipeline needed and did
not initially have.

**A second reason we hope will interest you.** This is a prediction study that
reaches a negative conclusion about its own model and publishes it. The
framework attains a ROC-AUC of 0.796 on a single locked evaluation year, which
in isolation would support a confident claim. Calibration analysis shows
systematic over-estimation above a predicted probability of 0.3, and decision
curve analysis shows that at the model's own F1-optimal operating point the net
benefit is negative — a decision maker holding that threshold would do better
intervening nowhere. We therefore report the framework as a ranking and triage
instrument for moderate intervention costs and explicitly decline to claim
operational utility as an alerting system. Studies that report the threshold at
which their model stops being useful are rare, and we think the methodological
point — that discrimination alone would have supported a materially stronger
claim than the evidence warrants — is worth making in a venue with your reach.

**Declarations.** All authors have approved the manuscript and agree to its
submission. The work has not been published previously and is not under
consideration for publication elsewhere, in whole or in part. The authors
declare no competing interests, financial or non-financial. No funding was
received for this research. The study uses publicly available aggregate
administrative records released under an open data licence and involves no human
subjects, so ethics approval and consent to participate are not applicable. The
data set is publicly available from the Peruvian National Open Data Platform,
and the complete analysis pipeline together with the derived result tables is
archived on Zenodo at https://doi.org/10.5281/zenodo.22118625.

**Suggested reviewers.** [Name, institution, institutional email, ORCID or
Scopus ID — three to five suggestions with expertise in data leakage in applied
machine learning, prediction model calibration and decision curve analysis, or
computational analysis of justice systems.]

We look forward to your assessment.

Yours sincerely,

Moisés Evangelista Gamarra (corresponding author)
Jerremi Aron Chancan Labajos
Servicio Nacional de Adiestramiento en Trabajo Industrial (SENATI), Lima, Peru
mevangelistag@senati.pe
