# Early Prediction of Low Birth Weight: Characterizing Predictive Performance in a National Peruvian Cohort (2015–2025)

**Area**: Maternal and perinatal health · **Status**: `Revision` · **Journal**: Pending decision

**Description:**
This project develops a reproducible machine-learning pipeline for the early prediction of low birth weight using a national Peruvian live-birth cohort from 2015–2025. The revision addresses the clinical prediction time point, information leakage, data partitioning, calibration, risk stratification, decision-curve interpretation, and independent validation of proposed risk categories.

The complete analysis is designed to run in Google Colab with data stored in Google Drive. The source dataset is not included in this repository because of its size and data-governance requirements; download it from the official source listed below.

This research studies early prediction of low birth weight in a national Peruvian cohort from 2015–2025. It presents a reproducible machine learning framework that combines temporal validation, calibration, interpretability, and equity auditing to assess predictive performance before clinical deployment.

## Research Team

### Authors

* **Moisés Evangelista Gamarra** — Principal Researcher
  <br />
  [ORCID](https://orcid.org/0009-0002-5382-1390) · [Google Scholar](https://scholar.google.com/citations?user=urGKMNQAAAAJ&hl=es)

* **Jerremi Aron Chancan Labajos** — Co-author
  <br />
   [ORCID](https://orcid.org/) · [Google Scholar](https://scholar.google.com/citations?hl=es&user=wV2fQskAAAAJ) · [LinkedIn](https://www.linkedin.com/in/jeremi-aron/) · [GitHub](https://github.com/lisk0vian)

* **Arnold Albert Huaman Zamora** — Co-author
  <br />
  [ORCID](https://orcid.org/0009-0006-8186-9251)

**Affiliation**: [Servicio Nacional de Adiestramiento en Trabajo Industrial, Lima, Perú](https://www.senati.edu.pe/especialidades/tecnologias-de-la-informacion/ingenieria-de-software-con-inteligencia-artificial)

## Dates

| Milestone            | Date        |
| -------------------- | ----------- |
| Research Start       | 28-jul-2026 |
| Experiments          | 29-jul-2026 |
| Manuscript Draft     | 29-jul-2026 |
| Internal Review      | 21-aug-2026 |
| Expected Publication | DD-MMM-YYYY |

## Links

**Dataset 1:** [Registros de Nacidos Vivos en el Perú (2015–2025)](https://www.datosabiertos.gob.pe/dataset/registros-de-nacidos-vivos-en-el-per%C3%BA-2015%E2%80%932025)

**Dataset 2:** [Registros de Nacidos Vivos en el Perú 2026](https://www.datosabiertos.gob.pe/dataset/registros-de-nacidos-vivos-en-el-per%C3%BA-2026)

## Contents

* [`notebooks/early_prediction_colab.ipynb`](notebooks/early_prediction_colab.ipynb) — complete Colab pipeline and results viewer.
* [`correction_objectives.md`](correction_objectives.md) — objectives for the methodological correction.
* [`references.bib`](references.bib) — BibTeX references for the data sources.
* [`C03-202610-Nacimiento_CORREGIDO.docx`](C03-202610-Nacimiento_CORREGIDO.docx) — corrected manuscript in Word format.
* [`C03-202610-Nacimiento_CORREGIDO.pdf`](C03-202610-Nacimiento_CORREGIDO.pdf) — PDF export for review and distribution.

Generated PDFs, figures, tables, and model binaries should be added only when they are available and reproducible from the notebook. They are intentionally not fabricated or committed from an unexecuted local run.

## Reproducibility

The complete analysis pipeline is available in [the research notebook](notebooks/bpn_pipeline.ipynb). The raw consolidated CSV is not committed because it exceeds GitHub's 100 MB file limit; it can be obtained from the official datasets above and placed in the configured data directory before execution.

The methodological review and requested corrections are recorded in [the reviewer feedback](remarks/reviewer-feedback.md).

## License

### Code

The source code developed for this research is licensed under the MIT License.

### Research Content

The original research content developed by the authors is licensed under
the Creative Commons Attribution 4.0 International License (CC BY 4.0).

When using or adapting this research, please provide appropriate
attribution to all credited authors.