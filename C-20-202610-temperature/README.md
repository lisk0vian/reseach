# Subseasonal Temperature Forecasting in Andean Stations: A Benchmark of Machine Learning and Deep Learning Models with Anomaly Decomposition

**Area**: Meteorology & Climate Informatics · **Status**: `In Progress` · **Journal**: Pending decision

**Description:**
This research proposes a reproducible machine learning framework for subseasonal air temperature forecasting at the Huayao meteorological station (Junín, Perú), leveraging hourly observations from the IGP Automatic Meteorological Station (EMA) network covering the period 2018–2025. The study introduces a thermodynamic preprocessing pipeline with zero-leakage climatological decomposition, enabling anomaly-based modeling that separates the deterministic seasonal signal from the stochastic component. The benchmarking covers classical ensemble methods (Random Forest, XGBoost, LightGBM, CatBoost), a deep recurrent architecture (LSTM in closed-loop), and a state-of-the-art probabilistic model (Temporal Fusion Transformer), all evaluated under a sliding multi-window protocol over the full year 2025. An ablation study quantifies the incremental contribution of each physical feature group to forecast skill.

## Research Team


### Authors

* **Moisés Evangelista Gamarra** — Principal Researcher
  <br />
  [ORCID](https://orcid.org/0009-0002-5382-1390) · [Google Scholar](https://scholar.google.com/citations?user=urGKMNQAAAAJ&hl=es)

* **Jerremi Aron Chancan Labajos** — Co-author
  <br />
   [ORCID](https://orcid.org/) · [Google Scholar](https://scholar.google.com/citations?hl=es&user=wV2fQskAAAAJ) · [LinkedIn](https://www.linkedin.com/in/jeremi-aron/) · [GitHub](https://github.com/lisk0vian)

### Affiliations

* [Servicio Nacional de Adiestramiento en Trabajo Industrial, Lima, Perú](https://www.senati.edu.pe/especialidades/tecnologias-de-la-informacion/ingenieria-de-software-con-inteligencia-artificial)

## Dates

| Milestone            | Date        |
| -------------------- | ----------- |
| Research Start       | DD-MMM-2026 |
| Experiments          | DD-MMM-2026 |
| Manuscript Draft     | DD-MMM-2026 |
| Internal Review      | DD-MMM-2026 |
| Expected Publication | DD-MMM-YYYY |

## Links

**Dataset:** [IGP EMA — Estación Huayao (2018–2025)](https://www.datosabiertos.gob.pe/)

## Repository Structure

```text
202610-temperature-forecasting/
├── data/
│   └── IGP_EstacionEMA_data_2018_2025.csv   # Raw hourly observations from IGP EMA network
├── notebooks/
│   ├── preprocess-andean-dataset.ipynb      # Phase 1 — Thermodynamic preprocessing & feature engineering
│   └── huayao-pipeline.ipynb                # Phase 2–4 — EDA, baselines, LSTM & TFT benchmarking
└── README.md
```

## Methodology Overview

### Phase 1 — Thermodynamic Preprocessing (`preprocess-andean-dataset.ipynb`)

- **Data ingestion & sanitization**: physical variables (`TT`, `HR`, `PP`, `RR`, `FF`, `DD`) cast to numeric, admin columns discarded.
- **Temporal grid alignment**: uniform hourly index reconstructed; connectivity gaps audited and reported.
- **Causal imputation**: forward-fill applied strictly in the temporal direction; precipitation filled with zero; initial NaN rows dropped.
- **Feature engineering**: harmonic encodings of diurnal and seasonal cycles (`hour_sin/cos`, `month_sin/cos`); wind decomposed into zonal/meridional components (`wind_u`, `wind_v`); Magnus dew-point depression (`dew_point_dep`) computed psicrométrically.
- **Zero-leakage climatology**: mean climatological signal extracted exclusively from the training partition (2018–2023) and subtracted from the full series, producing the thermal anomaly target (`TT_anomaly`).
- **Diagnostic outputs**: variability audit table, null-gap audit table, and four publication-quality figures (300 DPI).

### Phase 2 — EDA & Ensemble Baselines (`huayao-pipeline.ipynb`)

- **EDA**: historical scatter, temperature KDE, and Spearman correlation matrix.
- **Ablation study**: four feature configurations (A–D) tested with LightGBM in recursive inference mode; incremental MAE reduction quantified.
- **Ensemble benchmark**: Random Forest, XGBoost, LightGBM, CatBoost — hyperparameters optimized via Bayesian search (Optuna/TPE) on the 2024 validation set. Seasonal Naïve used as baseline.
- **Evaluation protocol**: sliding multi-window over 12 monthly start dates in 2025, horizon = 720 h (30 days). Metrics: MAE, RMSE, MASE, MBE, R².
- **Recursive inference**: closed-loop autoregressive inference for long-horizon forecasting.

### Phase 3 — Deep Recurrent Learning (`huayao-pipeline.ipynb`)

- **LSTM architecture**: 2-layer LSTM (hidden = 64, dropout = 0.2) trained for up to 35 epochs with early stopping on MSE.
- **Input sequence**: 168-hour context window over scaled anomaly and harmonic features.
- **Closed-loop inference**: predictions autoregressed; climatology added back to reconstruct the absolute temperature forecast.
- **Outputs**: convergence curve, forecast vs. observed, residual distribution (June 2025 window).

### Phase 4 — Temporal Fusion Transformer (`huayao-pipeline.ipynb`)

- **TFT configuration**: hidden size = 16, attention heads = 2, dropout = 0.18, QuantileLoss (7 quantiles), encoder length = 360 h.
- **Known future covariates**: harmonic encodings, climatology signal, hour/month categorical indices.
- **Probabilistic evaluation**: 80% prediction interval coverage (PICP) and mean prediction interval width (MPIW) reported alongside point metrics.
- **Interpretability**: encoder/decoder variable importance extracted via attention weights.

## Variables

| Variable          | Description                                  | Unit |
| ----------------- | -------------------------------------------- | ---- |
| `TT`              | Air temperature                              | °C   |
| `HR`              | Relative humidity                            | %    |
| `PP`              | Atmospheric pressure                         | hPa  |
| `RR`              | Precipitation                                | mm   |
| `wind_u`          | Zonal wind component                         | m/s  |
| `wind_v`          | Meridional wind component                    | m/s  |
| `dew_point_dep`   | Dew-point depression (Magnus formula)        | °C   |
| `TT_climatology`  | Mean climatological temperature (train only) | °C   |
| `TT_anomaly`      | Thermal anomaly (residual from climatology)  | °C   |
| `hour_sin/cos`    | Harmonic encoding of daily cycle             | —    |
| `month_sin/cos`   | Harmonic encoding of annual cycle            | —    |

## License

### Code

The source code developed for this research is licensed under the MIT License.

### Research Content

The original research content developed by the authors is licensed under
the Creative Commons Attribution 4.0 International License (CC BY 4.0).

When using or adapting this research, please provide appropriate
attribution to all credited authors.
