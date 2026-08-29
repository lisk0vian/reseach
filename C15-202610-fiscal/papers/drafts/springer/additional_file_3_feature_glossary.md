# Additional file 3 — Feature glossary

Companion to: *Prosecutorial congestion in Peru, 2019–2026: a leakage-audited,
calibration-aware framework for predicting proxy signals from open
administrative records.*

Figure 9 labels variables in readable English. The pipeline names them in
Spanish, after the field names of the source registry. This table maps one to
the other so that a reader reproducing the analysis can locate the exact column.

## Variables appearing in Figure 9

| Label in Figure 9 | Column in the pipeline | What it is |
|---|---|---|
| Judicial district × case type (freq.) | `freq_inter_distrito_tipo_caso` | Frequency encoding of the judicial district × case type interaction |
| Specialised-unit designation (freq.) | `freq_especializada` | Frequency encoding of the specialised-unit descriptor |
| Province (freq.) | `freq_prov_pjfs` | Frequency encoding of the province of the judicial district |
| Office type: superior | `tipo_fiscalia_SUPERIOR` | One-hot indicator for superior-level prosecution offices |
| Office type × specialty (freq.) | `freq_inter_tipo_fiscalia_especialidad` | Frequency encoding of the office type × specialty interaction |
| Specialty (freq.) | `freq_especialidad` | Frequency encoding of the specialty |
| Lagged mean case balance, district | `hist_saldo_mean_prev_dist_pjfs` | Expanding mean of the case balance over years strictly earlier than the target year, by fiscal district |
| Judicial district (freq.) | `freq_dist_pjfs` | Frequency encoding of the judicial district |
| Case type (freq.) | `freq_tipo_caso` | Frequency encoding of the case type |
| Geographic code (ubigeo) | `ubigeo_pjfs` | Peruvian geographic code of the judicial district |
| Case type: complaint | `tipo_caso_DENUNCIA` | One-hot indicator for the *denuncia* case type |
| Lagged mean cases received, district | `hist_ingresado_mean_prev_dist_pjfs` | Expanding mean of cases received over years strictly earlier than the target year, by fiscal district |
| Specialised unit: not specified | `especializada_NO_ESPECIFICADO` | One-hot indicator for the explicit "not specified" category of the specialised-unit descriptor |
| Missingness flag: specialised unit | `flag_nulo_especializada` | Binary indicator that the specialised-unit descriptor was absent in the source record |

## Naming conventions

| Prefix | Meaning |
|---|---|
| `freq_` | Frequency encoding, with frequencies estimated on the training partition (2019–2023) alone and mapped onto later partitions; unseen categories receive zero |
| `freq_inter_` | Frequency encoding of a second-order categorical interaction |
| `hist_` | Lagged aggregate computed as an expanding mean over the *previous* years of the same fiscal district, obtained by shifting the annual series one period before aggregation, so that no record contributes to its own history |
| `flag_nulo_` | Binary missingness indicator retained alongside an imputed field |
| `_pjfs` | Suffix marking a field of the judicial-district geography (*Poder Judicial / Fiscalía*) |
| bare uppercase suffix | One-hot level of a categorical field, e.g. `tipo_fiscalia_SUPERIOR` |

## Source fields

The engineered variables above derive from the raw fields of the *Casos
Fiscales* registry: `periodo`, `anio`, `distrito_fiscal`, `tipo_fiscalia`,
`materia`, `especialidad`, `tipo_caso`, `ingresado`, `atendido`, `ubigeo_pjfs`,
`dpto_pjfs`, `prov_pjfs`, `dist_pjfs`, `especializada`, `fecha_descarga`,
`fecha_corte`.

`ingresado` and `atendido` are the two volumetric fields excluded by the
temporal availability audit; they construct the proxy label and never enter the
feature set. The complete list of 74 selected features, and the 12 excluded
variables, are in the repository named under *Availability of data and
materials*.
