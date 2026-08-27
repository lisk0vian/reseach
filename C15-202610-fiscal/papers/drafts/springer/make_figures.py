# -*- coding: utf-8 -*-
"""Build every manuscript figure directly from the exported result tables in src/tables/.

No model is re-fitted here: each figure plots values already produced by the Colab
pipeline, so figures and tables in the manuscript come from a single source of truth.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "tables")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#1f5c99", "#c2571a"
INK, MUTED, GRID = "#2c3138", "#6b727c", "#d7dbe0"
# Wash used to mark the partial 2026 file wherever it is plotted alongside
# complete years. It is a surface, never a data colour.
PARTIAL = "#eceef1"

# The two data hues pass all six checks of the palette validator against a white
# surface (worst adjacent pair dE 20.4 protan / 28.7 normal vision, both above
# the thresholds), so identity survives colour-vision deficiency and greyscale
# print. They are used categorically where two series share an axis and
# divergingly (blue below zero, orange above, neutral rule at zero) where the
# quantity has a sign. No third data hue is introduced: figures that would need
# one use small multiples instead.


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


# Readable English labels for the engineered feature names. The raw column name
# is kept in additional_file_3_feature_glossary.md so that a reader reproducing
# the pipeline can map each label back to the exact column.
FEATURE_LABELS = {
    "freq_inter_distrito_tipo_caso": "Judicial district × case type (freq.)",
    "freq_especializada": "Specialised-unit designation (freq.)",
    "freq_prov_pjfs": "Province (freq.)",
    "tipo_fiscalia_SUPERIOR": "Office type: superior",
    "freq_inter_tipo_fiscalia_especialidad": "Office type × specialty (freq.)",
    "freq_especialidad": "Specialty (freq.)",
    "hist_saldo_mean_prev_dist_pjfs": "Lagged mean case balance, district",
    "freq_dist_pjfs": "Judicial district (freq.)",
    "freq_tipo_caso": "Case type (freq.)",
    "ubigeo_pjfs": "Geographic code (ubigeo)",
    "tipo_caso_DENUNCIA": "Case type: complaint",
    "hist_ingresado_mean_prev_dist_pjfs": "Lagged mean cases received, district",
    "especializada_NO_ESPECIFICADO": "Specialised unit: not specified",
    "flag_nulo_especializada": "Missingness flag: specialised unit",
}


def nice_feature(name):
    return FEATURE_LABELS.get(name, name)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def tbl(name):
    return pd.read_excel(os.path.join(SRC, name))


def tidy(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1: framework
def fig_framework():
    stages = [
        ("1. Data integration and quality audit", "9,593 annual records, 2019-2026, 16 raw fields"),
        ("2. Temporal availability audit", "12 variables dropped: 11 unavailable, 1 collinear"),
        ("3. Proxy label and sensitivity scenarios", "P75/P25 principal; P70/P30 to P85/P15 tested"),
        ("4. Consensus feature selection", "6 selectors fitted on 2019-2022, checked on 2023"),
        ("5. Temporal training and model selection", "walk-forward cross-validation, 7 model families"),
        ("6. Threshold, calibration and utility", "threshold on 2024, locked evaluation on 2025"),
        ("7. Explainability, robustness, fairness", "permutation, SHAP, ablation, PSI, subgroups"),
    ]
    n = len(stages)
    step, bh = 1.00, 0.70
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * step + 0.55)
    ax.axis("off")
    for i, (title, sub) in enumerate(stages):
        yy = (n - 1 - i) * step + 0.20
        shaded = i in (1, 5)
        box = FancyBboxPatch((0.3, yy), 9.4, bh, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.0, edgecolor=BLUE if shaded else "#b9c1c9",
                             facecolor=BLUE if shaded else "#f4f6f8")
        ax.add_patch(box)
        ax.text(0.62, yy + 0.45, title, fontsize=8.4, fontweight="bold",
                color="white" if shaded else INK, va="center")
        ax.text(0.62, yy + 0.19, sub, fontsize=7.4,
                color="#d3e2f2" if shaded else MUTED, va="center")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((5.0, yy), (5.0, yy - (step - bh) + 0.02),
                                         arrowstyle="-|>", mutation_scale=9,
                                         color=MUTED, linewidth=0.9))
    ax.text(5.0, n * step + 0.32, "shaded stages are the components introduced by this work",
            fontsize=7.4, color=MUTED, ha="center", va="center", style="italic")
    save(fig, "fig01_framework.png")


# ------------------------------------------------- Fig 2: temporal protocol
def fig_protocol():
    p = tbl("Tabla_04_Particion_temporal__train_valid_test_externa_.xlsx")
    years = list(range(2019, 2027))
    role = {2019: "T", 2020: "T", 2021: "T", 2022: "T", 2023: "T",
            2024: "V", 2025: "E", 2026: "X"}
    colour = {"T": BLUE, "V": ORANGE, "E": "#3d4450", "X": "#f4f6f8"}
    label = {"T": "Training (2019-2023)", "V": "Threshold selection (2024)",
             "E": "Locked evaluation (2025)", "X": "External, partial year (2026)"}
    n = {2024: 1195, 2025: 1199, 2026: 1123}
    for _, r in p.iterrows():
        pass
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    for i, yr in enumerate(years):
        k = role[yr]
        ax.add_patch(FancyBboxPatch((i, 0.55), 0.9, 0.75,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor=colour[k], edgecolor="#b9c1c9", linewidth=0.8))
        ax.text(i + 0.45, 0.97 if k == "X" else 0.92, str(yr), ha="center", va="center",
                fontsize=8.4, fontweight="bold",
                color="white" if k in ("T", "V", "E") else INK)
        if k == "X":
            ax.text(i + 0.45, 0.76, "Jan-May", ha="center", va="center", fontsize=6.4,
                    color=MUTED)
    # feature-selection sub-window
    ax.add_patch(FancyBboxPatch((0, 0.20), 3.9, 0.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="none", edgecolor=BLUE, linewidth=1.0, linestyle="--"))
    ax.text(1.95, 0.32, "feature selection fitted here", ha="center", va="center",
            fontsize=7.2, color=BLUE)
    ax.add_patch(FancyBboxPatch((4.0, 0.20), 0.9, 0.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="none", edgecolor=BLUE, linewidth=1.0, linestyle="--"))
    ax.text(4.45, 0.32, "internal\ncheck", ha="center", va="center", fontsize=6.6, color=BLUE)
    ax.annotate("prediction is issued before the reporting year closes",
                xy=(6.45, 1.42), xytext=(3.2, 1.72), fontsize=7.4, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour[k], edgecolor="#b9c1c9")
               for k in ["T", "V", "E", "X"]]
    ax.legend(handles, [label[k] for k in ["T", "V", "E", "X"]],
              loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.2)
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(0, 2.0)
    ax.axis("off")
    save(fig, "fig02_protocol.png")


# ------------------------------------------- Fig 3: annual volumes and prevalence
def fig_temporal():
    t = tbl("Tabla_78_Resumen_anual_temporal_completo.xlsx")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a = axes[0]
    x = np.arange(len(t))
    a.bar(x - 0.2, t.ingresado_total / 1e6, width=0.38, color=BLUE, label="Cases received")
    a.bar(x + 0.2, t.atendido_total / 1e6, width=0.38, color=ORANGE, label="Cases processed")
    a.set_xticks(x)
    a.set_xticklabels(t.anio, rotation=45)
    a.set_ylabel("Million cases")
    a.set_title("Annual workload")
    a.set_ylim(0, 1.95)
    a.set_xlim(-0.6, len(t) - 0.4)
    a.axvspan(len(t) - 1.5, len(t) - 0.4, color=PARTIAL, linewidth=0, zorder=0)
    a.text(len(t) - 1, 0.70, "partial\nyear\n(Jan-May)", fontsize=6.4, color=MUTED,
           ha="center", va="bottom", linespacing=1.25)
    a.legend(loc="upper left", ncol=1)
    tidy(a)
    b = axes[1]
    # 2019-2025 are complete years and are joined by a solid line. The 2026 file
    # covers January-May only, so the segment leading to it is drawn dashed with
    # a hollow marker: connecting it as a trend would assert the very continuity
    # that Section 4.1 denies.
    full = t[t.anio <= 2025]
    b.axvspan(2025.5, 2026.55, color=PARTIAL, linewidth=0, zorder=0)
    b.plot(full.anio, full.riesgo_pct, marker="o", markersize=4.5, color=BLUE, linewidth=1.8)
    b.plot([2025, 2026], [t.riesgo_pct.iloc[-2], t.riesgo_pct.iloc[-1]],
           color=BLUE, linewidth=1.4, linestyle=(0, (3, 2)))
    b.plot([2026], [t.riesgo_pct.iloc[-1]], marker="o", markersize=5.4,
           markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.5)
    b.axhline(14.68, color=MUTED, linestyle=":", linewidth=1.0)
    b.text(2019.05, 15.2, "pooled prevalence 14.68%", fontsize=6.8, color=MUTED)
    for xx, yy, off in [(2019, 11.54, (7, 6)), (2020, 19.44, (0, 8)), (2026, 20.21, (-2, 9))]:
        b.annotate("%.1f%%" % yy, (xx, yy), textcoords="offset points",
                   xytext=off, ha="center", fontsize=7, color=INK)
    b.text(2026.0, 8.7, "partial\nyear", fontsize=6.6, color=MUTED, ha="center", va="bottom")
    b.set_ylabel("Proxy risk prevalence (%)")
    b.set_title("Label prevalence by year")
    b.set_ylim(8, 23)
    b.set_xlim(2018.5, 2026.6)
    b.set_xticks(t.anio)
    b.set_xticklabels(t.anio, rotation=45)
    tidy(b)
    save(fig, "fig03_temporal.png")


# ------------------------------------------------- Fig 4: walk-forward validation
def fig_walkforward():
    """Small multiples, one metric per panel.

    Three metrics on one axis would need a third data hue that no accessible
    palette supplies alongside blue and orange; one panel each also lets every
    metric use the y-range that makes its own variation legible.

    The 2026 fold evaluates a five-month file. Section 3.6 forbids it from
    supporting any performance claim, so it is separated from the six
    complete-year folds by a wash, a hollow marker and a dashed connector, and
    the range annotated on each panel is computed over 2020-2025 only.
    """
    w = tbl("Tabla_34_Validacion_Walk-Forward_temporal__anio_a_anio_.xlsx")
    full = w[w.eval_year <= 2025]
    metrics = [("roc_auc", "ROC-AUC", (0.78, 0.95)),
               ("f1", "F1-score", (0.45, 0.72)),
               ("pr_auc", "PR-AUC", (0.42, 0.80))]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True)
    for ax, (col, name, ylim) in zip(axes, metrics):
        ax.axvspan(2025.5, 2026.5, color=PARTIAL, linewidth=0, zorder=0)
        ax.plot(full.eval_year, full[col], marker="o", markersize=4.6, color=BLUE,
                linewidth=1.8)
        ax.plot([2025, 2026], [w[col].iloc[-2], w[col].iloc[-1]], color=BLUE,
                linewidth=1.3, linestyle=(0, (3, 2)))
        ax.plot([2026], [w[col].iloc[-1]], marker="o", markersize=5.6,
                markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.5)
        lo, hi = full[col].min(), full[col].max()
        ax.axhspan(lo, hi, color=BLUE, alpha=0.06, linewidth=0, zorder=0)
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(*ylim)
        ax.annotate("six complete-year folds: %.3f-%.3f" % (lo, hi),
                    xy=(0.015, 0.90), xycoords="axes fraction", fontsize=6.9,
                    color=MUTED, va="top")
        tidy(ax)
    axes[0].text(2026, axes[0].get_ylim()[0] + 0.006, "partial\nyear", fontsize=6.4,
                 color=MUTED, ha="center", va="bottom")
    axes[-1].set_xticks(w.eval_year)
    axes[-1].set_xlabel("Evaluation year (trained on every preceding year)")
    fig.subplots_adjust(hspace=0.16)
    save(fig, "fig04_walkforward.png")


# ------------------------------------------------------- Fig 5: learning curve
def fig_learning():
    lc = tbl("Tabla_56_F1_train_vs_F1_CV__walk-forward__segun_tamano_de_entrenamien.xlsx")
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(lc.train_size, lc.train_f1_mean, marker="o", markersize=4, color=ORANGE,
            linewidth=1.8, label="Training F1")
    ax.fill_between(lc.train_size, lc.train_f1_mean - lc.train_f1_std,
                    lc.train_f1_mean + lc.train_f1_std, color=ORANGE, alpha=0.15, linewidth=0)
    ax.plot(lc.train_size, lc.cv_f1_mean, marker="s", markersize=4, color=BLUE,
            linewidth=1.8, linestyle="--", label="Walk-forward CV F1")
    ax.fill_between(lc.train_size, lc.cv_f1_mean - lc.cv_f1_std,
                    lc.cv_f1_mean + lc.cv_f1_std, color=BLUE, alpha=0.15, linewidth=0)
    # Draw the gap the text quotes instead of pointing at empty space.
    xg = lc.train_size.iloc[-1]
    ytop, ybot = lc.train_f1_mean.iloc[-1], lc.cv_f1_mean.iloc[-1]
    ax.annotate("", xy=(xg, ytop), xytext=(xg, ybot),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, linewidth=0.9,
                                shrinkA=1, shrinkB=1))
    ax.annotate("persistent generalisation gap\n%.2f at the largest sample" % (ytop - ybot),
                xy=(xg, (ytop + ybot) / 2), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.2, color=MUTED)
    ax.text(0.015, 0.03, "shaded bands: $\\pm$1 SD across walk-forward folds",
            transform=ax.transAxes, fontsize=6.6, color=MUTED)
    ax.set_xlabel("Training records")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    tidy(ax)
    save(fig, "fig05_learning_curve.png")


# ------------------------------------ Fig 6: reliability with signed deviation
def fig_calibration():
    """Three panels, because two of them would misreport the result.

    A signed-deviation panel alone encodes the size of each bin's error but not
    its weight, so the tallest bar is the one resting on ten units while the bin
    holding 932 of the 1,199 units is almost invisible. Panel (a) therefore
    scales each marker by n and carries a Wilson interval, and panel (c) plots
    the n-weighted contribution to the expected calibration error, which is what
    the aggregate figure of 0.089 actually sums.
    """
    r = tbl("Tabla_49_Bins_del_reliability_diagram__confianza_vs_tasa_observada_.xlsx")
    r = r[r.n > 0].copy()
    r["signed"] = r.confianza_promedio - r.tasa_observada
    r["events"] = (r.n * r.tasa_observada).round().astype(int)
    total_n = int(r.n.sum())
    r["contrib"] = r.n * r.signed.abs() / total_n
    ece = r.contrib.sum()
    under = r.loc[r.signed < 0, "contrib"].sum()
    ci = [wilson(k, n) for k, n in zip(r.events, r.n)]
    err_lo = r.tasa_observada.values - np.array([c[0] for c in ci])
    err_hi = np.array([c[1] for c in ci]) - r.tasa_observada.values

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9))

    a = axes[0]
    a.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.1, zorder=1)
    a.text(0.74, 0.86, "perfect\ncalibration", fontsize=6.2, color=MUTED, rotation=42,
           ha="center", va="center")
    a.errorbar(r.confianza_promedio, r.tasa_observada, yerr=[err_lo, err_hi],
               fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=1.8, zorder=2)
    a.plot(r.confianza_promedio, r.tasa_observada, color=BLUE, linewidth=1.3, zorder=3)
    # Marker area encodes the bin size, so the bin holding 932 of the 1,199 units
    # no longer looks like the one holding 10. Exact counts are labelled in (b).
    a.scatter(r.confianza_promedio, r.tasa_observada, s=10 + 170 * r.n / total_n,
              color=BLUE, edgecolor="white", linewidth=0.7, zorder=4)
    a.annotate("marker area\n$\\propto$ bin size", xy=(0.011, 0.072),
               xytext=(0.20, 0.055), fontsize=6.2, color=MUTED, va="center",
               arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.7))
    a.set_xlabel("Mean predicted probability", fontsize=7.0)
    a.set_ylabel("Observed proxy risk rate", fontsize=7.0)
    a.set_title("(a) Reliability", fontsize=8)
    a.set_xlim(-0.03, 1.03)
    a.set_ylim(-0.03, 1.03)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.tick_params(labelsize=6.6)
    tidy(a)

    b = axes[1]
    cols = [ORANGE if v > 0 else BLUE for v in r.signed]
    b.bar(r.confianza_promedio, r.signed, width=0.070, color=cols)
    b.axhline(0, color=MUTED, linewidth=0.9)
    for _, row in r.iterrows():
        off = 3 if row.signed > 0 else -8
        b.annotate("%d" % row.n, (row.confianza_promedio, row.signed),
                   textcoords="offset points", xytext=(0, off), ha="center",
                   fontsize=5.4, color=MUTED)
    b.set_xlabel("Mean predicted probability", fontsize=7.0)
    b.set_ylabel("Predicted $-$ observed", fontsize=7.0)
    b.set_title("(b) Signed deviation", fontsize=8)
    b.set_xlim(-0.03, 1.03)
    b.set_ylim(-0.19, 0.62)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    b.tick_params(labelsize=6.6)
    b.annotate("MCE 0.461\nrests on n=10", xy=(0.761, 0.470), xytext=(0.40, 0.575),
               fontsize=6.2, color=INK, ha="center",
               arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.7))
    tidy(b)

    c = axes[2]
    c.bar(r.confianza_promedio, r.contrib, width=0.070, color=cols)
    c.axhline(0, color=MUTED, linewidth=0.9)
    share_low = 100 * r.contrib.iloc[0] / ece
    c.annotate("932 units here\nsupply %.0f%% of ECE" % share_low,
               xy=(0.05, r.contrib.iloc[0] * 0.92), xytext=(0.30, 0.0455), fontsize=6.2,
               color=INK, arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.7))
    c.set_xlabel("Mean predicted probability", fontsize=7.0)
    c.set_ylabel(r"$(n_b/n)\times|$deviation$|$", fontsize=7.0)
    c.set_title("(c) Weighted contribution", fontsize=8)
    c.set_xlim(-0.03, 1.03)
    c.set_ylim(0, 0.060)
    c.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    c.tick_params(labelsize=6.6)
    tidy(c)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE),
               plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE)]
    fig.legend(handles,
               ["under-estimation: %.0f%% of the ECE of %.3f" % (100 * under / ece, ece),
                "over-estimation: %.0f%%" % (100 * (ece - under) / ece)],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               handlelength=1.1, fontsize=7)
    fig.subplots_adjust(wspace=0.46, bottom=0.30, top=0.90, left=0.09, right=0.985)
    save(fig, "fig06_calibration.png")


# --------------------------------------------------- Fig 7: risk stratification
STRATA = None


def strata_table():
    global STRATA
    r = tbl("Tabla_49_Bins_del_reliability_diagram__confianza_vs_tasa_observada_.xlsx")
    r = r[r.n > 0].copy()
    r["events"] = (r.n * r.tasa_observada).round().astype(int)
    band = pd.cut(r.bin_lo, bins=[-0.01, 0.099, 0.299, 0.599, 1.0],
                  labels=["Low\n<0.10", "Moderate\n0.10-0.30", "High\n0.30-0.60", "Very high\n>=0.60"])
    g = r.groupby(band, observed=False).agg(n=("n", "sum"), events=("events", "sum"),
                                            pred=("confianza_promedio", "mean")).reset_index()
    g["observed"] = g.events / g.n
    g["share_units"] = g.n / g.n.sum()
    g["share_events"] = g.events / g.events.sum()
    STRATA = g
    return g


def fig_strata():
    g = strata_table()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    a = axes[0]
    bars = a.bar(range(len(g)), g.observed * 100, color=BLUE, width=0.62)
    a.axhline(14.35, color=MUTED, linestyle=":", linewidth=1.0)
    a.text(3.42, 15.6, "base rate 14.35%", fontsize=7, color=MUTED, ha="right")
    for i, (rect, row) in enumerate(zip(bars, g.itertuples())):
        a.annotate("%.1f%%\n(n=%d)" % (row.observed * 100, row.n),
                   (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                   textcoords="offset points", xytext=(0, 4), ha="center",
                   fontsize=7, color=INK)
    a.set_xticks(range(len(g)))
    a.set_xticklabels(g.iloc[:, 0], fontsize=7.4)
    a.set_ylabel("Observed proxy risk rate (%)")
    a.set_title("Observed risk by predicted stratum")
    a.set_ylim(0, 68)
    tidy(a)
    b = axes[1]
    x = np.arange(len(g))
    b.bar(x - 0.19, g.share_units * 100, width=0.36, color=BLUE, label="Share of units")
    b.bar(x + 0.19, g.share_events * 100, width=0.36, color=ORANGE, label="Share of proxy events")
    for i, row in enumerate(g.itertuples()):
        b.annotate("%.0f%%" % (row.share_units * 100), (i - 0.19, row.share_units * 100),
                   textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.8, color=INK)
        b.annotate("%.0f%%" % (row.share_events * 100), (i + 0.19, row.share_events * 100),
                   textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.8, color=INK)
    b.set_xticks(x)
    b.set_xticklabels(g.iloc[:, 0], fontsize=7.4)
    b.set_ylabel("Percentage of the 2025 test set")
    b.set_title("Concentration of proxy events")
    b.set_ylim(0, 90)
    b.legend(ncol=1, loc="upper right")
    tidy(b)
    save(fig, "fig07_risk_strata.png")


# ------------------------------------------------------- Fig 8: decision curve
def dca_table():
    r = tbl("Tabla_49_Bins_del_reliability_diagram__confianza_vs_tasa_observada_.xlsx")
    r = r[r.n > 0].copy()
    r["events"] = (r.n * r.tasa_observada).round().astype(int)
    r = r.sort_values("bin_lo", ascending=False)
    n = int(r.n.sum())
    prev = r.events.sum() / n
    rows = []
    cn = ce = 0
    for _, row in r.iterrows():
        cn += int(row.n)
        ce += int(row.events)
        pt = float(row.bin_lo)
        if pt <= 0:
            continue
        tp, fp = ce, cn - ce
        odds = pt / (1 - pt)
        rows.append(dict(pt=pt, alerts=cn, tp=tp, fp=fp,
                         nb_model=tp / n - (fp / n) * odds,
                         nb_all=prev - (1 - prev) * odds, nb_none=0.0))
    d = pd.DataFrame(rows).sort_values("pt")
    d["delta"] = d.nb_model - d[["nb_all", "nb_none"]].max(axis=1)
    return d, n, prev


def fig_dca():
    d, n, prev = dca_table()
    # operating point tau = 0.65 from the confusion matrix of Table 73
    e = tbl("Tabla_73_Resumen_de_errores__TP_TN_FP_FN__-_test_2025.xlsx").set_index("tipo_error")["registros"]
    tp, fp = int(e["TP"]), int(e["FP"])
    tau = 0.65
    nb_op = tp / n - (fp / n) * (tau / (1 - tau))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(d.pt, d.nb_model, marker="o", markersize=4.5, color=BLUE, linewidth=1.9,
            label="Model")
    ax.plot(d.pt, d.nb_all, color=ORANGE, linewidth=1.6, linestyle="--",
            label="Flag every unit")
    ax.axhline(0, color=MUTED, linewidth=1.1, linestyle=":")
    ax.text(0.015, 0.008, "Flag no unit", fontsize=7.4, color=MUTED)
    # The operating point belongs to the model series, so it carries the model's
    # colour; only its shape distinguishes it. Painting it orange would assign it
    # to the "flag every unit" strategy it has nothing to do with.
    ax.plot([tau], [nb_op], marker="D", markersize=6.5, color=BLUE,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate("operating point $\\tau$ = 0.65\nnet benefit = %.3f (negative)" % nb_op,
                xy=(tau, nb_op), xytext=(0.38, -0.135), fontsize=7.4, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.9))
    ax.text(0.985, 0.02, "grid rows reconstructed from fixed-width reliability bins;\n"
                         "the operating point is exact",
            transform=ax.transAxes, fontsize=6.3, color=MUTED, ha="right", va="bottom")
    lo = d[d.delta > 0].pt.min()
    hi = d[d.delta > 0].pt.max()
    ax.axvspan(lo, hi, color=BLUE, alpha=0.07, linewidth=0)
    ax.text((lo + hi) / 2, 0.088, "model preferred\n$p_t \\in$ [%.1f, %.1f]" % (lo, hi),
            ha="center", fontsize=7.4, color=MUTED)
    ax.set_xlabel("Threshold probability $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.22, 0.115)
    ax.legend(loc="upper right", ncol=1)
    tidy(ax)
    save(fig, "fig08_decision_curve.png")
    return d, nb_op


# ------------------------------------------------------ Fig 9: variable importance
def fig_importance():
    p = tbl("Tabla_50_Permutation_Importance__test_2025_.xlsx").head(12)[::-1]
    s = tbl("Tabla_51_Ranking_SHAP_global.xlsx").head(12)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.6))
    for ax, dat, val, err, ttl, col in [
            (axes[0], p, "importance_mean", "importance_std",
             "Permutation importance\n(XGBoost, test 2025)", BLUE),
            (axes[1], s, "mean_abs_shap", None,
             "Mean absolute SHAP\n(LightGBM-Optuna, test 2025)", ORANGE)]:
        xerr = dat[err] if err and err in dat.columns else None
        ax.barh(range(len(dat)), dat[val], color=col, height=0.66,
                xerr=xerr, error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=1.8))
        ax.set_yticks(range(len(dat)))
        # Full readable labels, never truncated; the raw column names are given in
        # additional_file_3_feature_glossary.md.
        ax.set_yticklabels([nice_feature(f) for f in dat.feature], fontsize=6.3)
        ax.set_title(ttl, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_xlabel("Mean decrease in F1 ($\\pm$1 SD over permutations)", fontsize=7.2)
    axes[1].set_xlabel("Mean absolute SHAP value", fontsize=7.2)
    fig.subplots_adjust(wspace=1.45)
    save(fig, "fig09_importance.png")


# --------------------------------------------------------- Fig 10: ablation
def fig_ablation():
    a = tbl("Tabla_76_Ablation_Study__impacto_de_retirar_bloques_de_variables_.xlsx")
    a = a[a.dataset == "test_2025"].copy()
    nice = {"full_selected_features": "All 74 selected features",
            "without_historical_growth": "- historical / growth block",
            "without_territorial": "- territorial block",
            "without_frequency_encoding": "- frequency encodings",
            "without_interactions": "- categorical interactions"}
    a["label"] = a.variant.map(nice) + " (" + a.n_features.astype(str) + ")"
    base = a[a.variant == "full_selected_features"].f1.iloc[0]
    # Plot the change from the full feature set rather than the absolute F1. The
    # quantity of interest has a sign, so it gets a diverging encoding centred on
    # zero; the previous version painted every variant the same colour whether
    # removing the block helped or hurt.
    a = a[a.variant != "full_selected_features"].copy()
    a["delta"] = a.f1 - base
    a = a.sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    cols = [ORANGE if d < 0 else BLUE for d in a.delta]
    bars = ax.barh(range(len(a)), a.delta, color=cols, height=0.62)
    ax.axvline(0, color=MUTED, linewidth=1.0)
    for i, (rect, row) in enumerate(zip(bars, a.itertuples())):
        off = 5 if row.delta > 0 else -5
        ha = "left" if row.delta > 0 else "right"
        ax.annotate("%+.3f  (F1 %.3f)" % (row.delta, row.f1), (rect.get_width(), i),
                    textcoords="offset points", xytext=(off, 0), va="center",
                    ha=ha, fontsize=7, color=INK)
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels(a.label, fontsize=7.4)
    ax.set_xlabel("Change in F1 when the block is removed, against all 74 features\n"
                  "(untuned LightGBM, 2025 test set, decision threshold 0.50)", fontsize=7.6)
    ax.set_xlim(-0.105, 0.085)
    # Direction is already carried by the sign of the axis and by the diverging
    # fill; these keys stay in ink so that no text borrows a series colour.
    ax.text(-0.100, len(a) - 0.42, "$\\leftarrow$ removal degrades", fontsize=6.8,
            color=MUTED, ha="left")
    ax.text(0.080, len(a) - 0.42, "removal improves $\\rightarrow$", fontsize=6.8,
            color=MUTED, ha="right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    save(fig, "fig10_ablation.png")


if __name__ == "__main__":
    fig_framework()
    fig_protocol()
    fig_temporal()
    fig_walkforward()
    fig_learning()
    fig_calibration()
    fig_strata()
    d, nb_op = fig_dca()
    fig_importance()
    fig_ablation()

    print("\n--- derived risk strata (test 2025, from Table 49) ---")
    g = STRATA.copy()
    g.columns = ["stratum", "n", "events", "mean_predicted", "observed", "share_units", "share_events"]
    g["stratum"] = [s.replace("\n", " ") for s in g.stratum.astype(str)]
    print(g.round(4).to_string(index=False))
    g.to_csv(os.path.join(OUT, "..", "derived_strata.csv"), index=False)

    print("\n--- derived decision curve (test 2025) ---")
    print(d.round(4).to_string(index=False))
    print("net benefit at the operating point tau=0.65: %.4f" % nb_op)
    d.to_csv(os.path.join(OUT, "..", "derived_dca.csv"), index=False)
