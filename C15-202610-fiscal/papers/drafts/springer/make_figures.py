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
        ("2. Temporal availability audit", "12 variables excluded before any modelling"),
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
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for i, yr in enumerate(years):
        k = role[yr]
        ax.add_patch(FancyBboxPatch((i, 0.55), 0.9, 0.75,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor=colour[k], edgecolor="#b9c1c9", linewidth=0.8))
        ax.text(i + 0.45, 0.92, str(yr), ha="center", va="center", fontsize=8.4,
                fontweight="bold", color="white" if k in ("T", "V", "E") else INK)
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
    a.legend(loc="upper left", ncol=1)
    a.annotate("partial year", xy=(7, 0.68), xytext=(5.9, 1.15), fontsize=7, ha="center",
               color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.7))
    tidy(a)
    b = axes[1]
    b.plot(t.anio, t.riesgo_pct, marker="o", markersize=4.5, color=BLUE, linewidth=1.8)
    b.axhline(14.68, color=MUTED, linestyle=":", linewidth=1.0)
    b.text(2019.05, 15.1, "pooled prevalence 14.68%", fontsize=7, color=MUTED)
    for xx, yy, off in [(2019, 11.54, (6, 7)), (2020, 19.44, (0, 8)), (2026, 20.21, (-6, 8))]:
        b.annotate("%.1f%%" % yy, (xx, yy), textcoords="offset points",
                   xytext=off, ha="center", fontsize=7, color=INK)
    b.set_ylabel("Proxy risk prevalence (%)")
    b.set_title("Label prevalence by year")
    b.set_ylim(8, 23)
    b.set_xticks(t.anio)
    b.set_xticklabels(t.anio, rotation=45)
    tidy(b)
    save(fig, "fig03_temporal.png")


# ------------------------------------------------- Fig 4: walk-forward validation
def fig_walkforward():
    w = tbl("Tabla_34_Validacion_Walk-Forward_temporal__anio_a_anio_.xlsx")
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(w.eval_year, w.roc_auc, marker="o", markersize=4.5, color=BLUE,
            linewidth=1.8, label="ROC-AUC")
    ax.plot(w.eval_year, w.f1, marker="s", markersize=4.5, color=ORANGE,
            linewidth=1.8, linestyle="--", label="F1-score")
    ax.plot(w.eval_year, w.pr_auc, marker="^", markersize=4.5, color=MUTED,
            linewidth=1.3, linestyle=":", label="PR-AUC")
    for xx, yy in [(w.eval_year.iloc[0], w.roc_auc.iloc[0]), (w.eval_year.iloc[-1], w.roc_auc.iloc[-1])]:
        ax.annotate("%.3f" % yy, (xx, yy), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(w.eval_year)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Evaluation year (trained on all previous years)")
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="lower right")
    tidy(ax)
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
    ax.annotate("persistent generalisation gap\n(0.41 at the largest sample)",
                xy=(4050, 0.69), xytext=(2100, 0.72), fontsize=7.2, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8))
    ax.set_xlabel("Training records")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    tidy(ax)
    save(fig, "fig05_learning_curve.png")


# ------------------------------------ Fig 6: reliability with signed deviation
def fig_calibration():
    r = tbl("Tabla_49_Bins_del_reliability_diagram__confianza_vs_tasa_observada_.xlsx")
    r = r[r.n > 0].copy()
    r["signed"] = r.confianza_promedio - r.tasa_observada
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    a = axes[0]
    a.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.1)
    a.text(0.70, 0.79, "perfect\ncalibration", fontsize=7, color=MUTED, rotation=40,
           ha="center", va="center")
    a.plot(r.confianza_promedio, r.tasa_observada, marker="o", markersize=5,
           color=BLUE, linewidth=1.8)
    a.annotate("every bin above 0.6 falls\nwell below the diagonal",
               xy=(0.761, 0.30), xytext=(0.40, 0.14), fontsize=7.2, color=INK,
               arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8))
    a.set_xlabel("Mean predicted probability")
    a.set_ylabel("Observed proxy risk rate")
    a.set_title("Reliability, test 2025")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.02)
    tidy(a)
    b = axes[1]
    cols = [ORANGE if v > 0 else BLUE for v in r.signed]
    b.bar(r.confianza_promedio, r.signed, width=0.075, color=cols)
    b.axhline(0, color=MUTED, linewidth=0.9)
    b.set_xlabel("Mean predicted probability")
    b.set_ylabel("Predicted minus observed")
    b.set_title("Direction of miscalibration")
    b.set_xlim(0, 1)
    b.set_ylim(-0.16, 0.62)
    b.text(0.06, 0.50, "over-estimation\n(predicted > observed)", fontsize=7.2,
           color=ORANGE, va="center")
    b.text(0.06, -0.125, "under-estimation", fontsize=7.2, color=BLUE)
    b.annotate("MCE = 0.461", xy=(0.761, 0.461), xytext=(0.50, 0.545), fontsize=7,
               color=INK, ha="center",
               arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8))
    tidy(b)
    fig.subplots_adjust(wspace=0.34)
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
    a.text(-0.45, 16.2, "base rate 14.35%", fontsize=7, color=MUTED, ha="left")
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
    ax.plot([tau], [nb_op], marker="D", markersize=6.5, color=ORANGE,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax.annotate("operating point $\\tau$ = 0.65\nnet benefit = %.3f (negative)" % nb_op,
                xy=(tau, nb_op), xytext=(0.38, -0.135), fontsize=7.4, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.9))
    lo = d[d.delta > 0].pt.min()
    hi = d[d.delta > 0].pt.max()
    ax.axvspan(lo, hi, color=BLUE, alpha=0.07, linewidth=0)
    ax.text((lo + hi) / 2, 0.088, "model preferred\n$p_t \\in$ [%.1f, %.1f]" % (lo, hi),
            ha="center", fontsize=7.4, color=BLUE)
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
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.6))
    for ax, dat, val, ttl, col in [
            (axes[0], p, "importance_mean", "Permutation importance\n(XGBoost, test 2025)", BLUE),
            (axes[1], s, "mean_abs_shap", "Mean absolute SHAP\n(LightGBM-Optuna, test 2025)", ORANGE)]:
        ax.barh(range(len(dat)), dat[val], color=col, height=0.66)
        ax.set_yticks(range(len(dat)))
        ax.set_yticklabels([(f[:27] + "..." if len(f) > 30 else f) for f in dat.feature],
                           fontsize=6.4)
        ax.set_title(ttl, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
    axes[0].set_xlabel("Mean decrease in F1", fontsize=7.6)
    axes[1].set_xlabel("Mean absolute SHAP value", fontsize=7.6)
    fig.subplots_adjust(wspace=0.62)
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
    a = a.sort_values("f1")
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    cols = [BLUE if v == "full_selected_features" else ORANGE for v in a.variant]
    bars = ax.barh(range(len(a)), a.f1, color=cols, height=0.62)
    ax.axvline(base, color=MUTED, linestyle=":", linewidth=1.0)
    for i, (rect, row) in enumerate(zip(bars, a.itertuples())):
        delta = row.f1 - base
        txt = "%.3f" % row.f1 if row.variant == "full_selected_features" else "%.3f (%+.3f)" % (row.f1, delta)
        ax.annotate(txt, (rect.get_width(), i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels(a.label, fontsize=7.4)
    ax.set_xlabel("F1-score on the 2025 test set (decision threshold 0.50)")
    ax.set_xlim(0, 0.62)
    ax.spines[["top", "right"]].set_visible(False)
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
