from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfm.data.quality import HYPER, HYPO, SEVERE_HYPO, TARGET_LOW
from tfm.paths import FIGURES_DIR
from tfm.viz import style
from tfm.viz.style import CATEGORICAL, FIG_WIDTH_FULL, INK, ROLE

COHORT_COLORS = {"2018": ROLE["modelo"], "2020": ROLE["referencia"]}
HYPO_COLOR = ROLE["hipo"]
RANGE_COLOR = ROLE["rango_objetivo"]

HORIZON_COLORS = {30: ROLE["modelo"], 60: CATEGORICAL[1]}


def set_style() -> None:
    style.apply()


def add_glycemic_bands(ax, show_severe: bool = True) -> None:
    ax.axhspan(TARGET_LOW, HYPER, color=RANGE_COLOR, alpha=0.6, lw=0, zorder=0)
    ax.axhline(HYPO, color=HYPO_COLOR, lw=0.9, ls="--", zorder=1)
    ax.axhline(HYPER, color=ROLE["hiper"], lw=0.9, ls="--", zorder=1)
    if show_severe:
        ax.axhline(SEVERE_HYPO, color=HYPO_COLOR, lw=0.9, ls=":", zorder=1)
    ax.set_ylabel("Glucosa (mg/dL)")


def save_figure(fig, name: str, formats: tuple[str, ...] = ("pdf", "png")) -> list:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    return paths


def plot_day(ax, cgm, day, label: str | None = None):
    import pandas as pd

    day = pd.Timestamp(day).normalize()
    m = (cgm["ts"] >= day) & (cgm["ts"] < day + pd.Timedelta(days=1))
    sub = cgm.loc[m]
    ax.plot(sub["ts"], sub["value"], lw=1.4, color=ROLE["observado"], label=label)
    hypo = sub[sub["value"] < HYPO]
    ax.scatter(hypo["ts"], hypo["value"], s=14, color=HYPO_COLOR, zorder=3)
    add_glycemic_bands(ax)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax.set_xlabel("Hora")
    return ax


def plot_baseline_comparison(ax, table, metric: str, label: str, higher_is_better=False):
    order = table.sort_values(metric, ascending=not higher_is_better)
    colors = [ROLE["modelo"] if i == 0 else ROLE["referencia"] for i in range(len(order))]
    ax.barh(order["exp_id"], order[metric], color=colors)
    ax.set_xlabel(label)
    ax.invert_yaxis()
    for y, v in enumerate(order[metric]):
        ax.text(v, y, f" {v:.3f}" if v < 10 else f" {v:.2f}", va="center", fontsize=8)
    return ax


def plot_recall_with_ci(ax, table, label: str = "Recall de hipoglucemia (eventos)"):
    y = range(len(table))
    lo = table["recall"] - table["recall_ci_lo"]
    hi = table["recall_ci_hi"] - table["recall"]
    ax.errorbar(table["recall"], y, xerr=[lo, hi], fmt="o", color=ROLE["modelo"],
                ecolor=ROLE["modelo"], capsize=3, markersize=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(table["exp_id"])
    ax.set_xlabel(label)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3)
    return ax


REGIME_LABELS = {
    "global": "Global-11 sin adaptar",
    "individual": "Individual (solo N días)",
    "ft": "Global + ajuste fino",
    "ft_weighted": "Global + ajuste fino ponderado",
}

REGIME_STYLES = {
    "global": {"color": ROLE["referencia"], "marker": "o"},
    "individual": {"color": CATEGORICAL[1], "marker": "s"},
    "ft": {"color": ROLE["modelo"], "marker": "D"},
    "ft_weighted": {"color": CATEGORICAL[2], "marker": "^", "ls": "--"},
}
REFERENCE_STYLES = ("--", ":")


def plot_adaptation_curves(frame, references: dict | None = None, labels: dict | None = None):
    labels = REGIME_LABELS if labels is None else labels
    horizons = sorted(frame["horizon_min"].unique())
    fig, axes = plt.subplots(2, len(horizons), figsize=(FIG_WIDTH_FULL, 4.6),
                             sharex=True, squeeze=False)
    for j, h in enumerate(horizons):
        top, bottom = axes[0, j], axes[1, j]
        for regime, label in labels.items():
            g = frame[(frame["regime"] == regime) & (frame["horizon_min"] == h)]
            if g.empty:
                continue
            g = g.sort_values("n_days")
            yerr = [g["recall"] - g["recall_lo"], g["recall_hi"] - g["recall"]]
            st = REGIME_STYLES.get(regime, {})
            kw = {"color": st.get("color"), "marker": st.get("marker", "o"),
                  "ls": st.get("ls", "-"), "ms": 4, "lw": 1.4, "capsize": 2}
            top.errorbar(g["n_days"], g["recall"], yerr=yerr, label=label, **kw)
            bottom.errorbar(g["n_days"], g["rmse_mean"], yerr=g["rmse_std"], label=label, **kw)
        for style, (name, ref) in zip(REFERENCE_STYLES, (references or {}).items()):
            if "recall" in ref:
                top.axhline(ref["recall"][h], ls=style, color=INK["secundario"], lw=1, label=name)
            if "rmse" in ref:
                bottom.axhline(ref["rmse"][h], ls=style, color=INK["secundario"], lw=1, label=name)
        top.set_title(f"PH {h} min")
        bottom.set_xlabel("Días de adaptación")
        bottom.set_xticks([0, 1, 3, 7])
    axes[0, 0].set_ylabel("Recall por eventos\n(≤ 1 falsa alarma/día)")
    axes[1, 0].set_ylabel("RMSE (mg/dL)")
    handles = {}
    for ax in (axes[0, 0], axes[1, 0]):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    axes[0, 0].legend(handles.values(), handles.keys(), ncol=3,
                      loc="upper center", bbox_to_anchor=(0.5, 0.1),
                      bbox_transform=fig.transFigure, frameon=False)
    return fig


EXAMPLE_TITLES = {
    "median_lead": "Detectado, anticipación mediana",
    "max_lead": "Detectado, anticipación máxima",
    "worst_miss": "No detectado, nadir más bajo",
    "longest_false_alarm": "Falsa alarma más larga",
}


def plot_lead_times(tables_by_h: dict):
    horizons = sorted(tables_by_h)
    fig, axes = plt.subplots(1, len(horizons), figsize=(FIG_WIDTH_FULL, 2.5), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        ev = tables_by_h[h].events
        lead = ev.loc[ev["detected"].astype(bool), "lead_time_min"].astype(float)
        ax.hist(lead, bins=np.arange(-15, h + 20, 5), color=HORIZON_COLORS.get(h, ROLE["modelo"]),
                edgecolor=INK["fondo"], linewidth=1)
        ax.axvline(0, color=INK["tenue"], lw=1, ls="--")
        ax.set_title(f"PH {h} min (n = {len(lead)} detectados)")
        ax.set_xlabel("Anticipación de la alarma (min)")
    axes[0, 0].set_ylabel("Eventos")
    fig.tight_layout()
    return fig


def plot_period_breakdown(tables_by_h: dict):
    fig, (left, right) = plt.subplots(1, 2, figsize=(FIG_WIDTH_FULL, 2.7))
    horizons = sorted(tables_by_h)
    width = 0.8 / len(horizons)
    for i, h in enumerate(horizons):
        t = tables_by_h[h]
        x = np.arange(len(t)) + (i - (len(horizons) - 1) / 2) * width
        yerr = [t["recall"] - t["recall_lo"], t["recall_hi"] - t["recall"]]
        color = HORIZON_COLORS.get(h)
        left.bar(x, t["recall"], width, yerr=yerr, capsize=2, color=color,
                 edgecolor=INK["fondo"], linewidth=1, ecolor=INK["secundario"], label=f"PH {h} min")
        right.bar(x, t["false_alarms_per_day"], width, color=color,
                  edgecolor=INK["fondo"], linewidth=1, label=f"PH {h} min")
        for xi, n in zip(x, t["n_events"]):
            left.annotate(f"n={n}", (xi, 0.02), ha="center", fontsize=7, color=INK["fondo"])
    for ax, ylabel in ((left, "Recall por eventos (IC95 %)"), (right, "Falsas alarmas/día")):
        ax.set_xticks(np.arange(len(tables_by_h[horizons[0]])))
        ax.set_xticklabels([str(i).capitalize() for i in tables_by_h[horizons[0]].index])
        ax.set_ylabel(ylabel)
    left.set_ylim(0, 1.05)
    left.legend()
    fig.tight_layout()
    return fig


def plot_examples(frame, picks: dict, alpha: float, horizon_min: int, hours: float = 3.0):
    from matplotlib.gridspec import GridSpec
    from scipy.stats import norm

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 7.0))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[3, 1.1, 0.9, 3, 1.1], hspace=0.12, wspace=0.28)
    f = frame[frame["horizon_min"] == horizon_min].copy()
    f["target"] = pd.to_datetime(f["ts"]) + pd.Timedelta(minutes=horizon_min)
    legend_handles = None
    for k, (key, title) in enumerate(EXAMPLE_TITLES.items()):
        r, c = 3 * (k // 2), k % 2
        ax = fig.add_subplot(gs[r, c])
        low = fig.add_subplot(gs[r + 1, c], sharex=ax)
        row = picks.get(key)
        if row is None:
            ax.axis("off")
            low.axis("off")
            continue
        center = pd.Timestamp(row["start"])
        if key == "longest_false_alarm":
            center += pd.Timedelta(minutes=horizon_min)
        g = f[(f["patient"].astype(str) == str(row["patient"]))
              & (f["target"] >= center - pd.Timedelta(hours=hours))
              & (f["target"] <= center + pd.Timedelta(hours=hours))].sort_values("target")
        add_glycemic_bands(ax, show_severe=False)
        ax.fill_between(g["target"], g["y_pred"] - 1.96 * g["sigma"],
                        g["y_pred"] + 1.96 * g["sigma"], color=ROLE["modelo_banda"], lw=0,
                        label="Banda del 95 %")
        ax.plot(g["target"], g["y_pred"], color=ROLE["modelo"], lw=1.3, label="μ predicha")
        ax.plot(g["target"], g["y_true"], color=ROLE["observado"], lw=1.1, label="CGM")
        prob = norm.cdf((70 - g["y_pred"]) / g["sigma"])
        low.fill_between(g["target"], 0, prob, where=prob > alpha, color=ROLE["hipo_relleno"],
                         lw=0, step=None, label="Alarma activa")
        low.plot(g["target"], prob, color=ROLE["hipo"], lw=1.1, label="P(y < 70)")
        low.axhline(alpha, color=ROLE["hipo"], ls=":", lw=0.9, label=f"α = {style.fmt_es(alpha, 1)}")
        for a in (ax, low):
            a.axvline(center, color=INK["tenue"], ls=":", lw=0.9)
        low.set_ylim(0, 1)
        low.set_yticks([0, 1])
        low.set_ylabel("P(y<70)", fontsize=7)
        ax.set_ylabel("Glucosa (mg/dL)" if c == 0 else "")
        ax.tick_params(axis="x", labelbottom=False)
        low.tick_params(axis="x", labelrotation=0, labelsize=7)
        low.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        lead = row.get("lead_time_min", np.nan)
        extra = f", anticipación {lead:.0f} min" if pd.notna(lead) else ""
        ax.set_title(f"{title}\npaciente {row['patient']}{extra}", fontsize=8.5)
        if legend_handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = low.get_legend_handles_labels()
            legend_handles = (h1 + h2, l1 + l2)
    if legend_handles is not None:
        fig.legend(*legend_handles, loc="lower center", ncol=6, frameon=False)
    fig.subplots_adjust(bottom=0.09, top=0.93)
    return fig


def _label_alphas(ax, curve, min_gap: float = 0.1) -> None:
    xr = max(curve["false_alarms_per_day"].max(), 1e-9)
    last = None
    for _, r in curve.sort_values("false_alarms_per_day").iterrows():
        pos = (r["false_alarms_per_day"] / xr, r["recall"])
        if last is not None and abs(pos[0] - last[0]) < min_gap and abs(pos[1] - last[1]) < 0.15:
            continue
        ax.annotate(f"{r['alpha']:g}".replace(".", ","), (r["false_alarms_per_day"], r["recall"]),
                    textcoords="offset points", xytext=(5, 4 if r["recall"] < 0.1 else -10),
                    fontsize=7, color=INK["secundario"])
        last = pos


def plot_operating_curve(curves_by_h: dict, chosen_alpha: dict, budget: float = 1.0):
    horizons = sorted(curves_by_h)
    fig, axes = plt.subplots(1, len(horizons), figsize=(FIG_WIDTH_FULL, 2.9), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        c = curves_by_h[h].sort_values("false_alarms_per_day")
        ax.fill_between(c["false_alarms_per_day"], c["recall_lo"], c["recall_hi"],
                        color=ROLE["modelo_banda"], lw=0)
        ax.plot(c["false_alarms_per_day"], c["recall"], "o-", ms=3.5, color=ROLE["modelo"], zorder=3)
        _label_alphas(ax, c)
        pick = c[np.isclose(c["alpha"], chosen_alpha[h])].iloc[0]
        ax.plot([pick["false_alarms_per_day"]], [pick["recall"]], marker="*", ms=12,
                color=ROLE["hipo"], ls="", zorder=4,
                label=f"α elegido en validación ({style.fmt_es(pick['alpha'], 1)})")
        ax.axvline(budget, color=INK["tenue"], ls="--", lw=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"PH {h} min")
        ax.set_xlabel("Falsas alarmas por día (test)")
        ax.legend(loc="lower right")
    axes[0, 0].set_ylabel("Recall por eventos (IC95 %)")
    fig.tight_layout()
    return fig


def plot_operating_curves_compared(curves: dict, alpha_by_h: dict, colors: dict,
                                   budget: float = 1.0):
    horizons = sorted({h for by_h in curves.values() for h in by_h})
    fig, axes = plt.subplots(1, len(horizons), figsize=(FIG_WIDTH_FULL, 2.9), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        for label, by_h in curves.items():
            c = by_h[h].sort_values("false_alarms_per_day")
            color = colors[label]
            if {"recall_lo", "recall_hi"} <= set(c.columns):
                ax.fill_between(c["false_alarms_per_day"], c["recall_lo"], c["recall_hi"],
                                color=color, alpha=0.15, lw=0)
            ax.plot(c["false_alarms_per_day"], c["recall"], "o-", ms=3.5, color=color,
                    label=label, zorder=3)
            pick = c[np.isclose(c["alpha"], alpha_by_h[h])]
            if len(pick):
                ax.plot(pick["false_alarms_per_day"], pick["recall"], marker="*", ms=12,
                        color=ROLE["hipo"], mec=INK["fondo"], mew=0.6, ls="", zorder=4)
        ax.axvline(budget, color=INK["tenue"], ls="--", lw=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"PH {h} min")
        ax.set_xlabel("Falsas alarmas por día")
        ax.legend(loc="lower right")
    axes[0, 0].set_ylabel("Recall por eventos")
    fig.tight_layout()
    return fig
