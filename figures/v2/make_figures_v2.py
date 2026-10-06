"""Figures of the memoir, version 2 (6 Oct 2026): Chapters 3 and 4.

Reads ONLY the CSVs written by the extraction pipeline:
    ../../extraction/out/{clips,impacts}.csv, tracks/*.csv       (September 2026)
    ../../extraction_old/out/{clips,impacts}.csv                (March 2026)
and applies the estimator policy of ../../extraction/summarise_v2.primary() (imported,
not copied), so every point drawn is a value the summaries count.

    PY=/Users/User/Desktop/2026/20260200/20260209_toll_4_TFG_TFM-PhD/20260209_tfg/20260211_aingeru_palacios/20260310_datuak_tratatzeko/.venv/bin/python
    $PY make_figures_v2.py            # writes fN_*.pdf (LaTeX) and fN_*.svg (HTML) here

ThesisFigures rules followed: real LaTeX (usetex), one canvas width for the series,
one colour + marker per ball/surface group in every figure (the registry of the v1
pgfplots figures, ../common.tex), campaign and impact type on separate channels,
shared windows for every e axis and every speed axis, decimal comma in tick labels.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extraction"))
from summarise_v2 import primary, G  # noqa: E402  (one source of the selection rules)

NEW, OLD = ROOT / "extraction" / "out", ROOT / "extraction_old" / "out"
EXAMPLE = "tenis_zorua_h1.75_2"           # bounce train shown in Chapter 3

# ---------------------------------------------------------------- style (E1)
BASE = 19                                  # canvas 12 in wide, printed at ~6.3 in
CANVAS_W = 12.0
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": BASE, "axes.labelsize": BASE + 1, "axes.titlesize": BASE + 1,
    "xtick.labelsize": BASE - 2, "ytick.labelsize": BASE - 2, "legend.fontsize": BASE - 4,
    "axes.grid": True, "grid.alpha": 0.25, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "path",
})

# C1/C7b registry: group -> colour and marker, identical to the v1 figures (../common.tex)
GROUPS = [("golf", "zorua"), ("tenis", "egurra"), ("tenis", "zorua"), ("golf", "egurra")]
COLOR = {("golf", "zorua"): "#1F5673", ("tenis", "egurra"): "#C1553B",
         ("tenis", "zorua"): "#4E8098", ("golf", "egurra"): "#D9A441"}
MARKER = {("golf", "zorua"): "o", ("tenis", "egurra"): "s",
          ("tenis", "zorua"): "^", ("golf", "egurra"): "D"}
NAME = {g: rf"{g[0]} / \textit{{{g[1]}}}" for g in GROUPS}
FIT_COLOR = "0.15"
# channels: campaign = fill (September filled, March open); impact type = size + edge
#           (first impact large with dark edge, later bounce small and translucent)

XLIM_V = (1.0, 6.2)                        # W1: every impact-speed axis
YLIM_E = (0.66, 0.94)                      # W1: every e axis


def comma(dec):
    """T4: math-mode tick labels with a decimal comma."""
    return FuncFormatter(lambda x, _: "$" + f"{x:.{dec}f}".replace(".", "{,}").replace("-", "-") + "$")


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", stem)


def load(folder):
    clips, imp = pd.read_csv(folder / "clips.csv"), pd.read_csv(folder / "impacts.csv")
    for col in ("e_time", "e_audio", "v_in_time_mps"):
        if col not in imp:
            imp[col] = np.nan
    return clips, imp, primary(clips, imp)


cn, imn, pn = load(NEW)
co, imo, po = load(OLD)


def grp(df, g):
    return df[(df.ball == g[0]) & (df.surface == g[1])]


def scatter(ax, x, y, g, *, first, filled=True, **kw):
    if first:
        ax.plot(x, y, MARKER[g], ms=7.5, mec="0.1" if filled else COLOR[g], mew=0.9 if filled else 1.4,
                mfc=COLOR[g] if filled else "white", ls="none", zorder=4, **kw)
    else:
        ax.plot(x, y, MARKER[g], ms=4.5, mec="none", mfc=COLOR[g], alpha=0.45, ls="none", zorder=3, **kw)


# ------------------------------------------------------------- f6: bounce train (Ch. 3)
def fig_train():
    tr = pd.read_csv(NEW / "tracks" / f"{EXAMPLE}.csv")
    clip = cn[cn.video_file == f"{EXAMPLE}.mp4"].iloc[0]
    imp = imn[imn.video_file == f"{EXAMPLE}.mp4"].sort_values("impact")
    s = clip.scale_floor_px_per_m                      # px/m at the floor, from timing (no ruler)
    y0 = imp.y_contact_px.iloc[0]
    h = (y0 - tr.y_px) / s                             # height of the centroid above the first contact
    t0 = clip.t_release_s
    fig, ax = plt.subplots(figsize=(CANVAS_W, 5.0))
    ax.plot(tr.t_s - t0, h, ".", ms=2.2, color="0.45", zorder=2)
    g = ("tenis", "zorua")
    ti = imp.t_impact_s.to_numpy() - t0
    ax.plot([0], [(y0 - tr.y_px[(tr.t_s - t0).abs().idxmin()]) / s], "v", ms=10, color="0.1", zorder=5)
    ax.plot(ti, np.zeros_like(ti), MARKER[g], ms=9, mfc=COLOR[g], mec="0.1", mew=0.9, zorder=5, clip_on=False)
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.annotate(r"$t_{\mathrm{r}}$", (0, 1.72), xytext=(0.05, 1.82), fontsize=BASE - 1)
    ax.annotate(r"$t_{\text{caída}}$", xy=(ti[0] / 2, -0.12), ha="center", fontsize=BASE - 2)
    ax.annotate("", xy=(0, -0.06), xytext=(ti[0], -0.06),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.2"))
    for k in range(len(ti) - 1):
        mid = 0.5 * (ti[k] + ti[k + 1])
        ax.annotate("", xy=(ti[k], -0.06), xytext=(ti[k + 1], -0.06),
                    arrowprops=dict(arrowstyle="<->", lw=0.9, color=COLOR[g]))
        if k < 4:
            ax.text(mid, -0.12, rf"$T_{{{k + 1}}}$", ha="center", va="top", fontsize=BASE - 2, color="0.1")
    ax.set_xlim(-0.15, ti[-1] + 0.25)
    ax.set_ylim(-0.30, 1.95)
    ax.set_xlabel(r"tiempo desde la suelta, $t - t_{\mathrm{r}}$ (s)")
    ax.set_ylabel(r"altura sobre el contacto (m)")
    ax.xaxis.set_major_formatter(comma(1)); ax.yaxis.set_major_formatter(comma(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    save(fig, "f6_tren")


# ------------------------------------------------------------- f7: heights from fall time
def fig_heights():
    fig, ax = plt.subplots(figsize=(CANVAS_W, 4.6))
    for clips, filled, off in ((cn, True, 0.0), (co, False, 0.0)):
        ok = clips[(clips.status == "ok") & clips.h_drop_tt_m.notna()]
        ok = ok[~ok.notes.fillna("").str.contains("LABEL")]      # the mislabelled 1.03 m clip
        for j, g in enumerate(GROUPS):
            s = grp(ok, g)
            dx = (j - 1.5) * 0.022
            rel = 100 * (s.h_drop_tt_m / s.h_nom_m - 1)
            ax.plot(s.h_nom_m + dx, rel, MARKER[g], ms=6.5, mfc=COLOR[g] if filled else "white",
                    mec=COLOR[g] if not filled else "0.1", mew=1.2 if not filled else 0.6, ls="none",
                    alpha=0.9, zorder=3)
    ax.axhline(0, color="0.2", lw=0.9)
    ax.set_xlim(0.1, 1.9); ax.set_ylim(-12, 12)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    ax.xaxis.set_major_formatter(comma(2)); ax.yaxis.set_major_formatter(comma(0))
    ax.set_xlabel(r"altura nominal de caída $h_{\mathrm{nom}}$ (m)")
    ax.set_ylabel(r"$g t_{\text{caída}}^{2}/2h_{\mathrm{nom}} - 1$ (\%)")
    handles = [Line2D([], [], marker=MARKER[g], color=COLOR[g], mec="0.1", mew=0.6, ls="none", ms=7,
                      label=NAME[g]) for g in GROUPS]
    handles += [Line2D([], [], marker="o", mfc="0.5", mec="0.1", ls="none", ms=7, label="septiembre 2026"),
                Line2D([], [], marker="o", mfc="white", mec="0.5", mew=1.2, ls="none", ms=7, label="marzo 2026")]
    ax.legend(handles=handles, loc="upper right", ncol=3, frameon=True, framealpha=0.95)
    save(fig, "f7_alturas")


# ------------------------------------------------------------- f8: cross-checks
def fig_crosschecks():
    ok = cn[cn.status == "ok"]
    g2 = imn[(imn.flag == "ok") & imn.video_file.isin(ok.video_file) & (imn.impact >= 2)]
    panels = [("e_vel", r"$e_{\mathrm{vel}} - e_{\mathrm{time}}$", "(a) velocidades, deriva corregida"),
              ("e_audio", r"$e_{\mathrm{audio}} - e_{\mathrm{time}}$", r"(b) audio"),
              ("e_tt", r"$e_{\mathrm{tt}} - e_{\mathrm{time}}$", r"(c) subida/caída"),
              ("e_apex", r"$e_{\text{ápice}} - e_{\mathrm{time}}$", r"(d) alturas de ápice")]
    fig, axs = plt.subplots(2, 2, figsize=(CANVAS_W, 8.0), sharex=True, sharey=True)
    for ax, (col, lab, title) in zip(axs.flat, panels):
        for g in GROUPS:
            s = grp(g2, g)
            ax.plot(s.v_in_time_mps, s[col] - s.e_time, MARKER[g], ms=5, mfc=COLOR[g], mec="none",
                    alpha=0.7, ls="none")
        d = (g2[col] - g2.e_time).dropna()
        med = d.median(); mad = (d - med).abs().median()
        ax.axhline(0, color="0.2", lw=0.9)
        ax.axhline(med, color="0.2", lw=0.9, ls="--")
        ax.set_title(title, fontsize=BASE - 1)
        ax.text(0.97, 0.05, rf"mediana ${med:+.3f}$, MAD ${mad:.3f}$, $n={len(d)}$".replace(".", "{,}"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=BASE - 4)
        ax.set_ylabel(lab)
    for ax in axs[1]:
        ax.set_xlabel(r"velocidad de impacto $v = gT/2$ (m/s)")
    axs[0, 0].set_xlim(XLIM_V); axs[0, 0].set_ylim(-0.08, 0.08)
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(comma(0)); ax.yaxis.set_major_formatter(comma(2))
    handles = [Line2D([], [], marker=MARKER[g], color=COLOR[g], mec="none", ls="none", ms=7, label=NAME[g])
               for g in GROUPS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "f8_contrastes")


# ------------------------------------------------------------- f9: e(v), September
def fig_e_vs_v():
    fig, axs = plt.subplots(1, 2, figsize=(CANVAS_W, 5.4), sharey=True)
    vv = np.linspace(*XLIM_V, 50)
    for ax, ball in zip(axs, ("golf", "tenis")):
        for g in [g for g in GROUPS if g[0] == ball]:
            s = grp(pn, g)
            scatter(ax, s[s.impact >= 2].v, s[s.impact >= 2].e, g, first=False)
            scatter(ax, s[s.impact == 1].v, s[s.impact == 1].e, g, first=True)
            p = np.polyfit(s.v, s.e, 1)
            ax.plot(vv, np.polyval(p, vv), "-", color=COLOR[g], lw=1.6, zorder=5)
        ax.set_title(r"golf" if ball == "golf" else r"tenis")
        ax.set_xlim(XLIM_V); ax.set_ylim(YLIM_E)
        ax.set_xlabel(r"velocidad de impacto $v$ (m/s)")
        ax.xaxis.set_major_formatter(comma(0)); ax.yaxis.set_major_formatter(comma(2))
    axs[0].set_ylabel(r"coeficiente de restitución $e$")
    hs = [Line2D([], [], marker=MARKER[g], color=COLOR[g], mfc=COLOR[g], mec="0.1", ms=7.5, lw=1.6,
                 label=NAME[g]) for g in GROUPS]
    hs += [Line2D([], [], marker="o", mfc="0.5", mec="0.1", ms=7.5, ls="none", label="primer impacto ($e_{\\mathrm{tt}}$)"),
           Line2D([], [], marker="o", mfc="0.5", mec="none", alpha=0.5, ms=4.5, ls="none",
                  label="botes siguientes ($e_{\\mathrm{time}}$)")]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), frameon=False,
               handlelength=1.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "f9_e_v")


# ------------------------------------------------------------- f10: March vs September
def fig_campaigns():
    fig, axs = plt.subplots(1, 2, figsize=(CANVAS_W, 5.4), sharey=False,
                            gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    vv = np.linspace(*XLIM_V, 50)
    ax = axs[0]                                       # tennis + golf/tile: both campaigns, one line
    for g in GROUPS:
        sn, so = grp(pn, g), grp(po, g)
        p = np.polyfit(sn.v, sn.e, 1)
        ax.plot(vv, np.polyval(p, vv), "-", color=COLOR[g], lw=1.4, zorder=2)
        scatter(ax, so.v, so.e, g, first=True, filled=False)
        scatter(ax, sn[sn.impact == 1].v, sn[sn.impact == 1].e, g, first=True, filled=True, alpha=0.55)
    ax.set_xlim(XLIM_V); ax.set_ylim(YLIM_E)
    ax.set_xlabel(r"velocidad de impacto $v = g t_{\text{caída}}$ (m/s)")
    ax.set_ylabel(r"$e_{\mathrm{tt}}$ (primer impacto)")
    ax.set_title(r"(a) primeros impactos")
    ax = axs[1]                                       # residual of each March drop from the September line
    for j, g in enumerate(GROUPS):
        sn, so = grp(pn, g), grp(po, g)
        p = np.polyfit(sn.v, sn.e, 1)
        r = so.e - np.polyval(p, so.v)
        x = np.full(len(r), j) + np.linspace(-0.18, 0.18, len(r))
        ax.plot(x, r, MARKER[g], ms=7.5, mfc="white", mec=COLOR[g], mew=1.4, ls="none", zorder=3)
        ax.plot([j - 0.3, j + 0.3], [r.median()] * 2, "-", color="0.1", lw=1.6, zorder=4)
    ax.axhline(0, color="0.2", lw=0.9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([rf"{g[0]}" + "\n" + rf"\textit{{{g[1]}}}" for g in GROUPS])
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.10, 0.04)
    ax.set_ylabel(r"marzo $-$ recta de septiembre")
    ax.set_title(r"(b) a igual velocidad")
    ax.grid(axis="x", visible=False)
    for a, fx, fy in ((axs[0], comma(0), comma(2)), (axs[1], None, comma(2))):
        if fx: a.xaxis.set_major_formatter(fx)
        a.yaxis.set_major_formatter(fy)
    handles = [Line2D([], [], marker="o", mfc="0.5", mec="0.1", ls="none", ms=7, label="septiembre 2026"),
               Line2D([], [], marker="o", mfc="white", mec="0.4", mew=1.4, ls="none", ms=7, label="marzo 2026")]
    axs[0].legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95)
    fig.tight_layout(w_pad=1.2)
    save(fig, "f10_campanas")


if __name__ == "__main__":
    fig_train()
    fig_heights()
    fig_crosschecks()
    fig_e_vs_v()
    fig_campaigns()
