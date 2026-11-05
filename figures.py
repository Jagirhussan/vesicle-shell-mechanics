#!/usr/bin/env python3
"""figures.py — generate manuscript figures/panels from canonical data.

Creates the figure panels as individual vector PDFs, one file per
subplot, with the panel letter carried by the filename.

Inputs required are ``data/`` (canonical tables and the representative saved
states) and ``results/`` (outputs of analysis.py).  No simulation is started;
the spatial panels are deterministic post-processing of saved relaxed states.
Every plotted value comes from the CSV/NPZ inputs.

Panel provenance:
  fig01 a,b   data/representative_states.npz  (saved replicate-0 maps)
  fig02 a-f   representative_states.npz at eps=0.02 (labels, Kabsch-aligned
              tangential displacement, signed bond strain; shared colour
              limits spanning the real minima/maxima of both maps)
  fig02 g,h,i results/fig2_class_means.csv, results/fig2_contrasts.csv
  fig03       results/fig3_contrasts.csv, results/fig3_affine_classes.csv
  fig04       results/fig4_parents.csv, fig4_children.csv, fig4_contrasts.csv
  fig05       data/density_controls.csv, results/fig5_contrasts.csv
  figS01-S03  results/figS0*.csv

Usage:  python figures.py --data data --results results --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

import model

BLUE = "#0072B2"      # Colourblind-safe palette
ORANGE = "#D55E00"
COLORS = [BLUE, ORANGE]

plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
})

EPS_MAIN = 0.02
AFFINE_STRAIN_LINE = (np.sqrt(1.02) - 1) * 100
CLASS_ORDER = ["soft-soft", "interface", "stiff-stiff"]
CLASS_LABELS = ["Soft-soft", "Interface", "Stiff-stiff"]

TITLE_FONTSIZE = 16
XLABEL_FONTSIZE = 14
YLABEL_FONTSIZE = 14


# ------------------------------------------------------------- panel helpers --
def save_panel(fig, out: Path, stem: str, letter: str) -> None:
    """Save one single-axes figure as ``<figNN><letter>_<name>.pdf`` — the
    panel letter lives in the filename, not in the title."""
    family, name = stem.split("_", 1)
    fig.savefig(out / f"{family}{letter}_{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def hemisphere(ax, xyz, pairs=None, values=None, cmap=None, norm=None,
               scatter_c=None, s=4):
    """Front-hemisphere (z>0) view: nodes coloured by ``values``/``scatter_c``,
    or edges coloured by ``values`` when ``pairs`` is given."""
    visible = xyz[:, 2] > 0
    if values is not None and pairs is not None:
        mask = visible[pairs].all(1)
        seg = xyz[pairs[mask], :2]
        lc = LineCollection(seg, array=values[mask], cmap=cmap, norm=norm,
                            linewidth=0.55)
        ax.add_collection(lc)
        obj = lc
    else:
        c = scatter_c[visible] if scatter_c is not None else (
            values[visible] if values is not None else None)
        obj = ax.scatter(xyz[visible, 0], xyz[visible, 1], c=c, s=s,
                         cmap=cmap, norm=norm, linewidths=0)
    ax.set(xlim=(-21, 21), ylim=(-21, 21), aspect="equal")
    ax.axis("off")
    return obj


def interval(ax, group, x, offset=0.0, color=BLUE, metric="estimate"):
    v = group[metric].to_numpy()
    lo = group.ci_lo.to_numpy()
    hi = group.ci_hi.to_numpy()
    ax.errorbar(np.asarray(x) + offset, v, yerr=[v - lo, hi - v],
                fmt="o" if color == BLUE else "s", color=color,
                capsize=3, ms=4, lw=1)


def load_states(data: Path):
    """The two saved replicate-0 maps with their eps=0.02 relaxed states and
    every derived spatial field used by fig01/fig02."""
    z = np.load(data / "representative_states.npz")
    states = []
    for tag, title in (("mapA", "Clustered"), ("mapB", "Random")):
        net = model.network_from_arrays(
            z[f"{tag}_xyz0"], z[f"{tag}_pairs"], z[f"{tag}_l0"], None,
            z[f"{tag}_k_harmonic"], z[f"{tag}_bond_class"])
        x_rel = z[f"{tag}_x_rel_eps0.020"]
        x_aff = model.affine_positions(z[f"{tag}_xyz0"], EPS_MAIN)
        _, tangent = model.tangential_displacement(x_rel, x_aff)
        u_mag = np.linalg.norm(tangent, axis=1)
        strain, _l = net.extensions(x_rel)
        strain = strain / net.l0
        states.append({"tag": tag, "title": title, "xyz0": z[f"{tag}_xyz0"],
                       "pairs": net.pairs, "labels": z[f"{tag}_phase_labels"],
                       "u_tangent": u_mag, "strain": strain,
                       "map_id": str(z[f"{tag}_id"][0])})
    return states


def node_legend(ax):
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=l)
                       for c, l in [(BLUE, "Stiff node"), (ORANGE, "Soft node")]],
              frameon=False, fontsize=7, loc="lower right",
              handletextpad=0.1, borderaxespad=0.1)


# ------------------------------------------------------------------- figures --
def fig01(states, out: Path):
    """(a) clustered labels (b) random labels (c) defined mechanical test."""
    for letter, j in (("a", 0), ("b", 1)):
        fig, ax = plt.subplots(figsize=(3.2, 3.4), layout="constrained")
        hemisphere(ax, states[j]["xyz0"], scatter_c=states[j]["labels"],
                   cmap=ListedColormap([ORANGE, BLUE]))
        ax.set_title(f"{states[j]['title']} labels", loc="left",
                     fontsize=TITLE_FONTSIZE)
        ax.plot([-19, -9], [-19, -19], color="k", lw=2)
        ax.text(-14, -22, "10 nm", ha="center", fontsize=8)
        node_legend(ax)
        save_panel(fig, out, "fig01_model", letter)
    # schematic
    fig, ax = plt.subplots(figsize=(4.4, 3.1), layout="constrained")
    ax.axis("off")
    ax.set_title("Defined mechanical test", loc="left", fontsize=TITLE_FONTSIZE)
    boxes = [(0.86, "Plant EV lipid heterogeneity\nMotivation; no measured domain map"),
             (0.51, "Quenched stiff / soft graph\nAffine dilation: radius $\\times \\sqrt{(1 + εs)}$"),
             (0.15, "Tangential relaxation on target sphere\nCompare energy curvature and bond strain")]
    for y, text in boxes:
        ax.text(0.5, y, text, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.45", fc="#F2F5F7", ec="#9BA7AF"),
                fontsize=8)
    for y in (0.72, 0.37):
        ax.annotate("", (0.5, y - 0.08), (0.5, y), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->"))
    save_panel(fig, out, "fig01_model", "c")


def fig02(states, results: Path, out: Path, audit: list):
    """a-f spatial panels; g,h class dot plots; i global competition."""
    cm = pd.read_csv(results / "fig2_class_means.csv")
    cts = pd.read_csv(results / "fig2_contrasts.csv")

    # shared, *signed* colour limits spanning the real minima/maxima of both
    # saved maps (no clipping of compressive bonds)
    u_hi = max(float(s["u_tangent"].max()) for s in states)
    s_lo = min(float(s["strain"].min()) for s in states)
    s_hi = max(float(s["strain"].max()) for s in states)
    audit.append(f"fig02 signed strain limits: [{s_lo * 100:.3f}, {s_hi * 100:.3f}] %"
                 f" (compressive bonds shown, not clipped)")
    unorm = Normalize(0, u_hi)
    snorm = Normalize(s_lo, s_hi)
    col_titles = ["Labels", "Tangential motion", "Signed bond strain"]

    letter = "a"
    for r, st in enumerate(states):
        # labels panel
        fig, ax = plt.subplots(figsize=(3.3, 3.4), layout="constrained")
        hemisphere(ax, st["xyz0"], scatter_c=st["labels"],
                   cmap=ListedColormap([ORANGE, BLUE]))
        ax.set_title(col_titles[0], loc="left", fontsize=TITLE_FONTSIZE)
        ax.text(-0.12, 0.5, st["title"], rotation=90, va="center",
                fontsize=TITLE_FONTSIZE, transform=ax.transAxes)
        node_legend(ax)
        save_panel(fig, out, "fig02_local", letter)
        letter = chr(ord(letter) + 1)
        # tangential motion panel
        fig, ax = plt.subplots(figsize=(4.0, 3.4), layout="constrained")
        obj = hemisphere(ax, st["xyz0"], values=st["u_tangent"],
                         cmap="viridis", norm=unorm)
        ax.set_title(col_titles[1], loc="left", fontsize=TITLE_FONTSIZE)
        cbar = fig.colorbar(obj, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("nm", fontsize=YLABEL_FONTSIZE)
        save_panel(fig, out, "fig02_local", letter)
        letter = chr(ord(letter) + 1)
        # signed bond strain panel
        fig, ax = plt.subplots(figsize=(4.0, 3.4), layout="constrained")
        obj = hemisphere(ax, st["xyz0"], pairs=st["pairs"], values=st["strain"],
                         cmap="coolwarm", norm=snorm)
        ax.set_title(col_titles[2], loc="left", fontsize=TITLE_FONTSIZE)
        cbar = fig.colorbar(obj, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Δℓ / ℓ₀", fontsize=YLABEL_FONTSIZE)
        save_panel(fig, out, "fig02_local", letter)
        letter = chr(ord(letter) + 1)

    for letter, rule in (("g", "harmonic"), ("h", "arithmetic")):
        fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
        g = cm[cm.interface_rule == rule]
        for p, pat in enumerate(["SEEDED", "SPONTANEOUS"]):
            for i, cl in enumerate(CLASS_ORDER):
                y = g[(g.pattern == pat) & (g.bond_class == cl)].strain_mean.to_numpy() * 100
                ax.scatter(i + (p - 0.5) * 0.18 + np.linspace(-0.035, 0.035, len(y)),
                           y, s=10, color=COLORS[p], alpha=0.7)
                ax.plot([i + (p - 0.5) * 0.18 - 0.07, i + (p - 0.5) * 0.18 + 0.07],
                        [y.mean()] * 2, color="k", lw=1.5)
        ax.axhline(AFFINE_STRAIN_LINE, color="gray", ls="--", lw=0.8)
        ax.set_xticks(range(3), CLASS_LABELS, rotation=25)
        ax.set_ylabel("Map mean bond strain (%)", fontsize=YLABEL_FONTSIZE)
        ax.set_title(f"{rule.capitalize()}; n = 12/group", loc="left",
                     fontsize=TITLE_FONTSIZE)
        ax.set_ylim(0.45, 1.6)
        if rule == "harmonic":
            ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=l)
                               for c, l in zip(COLORS, ["Clustered", "Random"])],
                      frameon=False, fontsize=6, loc="upper right")
        save_panel(fig, out, "fig02_local", letter)

    fig, ax = plt.subplots(figsize=(3.5, 3.3), layout="constrained")
    g = cts[(cts.study == "stage2_main") & (cts.stiffness_set == "B")]
    for j, rule in enumerate(("harmonic", "arithmetic")):
        s = g[g.rule == rule].set_index("metric").loc[["K_aff", "L", "K_rel"]]
        interval(ax, s, [0, 1, 2], (j - 0.5) * 0.15, COLORS[j])
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks([0, 1, 2], ["ΔKaff", "ΔL", "ΔKrel"])
    ax.set_ylabel("Contrast (mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
    ax.set_title("Global competition; F2", loc="left", fontsize=TITLE_FONTSIZE)
    ax.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=c, label=l)
                       for m, c, l in zip(["o", "s"], COLORS,
                                          ["Harmonic", "Arithmetic"])],
              frameon=False, fontsize=6, loc="lower left")
    save_panel(fig, out, "fig02_local", "i")


def fig03(results: Path, out: Path):
    """a-c factorial contrasts; d exact bond-class affine accounting."""
    cts = pd.read_csv(results / "fig3_contrasts.csv")
    aff = pd.read_csv(results / "fig3_affine_classes.csv")
    base = cts[cts.study == "stage2_main"]
    order = [(0.15, "A"), (0.15, "B"), (0.5, "A"), (0.5, "B")]
    for letter, metric, title in (
            ("a", "K_aff", "Affine inventory"),
            ("b", "L", "Operational relaxation loss"),
            ("c", "K_rel", "Relaxed curvature (F2)")):
        fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
        for j, rule in enumerate(("harmonic", "arithmetic")):
            g = base[(base.rule == rule) & (base.metric == metric)].set_index(
                ["phi", "stiffness_set"]).loc[order]
            interval(ax, g, np.arange(4), (j - 0.5) * 0.16, COLORS[j])
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xticks(range(4), ["0.15 / A", "0.15 / B", "0.50 / A", "0.50 / B"])
        ax.set_xlabel("Stiff-node fraction / stiffness set", fontsize=XLABEL_FONTSIZE)
        ax.set_ylabel("Contrast (mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        save_panel(fig, out, "fig03_factorial", letter)
    fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
    for j, rule in enumerate(("harmonic", "arithmetic")):
        g = aff[aff.interface_rule == rule].set_index("pattern")[
            ["K_aff_soft_mNm", "K_aff_interface_mNm", "K_aff_stiff_mNm"]]
        vals = (g.loc["SEEDED"] - g.loc["SPONTANEOUS"])[
            ["K_aff_soft_mNm", "K_aff_interface_mNm", "K_aff_stiff_mNm"]]
        ax.bar(np.arange(3) + (j - 0.5) * 0.32, vals, width=0.30,
               color=COLORS[j], label=rule.capitalize())
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(range(3), CLASS_LABELS)
    ax.set_ylabel("Affine contribution (mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
    ax.set_title("Exact bond-class accounting", fontsize=TITLE_FONTSIZE)
    ax.legend(frameon=False, fontsize=8)
    save_panel(fig, out, "fig03_factorial", "d")


def fig04(results: Path, out: Path):
    """a,b parent-level shuffle effects; c parent-level contrasts."""
    pars = pd.read_csv(results / "fig4_parents.csv")
    child = pd.read_csv(results / "fig4_children.csv")
    sh = pd.read_csv(results / "fig4_contrasts.csv")
    sh["state"] = sh["state"].fillna("null")  # pandas reads the literal "null" as NaN
    for letter, rule in (("a", "harmonic"), ("b", "arithmetic")):
        fig, ax = plt.subplots(figsize=(3.4, 3.3), layout="constrained")
        g = pars[pars.interface_rule == rule]
        for pi, pat in enumerate(("SEEDED", "SPONTANEOUS")):
            for r, row in enumerate(
                    g[g.pattern == pat].sort_values("replicate").itertuples()):
                xx = pi + (r - 1) * 0.13
                ax.scatter(xx, row.L_excess_mNm, marker=["o", "s", "^"][r],
                           color=COLORS[pi], s=35)
                ch = child[(child.parent_assignment_id == row.parent_assignment_id)
                           & (child.interface_rule == rule)]
                ax.scatter(np.full(len(ch), xx),
                           row.L_mNm - ch.L_child_mNm,
                           color=COLORS[pi], s=8, alpha=0.35)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xticks([0, 1], ["Clustered", "Random"])
        ax.set_ylabel("Lparent − Lshuffle (mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
        ax.set_title(rule.capitalize(), fontsize=TITLE_FONTSIZE)
        ax.text(0.04, 0.60, "3 parents/group\nSmall dots: 3 children/parent",
                transform=ax.transAxes, va="top", fontsize=7)
        save_panel(fig, out, "fig04_shuffle", letter)
    fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
    for j, rule in enumerate(("harmonic", "arithmetic")):
        g = sh[sh.rule == rule].set_index("state").loc[["original", "null", "excess"]]
        interval(ax, g, np.arange(3), (j - 0.5) * 0.15, COLORS[j])
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(range(3), ["Original", "Null", "Difference"], rotation=25)
    ax.set_title("Parent-level contrasts", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Relaxation-loss contrast\n(mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
    ax.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=c, label=l)
                       for m, c, l in zip(["o", "s"], COLORS,
                                          ["Harmonic", "Arithmetic"])],
              frameon=False, fontsize=6, loc="upper right")
    save_panel(fig, out, "fig04_shuffle", "c")


def fig05(results: Path, out: Path):
    """a connectivity; b clustered morphology; c,d F2 relaxed contrasts."""
    dc = pd.read_csv(data_dir / "density_controls.csv")
    cts = pd.read_csv(results / "fig5_contrasts.csv")
    rhos = [0.35, 0.5, 0.75]
    fig, ax = plt.subplots(figsize=(3.5, 3.1), layout="constrained")
    for j, construction in enumerate(("fixed_cutoff", "coordination_matched")):
        g = dc[dc.construction == construction]
        base = dc[(dc.construction == "balanced_baseline")
                  & (dc.interface_rule == "harmonic")]
        g = pd.concat([g, base])
        ax.plot(rhos, g.groupby("rho_nodes_per_nm2").mean_degree.mean(),
                marker=["o", "s"][j], color=COLORS[j],
                label=["Fixed cutoff", "Coordination matched"][j])
    ax.set_xticks(rhos)
    ax.set_xlabel("Node density (nm⁻²)", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("Mean graph degree", fontsize=YLABEL_FONTSIZE)
    ax.set_title("Connectivity", fontsize=TITLE_FONTSIZE)
    ax.legend(frameon=False, fontsize=8)
    save_panel(fig, out, "fig05_density", "a")
    fig, ax = plt.subplots(figsize=(3.5, 3.1), layout="constrained")
    for j, construction in enumerate(("fixed_cutoff", "coordination_matched")):
        g = dc[dc.construction == construction]
        base = dc[(dc.construction == "balanced_baseline")
                  & (dc.interface_rule == "harmonic")]
        g = pd.concat([g, base])
        ax.plot(rhos, g[g.pattern == "SEEDED"].groupby(
            "rho_nodes_per_nm2").domain_mean_r_eq_nm.mean(),
            marker=["o", "s"][j], color=COLORS[j])
    ax.set_xticks(rhos)
    ax.set_xlabel("Node density (nm⁻²)", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("Mean domain radius (nm)", fontsize=YLABEL_FONTSIZE)
    ax.set_title("Clustered morphology", fontsize=TITLE_FONTSIZE)
    save_panel(fig, out, "fig05_density", "b")
    for letter, k, rule in (("c", 0, "harmonic"), ("d", 1, "arithmetic")):
        fig, ax = plt.subplots(figsize=(3.5, 3.1), layout="constrained")
        for j, construction in enumerate(("fixed_cutoff", "coordination_matched")):
            s = cts[((cts.construction == construction)
                     | (cts.construction == "balanced_baseline"))
                    & (cts.rule == rule) & (cts.metric == "K_rel")].sort_values("rho")
            interval(ax, s, s.rho.to_numpy(), (j - 0.5) * 0.006, COLORS[j])
        ax.set_xticks(rhos)
        ax.set_xlabel("Node density (nm⁻²)", fontsize=XLABEL_FONTSIZE)
        ax.set_ylabel("ΔKrel (mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
        ax.set_title(f"{rule.capitalize()}; F2, n = 4/group",
                     fontsize=TITLE_FONTSIZE)
        ax.axhline(0, color="gray", lw=0.7)
        save_panel(fig, out, "fig05_density", letter)


def figS01(results: Path, out: Path):
    """Estimator dependence of the primary curvature contrast (per rule)."""
    fits = pd.read_csv(results / "figS01_fits.csv")
    for letter, rule in (("a", "harmonic"), ("b", "arithmetic")):
        fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
        for k, m in enumerate(("F1", "F2", "F3")):
            g = fits[(fits.interface_rule == rule) & (fits.model == m)]
            est = []
            for w in (0.01, 0.02, 0.04):
                a = g[np.isclose(g.window, w) & (g.pattern == "SEEDED")].K_mNm.to_numpy()
                b = g[np.isclose(g.window, w) & (g.pattern == "SPONTANEOUS")].K_mNm.to_numpy()
                est.append(a.mean() - b.mean())
            ax.plot([0.01, 0.02, 0.04], est, marker=["o", "s", "^"][k], label=m)
        ax.set_title(rule.capitalize(), fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Maximum nominal strain", fontsize=XLABEL_FONTSIZE)
        ax.set_ylabel("ΔKrel (mN m⁻¹)", fontsize=YLABEL_FONTSIZE)
        ax.set_xticks([0.01, 0.02, 0.04])
        ax.legend(frameon=False)
        save_panel(fig, out, "figS01_fits", letter)


def figS02(results: Path, out: Path):
    """Local-strain sensitivity across stiff fractions and interface rules."""
    lc = pd.read_csv(results / "figS02_local_contrasts.csv")
    letter = "a"
    for phi in (0.15, 0.5):
        for rule in ("harmonic", "arithmetic"):
            fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
            for j, eps in enumerate((0.02, 0.04)):
                g = lc[(np.isclose(lc.phi, phi)) & (lc.rule == rule)
                       & np.isclose(lc.epsilon_s, eps)
                       & (lc.metric == "strain_mean")].set_index("bond_class")
                g = g.loc[CLASS_ORDER]
                for arr in ("estimate", "ci_lo", "ci_hi"):
                    g[arr] = g[arr] * 100
                interval(ax, g, np.arange(3), (j - 0.5) * 0.15, COLORS[j])
            ax.axhline(0, color="gray", lw=0.7)
            ax.set_xticks(range(3), CLASS_LABELS)
            ax.set_title(f"φ = {phi:.2f}; {rule}", fontsize=TITLE_FONTSIZE)
            ax.set_ylabel("Strain contrast (percentage points)",
                          fontsize=YLABEL_FONTSIZE)
            ax.legend(handles=[plt.Line2D([], [], marker=m, color=c_, label=l)
                               for m, c_, l in [("o", BLUE, "εs = 0.02"),
                                                ("s", ORANGE, "εs = 0.04")]],
                      frameon=False, fontsize=6, loc="upper right")
            save_panel(fig, out, "figS02_local_sensitivity", letter)
            letter = chr(ord(letter) + 1)


def figS03(results: Path, out: Path):
    """Within-map bond-strain distributions at phi=0.50, eps=0.02, Set B."""
    loc = pd.read_csv(results / "figS03_distributions.csv")
    if "phi" in loc.columns:
        loc = loc[np.isclose(loc.phi, 0.5)]
    letter = "a"
    for rule in ("harmonic", "arithmetic"):
        for cl in CLASS_ORDER:
            fig, ax = plt.subplots(figsize=(3.6, 3.2), layout="constrained")
            g = loc[(loc.interface_rule == rule) & (loc.bond_class == cl)]
            assert len(g) == 24, f"figS03 {rule}/{cl}: {len(g)} rows (want 24)"
            for j, pat in enumerate(("SEEDED", "SPONTANEOUS")):
                s = g[g.pattern == pat].sort_values("replicate")
                xx = np.arange(len(s)) + (j - 0.5) * 0.2
                ax.vlines(xx, s.strain_q05 * 100, s.strain_q95 * 100,
                          color=COLORS[j], alpha=0.6, lw=0.7)
                ax.scatter(xx, s.strain_median * 100, s=6, color=COLORS[j])
            ax.set_title(f"{rule}; {cl}", fontsize=TITLE_FONTSIZE)
            ax.set_xlabel("Map replicate", fontsize=XLABEL_FONTSIZE)
            ax.set_ylabel("Bond strain (%)", fontsize=YLABEL_FONTSIZE)
            ax.set_xticks([0, 5, 11])
            save_panel(fig, out, "figS03_distributions", letter)
            letter = chr(ord(letter) + 1)


# ------------------------------------------------------------------------ main
data_dir: Path  # used by fig05 for the density table (canonical input)


def main(argv=None) -> int:
    global data_dir
    ap = argparse.ArgumentParser(description="Regenerate manuscript figure panels")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    data_dir = Path(args.data)
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    audit = [f"figures.py — inputs: {data_dir}, {results}; output: {out}"]
    states = load_states(data_dir)
    audit.append(f"fig01/02 spatial states: {states[0]['map_id']}, "
                 f"{states[1]['map_id']} (saved replicate-0 maps, eps=0.02)")

    fig01(states, out)
    fig02(states, results, out, audit)
    fig03(results, out)
    fig04(results, out)
    fig05(results, out)
    figS01(results, out)
    figS02(results, out)
    figS03(results, out)

    for line in audit:
        print(line)
    n = len(list(out.glob("fig*.pdf")))
    print(f"wrote {n} vector figure panels to {out} "
          "(one PDF per subplot; panel letter in the filename)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
