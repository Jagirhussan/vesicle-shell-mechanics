#!/usr/bin/env python3
"""analysis.py — tables, statistics and verification from the canonical data.

Reads only ``data/`` (the small canonical package) and generates every
manuscript-level number: condition summaries, clustered-minus-
random contrasts with map-bootstrap intervals, class-resolved local mechanics,
the D2 stiffness-placement counterfactual, the D1 density controls, and the
F1/F2/F3 / tangent-space estimator checks.  It never starts a simulation.

Statistical unit: the label map (12 per architecture at baseline, 4 for the
density controls, 3 parent maps per architecture for D2).  Harmonic and
arithmetic rules evaluated on the same map are paired repeated measurements;
the two architectures are never paired.  D2 children are averaged within
parent before contrasts (no pseudoreplication of the 36 shuffles).

Bootstrap: 15000 percentile draws resampling maps independently within each
architecture, public analysis seed 20261020.  A fresh generator with the same
seed per metric call makes the index draws identical across metrics within a
group, preserving joint covariance, so the accounting identity
dK_rel = dK_aff - dL holds draw by draw (verified below and in every build).

Commands:
    python analysis.py check --data data
        verify file hashes/counts/schemas/invariants; refit all 306 settings
        from the shipped energy curves and compare against the archived fit
        records; recompute representative-map local statistics and the
        infinitesimal tangent curvature.  Exits nonzero on any failure.
    python analysis.py build --data data --out results
        regenerate all tables and figure source CSVs; compare headline values
        against the published manuscript numbers (tolerance reflects stored
        rounding) and exit nonzero on a discrepancy.
    python analysis.py validate-raw --archive PATH --out results
        optional slow mode: recompute relaxed energies from a raw archive
        (the DOI-pending source_snapshot layout) and compare with the shipped
        curves.  Emits NOT_VERIFIED when the archive is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import model

BOOTSTRAP_SEED = 20261020
BOOTSTRAP_DRAWS = 15000
CLASS_NAMES = list(model.CLASS_NAMES)

# Published manuscript values (tables), used ONLY as regression
# references for the acceptance cross-check; every reported value is
# recomputed from data first.  Tolerances reflect stored rounding.
PUBLISHED = {
    "factorial": {  # (phi, set, rule): (dK_aff, dL, dK_rel, ci_lo, ci_hi)
        (0.15, "A", "harmonic"): (7.03, 7.20, -0.18, -0.47, 0.13),
        (0.15, "A", "arithmetic"): (0.02, 5.56, -5.54, -5.97, -5.08),
        (0.15, "B", "harmonic"): (12.29, 12.68, -0.39, -1.22, 0.48),
        (0.15, "B", "arithmetic"): (0.05, 10.50, -10.45, -11.54, -9.30),
        (0.50, "A", "harmonic"): (18.40, 22.12, -3.72, -4.59, -2.94),
        (0.50, "A", "arithmetic"): (0.02, 23.58, -23.56, -24.75, -22.33),
        (0.50, "B", "harmonic"): (32.18, 37.84, -5.66, -7.67, -3.83),
        (0.50, "B", "arithmetic"): (0.05, 38.59, -38.54, -41.03, -36.04),
    },
    "primary_means": {"clustered": 266.91, "random": 272.57},
    "local_strain": {  # (rule, class): (clustered %, random %, diff pp, lo, hi)
        ("harmonic", "soft-soft"): (1.313, 1.109, 0.204, 0.195, 0.214),
        ("harmonic", "stiff-stiff"): (0.593, 0.741, -0.148, -0.157, -0.140),
        ("harmonic", "interface"): (1.030, 1.007, 0.023, 0.015, 0.033),
        ("arithmetic", "soft-soft"): (1.371, 1.199, 0.173, 0.164, 0.182),
        ("arithmetic", "stiff-stiff"): (0.634, 0.821, -0.187, -0.196, -0.178),
        ("arithmetic", "interface"): (0.817, 0.922, -0.105, -0.113, -0.095),
    },
    "local_force": {("harmonic", "stiff-stiff"): (2.185, 2.758),
                    ("arithmetic", "stiff-stiff"): (2.335, 3.052)},
    "node_disp_rms": {"harmonic": ((0.045, 0.046), (0.019, 0.020))},
    "shuffle": {  # rule: (original, null, excess, excess_lo, excess_hi)
        "harmonic": (38.87, 17.86, 21.01, 18.41, 23.91),
        "arithmetic": (40.01, 17.56, 22.45, 19.36, 25.97),
    },
    "density": {  # (construction, rho, rule): (dK_rel, lo, hi)
        ("balanced_baseline", 0.50, "arithmetic"): (-40.24, -42.85, -38.25),
        ("balanced_baseline", 0.50, "harmonic"): (-6.67, -8.39, -5.49),
        ("fixed_cutoff", 0.35, "arithmetic"): (-20.68, -23.09, -18.71),
        ("fixed_cutoff", 0.35, "harmonic"): (-0.11, -2.76, 1.79),
        ("fixed_cutoff", 0.75, "arithmetic"): (-119.00, -120.20, -117.83),
        ("fixed_cutoff", 0.75, "harmonic"): (-37.16, -38.83, -35.36),
        ("coordination_matched", 0.35, "arithmetic"): (-41.59, -44.87, -38.36),
        ("coordination_matched", 0.35, "harmonic"): (-7.10, -10.36, -3.46),
        ("coordination_matched", 0.75, "arithmetic"): (-38.88, -40.86, -36.18),
        ("coordination_matched", 0.75, "harmonic"): (-5.34, -6.97, -3.33),
    },
    "fit_sensitivity": {  # (model, window): max |K - K_F1@0.02| over settings
        ("F1", 0.01): 0.0819, ("F1", 0.02): 0.0000, ("F1", 0.04): 0.2978,
        ("F2", 0.01): 0.0971, ("F2", 0.02): 0.0563, ("F2", 0.04): 0.1056,
        ("F3", 0.01): 4.1067, ("F3", 0.02): 8.2115, ("F3", 0.04): 16.3549,
    },
    "tangent": {"max_diff_F2": 0.0150,
                "contrast_harmonic": -5.67, "contrast_arithmetic": -38.55},
}


# ------------------------------------------------------------------ utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contrast(group: pd.DataFrame, metric: str) -> dict:
    """Clustered-minus-random contrast of ``metric`` with the map bootstrap."""
    a = group[group.pattern == "SEEDED"][metric].to_numpy()
    b = group[group.pattern == "SPONTANEOUS"][metric].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = (a[rng.integers(len(a), size=(BOOTSTRAP_DRAWS, len(a)))].mean(1)
             - b[rng.integers(len(b), size=(BOOTSTRAP_DRAWS, len(b)))].mean(1))
    return {"metric": metric, "estimate": float(a.mean() - b.mean()),
            "mean_clustered": float(a.mean()), "mean_random": float(b.mean()),
            "sd_clustered": float(a.std(ddof=1)), "sd_random": float(b.std(ddof=1)),
            "ci_lo": float(np.quantile(draws, 0.025)),
            "ci_hi": float(np.quantile(draws, 0.975)),
            "n_clustered": len(a), "n_random": len(b)}


def draw_indices(n_a: int, n_b: int) -> tuple:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return (rng.integers(n_a, size=(BOOTSTRAP_DRAWS, n_a)),
            rng.integers(n_b, size=(BOOTSTRAP_DRAWS, n_b)))


def joint_draws(group: pd.DataFrame, metric: str, indices: tuple) -> np.ndarray:
    a = group[group.pattern == "SEEDED"][metric].to_numpy()
    b = group[group.pattern == "SPONTANEOUS"][metric].to_numpy()
    return a[indices[0]].mean(1) - b[indices[1]].mean(1)


# ----------------------------------------------------------------------- check
def check(data: Path) -> None:
    failures = []

    def expect(cond, msg):
        if cond:
            print(f"  ok: {msg}")
        else:
            failures.append(msg)
            print(f"  FAIL: {msg}")

    manifest = json.loads((data / "manifest.json").read_text())
    print(f"analysis.py check — dataset {manifest['dataset_version']}")

    for name, rec in manifest["files"].items():
        p = data / name
        expect(p.exists() and p.stat().st_size == rec["bytes"]
               and sha256(p) == rec["sha256"], f"{name}: sha256/size match manifest")

    ns = pd.read_csv(data / "network_summary.csv")
    lm = pd.read_csv(data / "local_mechanics.csv")
    sp = pd.read_csv(data / "shuffle_parents.csv")
    sc = pd.read_csv(data / "shuffle_children.csv")
    dc = pd.read_csv(data / "density_controls.csv")
    ft = pd.read_csv(data / "fit_and_tangent_checks.csv")
    curves = pd.read_csv(data / "strain_curves.csv")

    expect(len(ns) == 194 and (ns.role == "factorial").sum() == 192
           and (ns.role == "uniform_control").sum() == 2,
           "network_summary: 192 factorial + 2 labelled uniform rows")
    expect(not ns.duplicated(["setting_id"]).any()
           and not ns.duplicated(["phi", "stiffness_set", "pattern",
                                  "interface_rule", "replicate"]).any(),
           "network_summary: unique setting keys")
    expect(len(lm) == 576 and len(sp) == 12 and len(sc) == 36 and len(dc) == 80,
           "row counts local/shuffle/density")
    expect(len(curves) == 306 * 9, "strain_curves: 306 settings x 9 strains")
    fac = ns[ns.role == "factorial"]
    expect(all(len(g) == 48 for _, g in fac.groupby(["phi", "pattern"])),
           "12 maps per architecture and fraction (x 2 sets x 2 rules)")
    expect(np.allclose(ns.K_aff_mNm - ns.K_rel_F2_mNm, ns.L_mNm),
           "identity L = K_aff - K_rel(F2) on every row")
    expect(np.allclose(ns.K_rel_F2_mNm / ns.K_aff_mNm, ns.q),
           "identity q = K_rel/K_aff")
    expect(bool(fac.numerical_qc_pass.all()),
           "all factorial rows pass archived numerical QC")
    expect(bool(fac[fac.pattern == "SEEDED"].morans_I.mean()
                > fac[fac.pattern == "SPONTANEOUS"].morans_I.mean()),
           "clustered maps have larger mean Moran's I")

    expect(set(sc.parent_assignment_id) == set(sp.parent_assignment_id)
           and bool(sc.binwise_inventory_exact.all())
           and bool(sc.global_inventory_exact.all()),
           "D2 children: 6 parents linked, binwise stiffness multisets exact")
    expect(float(sc.affine_drift_mNm.abs().max()) < 0.5
           and np.allclose(sc.affine_drift_mNm, sc.affine_drift_qc_mNm, atol=1e-12),
           f"D2 affine drift measured and small (max "
           f"{sc.affine_drift_mNm.abs().max():.4f} mN/m; ~0.1% of K_aff, "
           f"binwise multisets exact)")

    d1b = dc[dc.construction == "coordination_matched"]
    expect(np.allclose(d1b.cutoff_nm, 2.5 * np.sqrt(0.5 / d1b.rho_nodes_per_nm2)),
           "D1-B cutoffs follow 2.5*sqrt(0.5/rho)")
    d1a = dc[dc.construction == "fixed_cutoff"]
    expect(bool(np.allclose(d1a.cutoff_nm, 2.5)), "D1-A cutoffs fixed at 2.5 nm")
    for rho, (N_ref, E_a, E_b) in ((0.35, (1759, 5633, 7019)),
                                   (0.75, (3769, 25904, 15050))):
        ga = d1a[np.isclose(d1a.rho_nodes_per_nm2, rho)]
        gb = d1b[np.isclose(d1b.rho_nodes_per_nm2, rho)]
        expect(int(ga.N.iloc[0]) == N_ref and int(ga.n_bonds.iloc[0]) == E_a
               and int(gb.n_bonds.iloc[0]) == E_b,
               f"density rho={rho}: archived (N,E) reproduced")

    # affine energy column: functional form on every setting, K_aff on baseline
    ratio_dev = aff_dev = 0.0
    ns_idx = ns.set_index("setting_id")
    for sid, g in curves.groupby("setting_id"):
        g = g.sort_values("epsilon_s")
        eps = g.epsilon_s.to_numpy()
        s = np.sqrt(1.0 + eps) - 1.0
        pos = eps > 0
        form = (g.U_aff_kJmol.to_numpy()[pos] / s[pos] ** 2) \
            * (s[pos][0] ** 2 / g.U_aff_kJmol.to_numpy()[pos][0])
        ratio_dev = max(ratio_dev, float(np.abs(form - 1.0).max()))
        if sid in ns_idx.index:
            S = float(ns_idx.loc[sid, "K_aff_mNm"]) / model.C_MNM_PER_OMM * 4 * model.nominal_area()
            aff_dev = max(aff_dev, float(np.abs(
                g.U_aff_kJmol.to_numpy() - 0.5 * S * s**2).max()))
    expect(ratio_dev < 1e-9,
           "affine energies follow 0.5*S*(sqrt(1+e)-1)^2 form on all 306 settings")
    expect(aff_dev < 1e-6,
           "affine energies consistent with K_aff rows (S = 4*A0*K_aff/c)")

    # independent refit of every setting from the shipped energy curves
    fits = ft[ft.kind == "fit"]
    expect(len(fits) == 5508, "5508 archived fit records")
    max_dev = 0.0
    for sid, g in curves.groupby("setting_id"):
        g = g.sort_values("epsilon_s")
        eps = g.epsilon_s.to_numpy()
        poly = g.A_poly_nm2.to_numpy() / g.A_poly_nm2.to_numpy()[0] - 1.0
        U = g.U_rel_kJmol.to_numpy()
        for window in model.FIT_WINDOWS:
            for m in ("F1", "F2", "F3"):
                f = model.fit_modulus(eps, U, window, m)
                ref = fits[(fits.setting_id == sid) & (fits.measure == "sphere")
                           & np.isclose(fits.window, window) & (fits.model == m)]
                max_dev = max(max_dev, abs(f["K_mNm"] - float(ref.K_mNm.iloc[0])))
            # polyhedral: predictor = polyhedral strain, membership nominal,
            # scaling by the max selected predictor (archived convention)
            sel = eps <= window + 1e-12
            w = float(poly[sel].max()) + 1e-14
            t = poly[sel] / w
            yy = (model.C_MNM_PER_OMM * (U[sel] - U[0]) / g.A_poly_nm2.to_numpy()[0])
            for m in ("F1", "F2", "F3"):
                powers = {"F1": [1, 2, 3], "F2": [2, 3], "F3": [2]}[m]
                X = np.array([p_**q for q in powers for p_ in [t]]).T
                beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
                K = 2.0 * beta[powers.index(2)] / w**2
                ref = fits[(fits.setting_id == sid) & (fits.measure == "polyhedral")
                           & np.isclose(fits.window, window) & (fits.model == m)]
                max_dev = max(max_dev, abs(K - float(ref.K_mNm.iloc[0])))
    expect(max_dev < 1e-9,
           f"all 5508 F1/F2/F3 fits reproduced from energy curves (max dev {max_dev:.2e})")

    # representative-map local statistics recomputed from saved states
    z = np.load(data / "representative_states.npz")
    for tag, sid in (("mapA", "phi050_seeded_rep0000_B_harmonic"),
                     ("mapB", "phi050_spontaneous_rep0000_B_harmonic")):
        net = model.network_from_arrays(
            z[f"{tag}_xyz0"], z[f"{tag}_pairs"], z[f"{tag}_l0"], None,
            z[f"{tag}_k_harmonic"], z[f"{tag}_bond_class"])
        x_rel = z[f"{tag}_x_rel_eps0.020"]
        cls = net.class_summary(x_rel)
        row = lm[(lm.setting_id == sid) & np.isclose(lm.epsilon_s, 0.02)]
        devs = [abs(cls[c]["strain_mean"]
                    - float(row[row.bond_class == c].strain_mean.iloc[0]))
                for c in CLASS_NAMES]
        fdev = abs(cls["stiff-stiff"]["force_mean"]
                   - float(row[row.bond_class == "stiff-stiff"]
                           .force_kJmol_nm_mean.iloc[0]))
        expect(max(devs) < 1e-10 and fdev < 1e-10,
               f"{tag}: class strain/force recomputed from saved state "
               f"(max dev {max(devs + [fdev]):.2e})")
        _, tangent = model.tangential_displacement(
            x_rel, model.affine_positions(z[f"{tag}_xyz0"], 0.02))
        labels = z[f"{tag}_phase_labels"]
        rms_soft = float(np.sqrt(np.mean(np.sum(tangent[labels == 0]**2, axis=1))))
        rms_stiff = float(np.sqrt(np.mean(np.sum(tangent[labels == 1]**2, axis=1))))
        rdev = max(abs(rms_soft - float(row.soft_disp_rms_nm.iloc[0])),
                   abs(rms_stiff - float(row.stiff_disp_rms_nm.iloc[0])))
        expect(rdev < 1e-10, f"{tag}: Kabsch tangential node displacement (dev {rdev:.2e})")
        h = model.tangent_curvature(net, z[f"{tag}_xyz0"])
        href = ft[(ft.kind == "hessian") & (ft.setting_id == sid)]
        expect(abs(h["K0_mNm"] - float(href.K_mNm.iloc[0])) < 1e-6,
               f"{tag}: infinitesimal tangent curvature "
               f"(dK={h['K0_mNm'] - float(href.K_mNm.iloc[0]):+.2e} mN/m)")

    if failures:
        print(f"CHECK FAILED: {len(failures)} failures")
        sys.exit(1)
    print("CHECK PASSED")


# ----------------------------------------------------------------------- build
def build(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ns = pd.read_csv(data / "network_summary.csv")
    lm = pd.read_csv(data / "local_mechanics.csv")
    sp = pd.read_csv(data / "shuffle_parents.csv")
    dc = pd.read_csv(data / "density_controls.csv")
    ft = pd.read_csv(data / "fit_and_tangent_checks.csv")
    fits = ft[ft.kind == "fit"]
    hess = ft[ft.kind == "hessian"]
    fac = ns[ns.role == "factorial"].copy()

    # ---------------------------------------------- baseline contrasts (S2) ---
    rows = []
    for keys, g in fac.groupby(["phi", "stiffness_set", "interface_rule"]):
        for metric, col in (("K_aff", "K_aff_mNm"), ("K_rel", "K_rel_F2_mNm"),
                            ("L", "L_mNm"), ("q", "q")):
            c = contrast(g, col)
            c["metric"] = metric
            rows.append(dict(study="stage2_main", phi=keys[0], stiffness_set=keys[1],
                             rule=keys[2], rho=0.5, **c))
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "contrasts.csv", index=False)

    g = fac[(fac.phi == 0.5) & (fac.stiffness_set == "B")]
    idx = draw_indices(12, 12)
    id_dev = 0.0
    for rule in ("harmonic", "arithmetic"):
        gr = g[g.interface_rule == rule]
        d_ka = joint_draws(gr, "K_aff_mNm", idx)
        d_l = joint_draws(gr, "L_mNm", idx)
        d_kr = joint_draws(gr, "K_rel_F2_mNm", idx)
        id_dev = max(id_dev, float(np.abs(d_kr - (d_ka - d_l)).max()))
    assert id_dev < 1e-9, "joint-bootstrap identity violated"
    print(f"  identity dK_rel = dK_aff - dL holds within every bootstrap draw "
          f"(max dev {id_dev:.2e} mN/m)")
    contrasts[contrasts.metric.isin(["K_aff", "L", "K_rel"])].pivot_table(
        index=["phi", "stiffness_set", "rule"], columns="metric",
        values=["estimate", "ci_lo", "ci_hi"]).to_csv(out / "table_factorial.csv")

    # ------------------------------------------- local mechanics (Table 1) ---
    rows = []
    for keys, grp in lm.groupby(["phi", "interface_rule", "epsilon_s", "bond_class"]):
        for metric, col in (("strain_mean", "strain_mean"),
                            ("extension_nm_mean", "extension_nm_mean"),
                            ("force_kJmol_nm_mean", "force_kJmol_nm_mean"),
                            ("energy_fraction", "energy_fraction")):
            c = contrast(grp, col)
            c["metric"] = metric
            rows.append(dict(phi=keys[0], rule=keys[1], epsilon_s=keys[2],
                             bond_class=keys[3], **c))
    local_contrasts = pd.DataFrame(rows)
    local_contrasts.to_csv(out / "local_contrasts.csv", index=False)
    primary = local_contrasts[(np.isclose(local_contrasts.epsilon_s, 0.02))
                              & (local_contrasts.phi == 0.5)
                              & (local_contrasts.metric == "strain_mean")]
    tl = primary[["rule", "bond_class", "mean_clustered", "mean_random",
                  "estimate", "ci_lo", "ci_hi"]].copy()
    for c in ("mean_clustered", "mean_random", "estimate", "ci_lo", "ci_hi"):
        tl[c] *= 100.0
    tl.to_csv(out / "table_local_strain.csv", index=False)
    forces = local_contrasts[(local_contrasts.phi == 0.5)
                             & np.isclose(local_contrasts.epsilon_s, 0.02)
                             & (local_contrasts.metric == "force_kJmol_nm_mean")]
    forces[["rule", "bond_class", "mean_clustered", "mean_random"]].to_csv(
        out / "table_local_force.csv", index=False)

    # --------------------------------------------------- shuffle D2 (Table S3)
    sh_rows = []
    for rule, grp in sp.groupby("interface_rule"):
        for state, col in (("original", "L_mNm"), ("null", "L_null_mean_mNm"),
                           ("excess", "L_excess_mNm")):
            sh_rows.append(dict(rule=rule, quantity="L", state=state,
                                **contrast(grp, col)))
    shuffle_contrasts = pd.DataFrame(sh_rows)
    shuffle_contrasts.to_csv(out / "shuffle_contrasts.csv", index=False)
    shuffle_contrasts.to_csv(out / "table_shuffle.csv", index=False)

    # --------------------------------------------------- density D1 (Table S4)
    dens_rows = []
    for keys, grp in dc.groupby(["construction", "rho_nodes_per_nm2", "interface_rule"]):
        for metric, col in (("K_aff", "K_aff_mNm"), ("K_rel", "K_rel_F2_mNm"),
                            ("L", "L_mNm")):
            c = contrast(grp, col)
            c["metric"] = metric
            dens_rows.append(dict(construction=keys[0], rho=keys[1], rule=keys[2], **c))
    density_contrasts = pd.DataFrame(dens_rows)
    density_contrasts.to_csv(out / "density_contrasts.csv", index=False)
    density_contrasts[density_contrasts.metric == "K_rel"][
        ["construction", "rho", "rule", "estimate", "ci_lo", "ci_hi",
         "n_clustered"]].to_csv(out / "table_density.csv", index=False)

    # ----------------------------------------- fit sensitivity (Table S1) ---
    fsp = fits[fits.measure == "sphere"]  # all 306 saved evaluations
    base = fsp[np.isclose(fsp.window, 0.02) & (fsp.model == "F1")].set_index("setting_id")
    fit_rows = []
    for (m, w), grp in fsp.groupby(["model", "window"]):
        devs = (grp.set_index("setting_id").K_mNm - base.K_mNm).abs()
        fit_rows.append(dict(model=m, window=float(w),
                             n_points=int(grp.n_points.iloc[0]),
                             max_abs_dev_mNm=float(devs.max()),
                             min_K_mNm=float(grp.K_mNm.min()),
                             n_negative=int((grp.K_mNm < 0).sum())))
    table_fits = pd.DataFrame(fit_rows)
    table_fits.to_csv(out / "table_fit_validation.csv", index=False)

    prim = fits[(fits.measure == "sphere") & (fits.model == "F2")
                & np.isclose(fits.window, 0.02) & (fits.study_id == "stage2_main")
                & (fits.stiffness_set == "B") & np.isclose(fits.phi, 0.5)
                & (fits.interface_rule != "uniform")]
    merged = prim.merge(hess, on="setting_id", suffixes=("_fit", "_hess"))
    merged["abs_diff"] = (merged.K_mNm_fit - merged.K_mNm_hess).abs()
    tang = {"n_settings": int(len(merged)),
            "max_abs_diff_F2_vs_tangent_mNm": float(merged.abs_diff.max())}
    for rule in ("harmonic", "arithmetic"):
        gg = merged[merged.interface_rule_fit == rule]
        tang[f"contrast_{rule}_mNm"] = float(
            gg[gg.pattern_fit == "SEEDED"].K_mNm_hess.mean()
            - gg[gg.pattern_fit == "SPONTANEOUS"].K_mNm_hess.mean())
    merged.to_csv(out / "tangent_crosscheck.csv", index=False)

    # ------------------------------------------------------ figure sources ---
    fig2 = lm[np.isclose(lm.epsilon_s, 0.02) & np.isclose(lm.phi, 0.5)]
    fig2[["setting_id", "pattern", "interface_rule", "bond_class", "strain_mean",
          "force_kJmol_nm_mean", "soft_disp_rms_nm", "stiff_disp_rms_nm",
          "compressed_fraction"]].to_csv(out / "fig2_class_means.csv", index=False)
    contrasts[np.isclose(contrasts.phi, 0.5)].to_csv(out / "fig2_contrasts.csv",
                                                     index=False)
    contrasts.to_csv(out / "fig3_contrasts.csv", index=False)
    fac[(fac.phi == 0.5) & (fac.stiffness_set == "B")].groupby(
        ["interface_rule", "pattern"])[
        ["K_aff_soft_mNm", "K_aff_stiff_mNm", "K_aff_interface_mNm"]].mean().to_csv(
        out / "fig3_affine_classes.csv")
    sp.to_csv(out / "fig4_parents.csv", index=False)
    sc = pd.read_csv(data / "shuffle_children.csv")
    sc[["parent_assignment_id", "parent_pattern", "shuffle_id", "interface_rule",
        "L_child_mNm", "affine_drift_mNm"]].to_csv(out / "fig4_children.csv",
                                                   index=False)
    shuffle_contrasts.to_csv(out / "fig4_contrasts.csv", index=False)
    dc.groupby(["construction", "rho_nodes_per_nm2"]).mean_degree.mean().to_csv(
        out / "fig5_degree.csv")
    dc[dc.pattern == "SEEDED"].groupby(
        ["construction", "rho_nodes_per_nm2"]).domain_mean_r_eq_nm.mean().to_csv(
        out / "fig5_domain_radius.csv")
    density_contrasts.to_csv(out / "fig5_contrasts.csv", index=False)
    fits[(fits.study_id == "stage2_main") & np.isclose(fits.phi, 0.5)
         & (fits.stiffness_set == "B") & (fits.measure == "sphere")][
        ["setting_id", "pattern", "interface_rule", "model", "window",
         "K_mNm"]].to_csv(out / "figS01_fits.csv", index=False)
    local_contrasts.to_csv(out / "figS02_local_contrasts.csv", index=False)
    lm[np.isclose(lm.epsilon_s, 0.02) & np.isclose(lm.phi, 0.5)][
        ["setting_id", "pattern", "interface_rule", "bond_class", "replicate", "phi",
         "strain_q05", "strain_median", "strain_q95", "strain_mean"]].to_csv(
        out / "figS03_distributions.csv", index=False)

    # ---------------------------------------------- acceptance cross-check ---
    report = []
    ok_all = True

    def cmp(name, got, want, tol):
        nonlocal ok_all
        good = abs(got - want) <= tol
        ok_all &= good
        report.append({"check": name, "recomputed": round(float(got), 6),
                       "published": want, "abs_diff": round(abs(got - want), 6),
                       "tolerance": tol, "pass": bool(good)})

    for (phi, sset, rule), (ka, dl, kr, lo, hi) in PUBLISHED["factorial"].items():
        gg = contrasts[np.isclose(contrasts.phi, phi)
                       & (contrasts.stiffness_set == sset) & (contrasts.rule == rule)]
        cmp(f"factorial dK_aff phi={phi} {sset} {rule}",
            gg[gg.metric == "K_aff"].estimate.iloc[0], ka, 0.02)
        cmp(f"factorial dL phi={phi} {sset} {rule}",
            gg[gg.metric == "L"].estimate.iloc[0], dl, 0.02)
        cmp(f"factorial dK_rel phi={phi} {sset} {rule}",
            gg[gg.metric == "K_rel"].estimate.iloc[0], kr, 0.02)
        cmp(f"factorial dK_rel lo phi={phi} {sset} {rule}",
            gg[gg.metric == "K_rel"].ci_lo.iloc[0], lo, 0.02)
        cmp(f"factorial dK_rel hi phi={phi} {sset} {rule}",
            gg[gg.metric == "K_rel"].ci_hi.iloc[0], hi, 0.02)
    g = fac[(fac.phi == 0.5) & (fac.stiffness_set == "B")
            & (fac.interface_rule == "harmonic")]
    cmp("primary K_rel clustered mean", g[g.pattern == "SEEDED"].K_rel_F2_mNm.mean(),
        PUBLISHED["primary_means"]["clustered"], 0.02)
    cmp("primary K_rel random mean",
        g[g.pattern == "SPONTANEOUS"].K_rel_F2_mNm.mean(),
        PUBLISHED["primary_means"]["random"], 0.02)
    for (rule, bclass), (mc, mr, d, lo, hi) in PUBLISHED["local_strain"].items():
        gg = primary[(primary.rule == rule) & (primary.bond_class == bclass)]
        cmp(f"local {rule} {bclass} clustered %", gg.mean_clustered.iloc[0] * 100, mc, 0.002)
        cmp(f"local {rule} {bclass} random %", gg.mean_random.iloc[0] * 100, mr, 0.002)
        cmp(f"local {rule} {bclass} diff pp", gg.estimate.iloc[0] * 100, d, 0.002)
        cmp(f"local {rule} {bclass} lo", gg.ci_lo.iloc[0] * 100, lo, 0.002)
        cmp(f"local {rule} {bclass} hi", gg.ci_hi.iloc[0] * 100, hi, 0.002)
    for (rule, bclass), (fc, fr) in PUBLISHED["local_force"].items():
        gg = forces[(forces.rule == rule) & (forces.bond_class == bclass)]
        cmp(f"local force {rule} {bclass} clustered", gg.mean_clustered.iloc[0], fc, 0.002)
        cmp(f"local force {rule} {bclass} random", gg.mean_random.iloc[0], fr, 0.002)
    disp = lm[np.isclose(lm.epsilon_s, 0.02) & np.isclose(lm.phi, 0.5)
              & (lm.interface_rule == "harmonic")]
    (lo_c, hi_c), (lo_r, hi_r) = PUBLISHED["node_disp_rms"]["harmonic"]
    clustered = disp[disp.pattern == "SEEDED"]
    random = disp[disp.pattern == "SPONTANEOUS"]
    ok_all &= (lo_c - 1e-3 <= clustered.soft_disp_rms_nm.mean() <= hi_c + 1e-3
               and lo_c - 1e-3 <= clustered.stiff_disp_rms_nm.mean() <= hi_c + 1e-3)
    ok_all &= (lo_r - 1e-3 <= random.soft_disp_rms_nm.mean() <= hi_r + 1e-3
               and lo_r - 1e-3 <= random.stiff_disp_rms_nm.mean() <= hi_r + 1e-3)
    report.append({"check": "node disp RMS harmonic clustered soft/stiff nm",
                   "recomputed": [round(clustered.soft_disp_rms_nm.mean(), 5),
                                  round(clustered.stiff_disp_rms_nm.mean(), 5)],
                   "published": [lo_c, hi_c],
                   "pass": bool(lo_c - 1e-3 <= clustered.soft_disp_rms_nm.mean()
                                <= hi_c + 1e-3
                                and lo_c - 1e-3 <= clustered.stiff_disp_rms_nm.mean()
                                <= hi_c + 1e-3)})
    report.append({"check": "node disp RMS harmonic random soft/stiff nm",
                   "recomputed": [round(random.soft_disp_rms_nm.mean(), 5),
                                  round(random.stiff_disp_rms_nm.mean(), 5)],
                   "published": [lo_r, hi_r],
                   "pass": bool(lo_r - 1e-3 <= random.soft_disp_rms_nm.mean()
                                <= hi_r + 1e-3
                                and lo_r - 1e-3 <= random.stiff_disp_rms_nm.mean()
                                <= hi_r + 1e-3)})
    for rule, (o, n, e, lo, hi) in PUBLISHED["shuffle"].items():
        gg = shuffle_contrasts[shuffle_contrasts.rule == rule].set_index("state")
        cmp(f"shuffle {rule} original", gg.loc["original", "estimate"], o, 0.02)
        cmp(f"shuffle {rule} null", gg.loc["null", "estimate"], n, 0.02)
        cmp(f"shuffle {rule} excess", gg.loc["excess", "estimate"], e, 0.02)
        cmp(f"shuffle {rule} excess lo", gg.loc["excess", "ci_lo"], lo, 0.02)
        cmp(f"shuffle {rule} excess hi", gg.loc["excess", "ci_hi"], hi, 0.02)
    for (cons, rho, rule), (kr, lo, hi) in PUBLISHED["density"].items():
        gg = density_contrasts[(density_contrasts.construction == cons)
                               & np.isclose(density_contrasts.rho, rho)
                               & (density_contrasts.rule == rule)
                               & (density_contrasts.metric == "K_rel")]
        cmp(f"density {cons} rho={rho} {rule}", gg.estimate.iloc[0], kr, 0.02)
        cmp(f"density {cons} rho={rho} {rule} lo", gg.ci_lo.iloc[0], lo, 0.02)
        cmp(f"density {cons} rho={rho} {rule} hi", gg.ci_hi.iloc[0], hi, 0.02)
    for (m, w), dev in PUBLISHED["fit_sensitivity"].items():
        got = table_fits[(table_fits.model == m)
                         & np.isclose(table_fits.window, w)].max_abs_dev_mNm.iloc[0]
        cmp(f"fit sensitivity {m} w={w}", got, dev, 0.005)
    cmp("tangent max |F2-K0|", tang["max_abs_diff_F2_vs_tangent_mNm"],
        PUBLISHED["tangent"]["max_diff_F2"], 0.002)
    cmp("tangent contrast harmonic", tang["contrast_harmonic_mNm"],
        PUBLISHED["tangent"]["contrast_harmonic"], 0.02)
    cmp("tangent contrast arithmetic", tang["contrast_arithmetic_mNm"],
        PUBLISHED["tangent"]["contrast_arithmetic"], 0.02)

    ver = pd.DataFrame(report)
    ver.to_csv(out / "acceptance_crosscheck.csv", index=False)
    n_pass = int(ver["pass"].sum())
    print(ver.to_string(index=False))
    print(f"acceptance cross-check: {n_pass}/{len(ver)} pass")
    (out / "verification_summary.json").write_text(json.dumps(
        {"n_checks": len(ver), "n_pass": n_pass, "all_passed": bool(ok_all),
         "tangent": tang,
         "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                       "identity_max_dev_mNm": id_dev}}, indent=2))
    if not ok_all:
        print("ACCEPTANCE FAILED — discrepancies listed above; see "
              "acceptance_crosscheck.csv")
        sys.exit(1)
    print(f"BUILD PASSED — tables and figure sources written to {out}")


# ---------------------------------------------------------------- validate-raw
def validate_raw(archive, data: Path, out: Path) -> None:
    """Slow optional mode: recompute energies from a raw archive and compare.

    Without an accessible archive this emits NOT_VERIFIED (never a pass).
    """
    if archive is None or not Path(archive).exists():
        print("NOT_VERIFIED: no raw archive provided/accessible "
              "(public DOI deposit pending author approval)")
        return
    root = Path(archive)
    cm = root / "data" / "derived" / "canonical_moduli.csv"
    if not cm.exists():
        print(f"NOT_VERIFIED: {cm} not found in archive")
        return
    canon = pd.read_csv(cm)
    curves = pd.read_csv(data / "strain_curves.csv")
    lookup = {(r.setting_id, round(r.epsilon_s, 6)): r.U_rel_kJmol
              for r in curves.itertuples()}
    max_gap, n = 0.0, 0
    for row in canon.itertuples():
        pos = np.load(root / row.positions_path)
        net = np.load(root / (row.correction_parent_path
                              if row.study_id == "stage2b_validation"
                              else row.raw_source_path))
        k = net["k_shuffle"] if row.study_id == "stage2c_d2" else net["k_omm"]
        pairs, l0 = net["pairs"], net["l0"]
        prefix = "chosen" if row.study_id == "stage2b_validation" else "relaxed"
        for it, eps in enumerate(pos["strains"]):
            x = pos[f"{prefix}_eps_{it:03d}"]
            lengths = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
            energy = float(0.5 * np.dot(k, (lengths - l0) ** 2))
            ref = lookup[(row.setting_id, round(float(eps), 6))]
            max_gap = max(max_gap, abs(energy - ref))
            n += 1
    print(f"VALIDATE-RAW: {n} energies recomputed from archive; "
          f"max |gap| = {max_gap:.3e} kJ/mol vs shipped curves")
    (out / "validate_raw_summary.json").write_text(json.dumps(
        {"n_states": n, "max_abs_gap_kJmol": max_gap}, indent=2))
    if max_gap > 1e-8:
        print("VALIDATE-RAW FAILED")
        sys.exit(1)


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify schemas/hashes/invariants and refit")
    c.add_argument("--data", default="data")
    b = sub.add_parser("build", help="regenerate tables and figure sources")
    b.add_argument("--data", default="data")
    b.add_argument("--out", default="results")
    v = sub.add_parser("validate-raw", help="optional slow raw-archive check")
    v.add_argument("--archive", default=None)
    v.add_argument("--data", default="data")
    v.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        check(Path(args.data))
    elif args.cmd == "build":
        build(Path(args.data), Path(args.out))
    elif args.cmd == "validate-raw":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        validate_raw(args.archive, Path(args.data), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
