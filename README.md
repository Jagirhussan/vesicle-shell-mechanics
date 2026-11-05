# Vesicle shell mechanics

Minimal, verifiable code and data release for the manuscript
**“Clustering-induced softening through non-affine relaxation in heterogeneous spherical networks”**.
The model is an idealised, athermal,
permanently bonded central-force network on a sphere with prescribed *stiff*
and *soft* node classes. Uniform radial dilation defines an exact **affine
stiffness inventory** `K_aff`; subsequent tangential relaxation at fixed node
radius defines a protocol-specific **relaxed energy curvature** `K_rel`; their
difference `L = K_aff − K_rel` measures relaxation loss. The paper's central
result is that clustering stiff nodes can raise the affine inventory and yet
*soften* the relaxed shell (`ΔK_rel < 0`) because relaxation loss grows faster.
The node labels are mechanical classes, not measured lipid phases; the node
fraction is not a molar fraction; nothing here predicts a native-vesicle
modulus, stress field, delivery performance or rupture.

## What is in this repository

- **`model.py`** — the physical model in one readable file: deterministic
  Fibonacci-sphere geometry and frozen distance-cutoff spring graph, the
  archived SEEDED (cluster-growth) / SPONTANEOUS (uniform-random) label
  generators with explicit seeds, harmonic/arithmetic interface stiffness,
  SciPy L-BFGS-B relaxation constrained to the target sphere, exact affine
  curvature, the F1/F2/F3 curvature estimators, the independent
  tangent-space (constrained-Hessian) check, the D2 length-bin stiffness
  shuffle and D1 density constructions, plus `self-test` / `example` / `run`
  command-line modes. OpenMM is optional and used only as an energy/force
  spot check.
- **`analysis.py`** — verifies the canonical data (`check`) and regenerates
  every manuscript-level table and statistic (`build`): clustered-vs-random
  contrasts, 15 000-draw map bootstrap (seed 20261020), class-resolved local
  bond/node mechanics, the D2 parent-level counterfactual, D1 density
  controls, fit-sensitivity and tangent checks, and an acceptance cross-check
  against the published numbers. Never starts a simulation.
- **`figures.py`** — regenerates all five main figures and three SI figures as
  vector PDFs from `data/` + `results/` only.
- **`data/`** — canonical inputs (see below). Everything needed for Tier-1
  (figure/table) reproduction; compact representative arrays for Tier-2
  spot checks.

## Install

Python 3.10+ with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib
```

`openmm` is optional (energy/force cross-check only; reported as SKIPPED when
absent — never silently passed).

## Quickstart (minutes, no simulation)

```bash
python analysis.py check --data data          # hashes, schemas, refits, invariants
python analysis.py build --data data --out results   # tables + figure sources + acceptance check
python figures.py --data data --results results --out figures
```

Expected outputs: `results/*.csv` (tables `table_primary`-style sources,
`acceptance_crosscheck.csv` — 130/130 manuscript-value checks — and per-figure
sources) and 35 vector figure panels in `figures/` — one PDF per subplot,
named with the panel letter (fig01a_model.pdf, fig01b_model.pdf, …
fig02a_local–fig02i_local, fig03a–d_factorial, fig04a–c_shuffle,
fig05a–d_density, figS01a/b_fits, figS02a–d_local_sensitivity,
figS03a–f_distributions).

## One-run model reproducibility (≈ 5 min + verify ≈ 25 min)

```bash
python model.py self-test            # incl. saved-fixture energy/modulus regression
python model.py example --out results/example          # fresh deterministic network
python model.py example --out results/example --verify # rerun saved fixtures:
                                      # 2 baseline maps × 2 rules, one D2 child,
                                      # one D1-B density network vs archived values
```

`example` regenerates the representative φ=0.50, set-B harmonic clustered
network from the archived seed recipe and records configuration, seed, code
hash and environment in `results/example/summary.json`; `--verify` proves the
freshly generated labels are *identical* to the archived realisation and that
relaxed energy curves, F2 moduli, class strains, the shuffled child and the
coordination-matched network all reproduce (documented tolerances ~10⁻⁶ kJ/mol
and 0.01 mN/m; observed agreement is ~10⁻¹²). `python model.py run --phi …
--pattern … --rep … --set {A,B} --rule …` runs any explicit setting. The full
192-evaluation campaign is available but deliberately opt-in:
`python model.py run-full --confirm-expensive --out results/campaign`
(hours of CPU).

## Paper-to-data map

| Manuscript item | Source | Command |
|---|---|---|
| Fig. 1 (a,b maps; c schematic) | `data/representative_states.npz` (saved replicate-0 maps) | `python figures.py` |
| Fig. 2 (spatial panels, class dots, competition) | `representative_states.npz`, `results/fig2_*.csv` | `figures.py` |
| Fig. 3 (factorial; class affine accounting) | `results/fig3_*.csv` ← `data/network_summary.csv` | `figures.py` |
| Fig. 4 (D2 shuffle) | `data/shuffle_parents.csv`, `shuffle_children.csv` | `figures.py` |
| Fig. 5 (D1 density controls) | `data/density_controls.csv` | `figures.py` |
| Figs. S1–S3 | `data/fit_and_tangent_checks.csv`, `local_mechanics.csv` | `figures.py` |
| Table 1 + forces/displacements | `data/local_mechanics.csv` | `analysis.py build` |
| Tables S1–S4 | `strain_curves.csv`, `network_summary.csv`, `shuffle_*.csv`, `density_controls.csv` | `analysis.py build` |

Independent sample sizes: 12 maps per architecture at baseline (192 crossed
evaluations on 48 maps); 4 maps per architecture for each density control;
3 parent maps per architecture for D2 (36 dependent children, averaged within
parent before contrasts). Two uniform-stiffness controls and 12 numerical
validation repeats are labelled as methodological checks in
`fit_and_tangent_checks.csv` / `network_summary.csv` and excluded from
inferential samples.

Canonical data files: `network_summary.csv` (194 rows: 192 factorial + 2
labelled uniform), `strain_curves.csv` (306 × 9 relaxed/affine energies and
polyhedral areas — enables every F1/F2/F3 refit), `local_mechanics.csv` (576
map×rule×strain×bond-class rows with node-class displacement and compressed
fractions), `shuffle_parents.csv` / `shuffle_children.csv` (12 / 36 rows),
`density_controls.csv` (80 rows), `fit_and_tangent_checks.csv` (5 508 fit
records + 48 infinitesimal checks), `representative_states.npz` (saved
networks/states, map provenance included, expected reference values), and
`manifest.json` (schema, units, SHA-256 of every file, source hashes).
`strain_curves.csv` is one addition to the PRD's 8-file sketch; it is what
makes every fit in the paper independently recomputable without simulation.

## Scientific assumptions and limitations

Athermal, zero-temperature minimisation; permanent bonds (frozen graph);
every node fixed to the dilated sphere (no bending, pressure, fluidity,
diffusion, pore formation or rupture); zero-stress reference state; two
illustrative stiffness sets (A: 60/300, B: 210/720 mN/m) that are *not*
calibrated plant-EV measurements; `K_rel` is an operational energy curvature
from the zero-slope F2 fit over ε_s ≤ 0.02 (sphere-normalised), not a thermally
sampled modulus; `L` is an operational difference, not measured dissipation.
D2 uses n = 3 parents per architecture (conditional intervals); D1-B matches
mean coordination but changes physical domain size — neither control is a
continuum mesh-convergence test. Full statements in the manuscript's
“Model and methods” and SI.

## Data availability

This repository contains the code needed for Tier-1 (all figures/tables)
and representative Tier-2 checks. The **full raw ensemble** (~700 MB: per-map
`network.npz`, `relaxed_positions.npz`, strain curves for all 294 saved
realisations) is **not publicly accessible**. Users can rerun the reproducible 
simulation to generate the dataset.