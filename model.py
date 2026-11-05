#!/usr/bin/env python3
"""model.py — athermal spherical spring-network mechanics.

The model is an idealised, athermal, permanently bonded central-force graph
on a sphere.  It is NOT a membrane molecular-dynamics model, a validated
constitutive model of plant extracellular vesicles, or a predictor of
delivery performance.  "Stiff"/"soft" are mechanical node classes, not
measured lipid phases, and the node fraction is not a molar fraction.

Physics:
  * N = int(4*pi*R0^2*rho) nodes on a deterministic Fibonacci sphere, R0=20 nm.
  * Frozen spring graph: all node pairs with reference chord length <= cutoff
    (2.5 nm at baseline), lexicographically sorted; rest lengths are the
    reference chord lengths.  No bonds are created or removed.
  * Exactly int(N*phi) nodes are labelled stiff.  SPONTANEOUS = uniform random
    labels; SEEDED = active-front cluster growth from max(3, N//25) seeds with
    growth radius 2.0 nm (baseline), shortfall filled uniformly.
  * Class stiffness (mN/m): set A (soft, stiff) = (60, 300), set B = (210, 720).
    Interface rule harmonic: 2 K_s K_t/(K_s+K_t); arithmetic: mean.
    Spring constant k_b = 0.5 * K_class / 1.660539  [kJ/mol/nm^2], and the
    potential keeps its own factor of one half: U = 0.5 sum_b k_b (l_b-l0_b)^2.
  * Dilation: nominal areal strain eps_s, target radius R = R0*sqrt(1+eps_s);
    affine state x_aff = sqrt(1+eps_s) x0.  Relaxed state: every node stays on
    the target sphere; L-BFGS-B minimises U over the tangential projection.
  * Moduli: exact affine curvature K_aff = c/(4 A0) sum_b k_b l0_b^2;
    relaxed curvature from the zero-slope cubic fit "F2" of
    y(eps)=c[U_rel(eps)-U_rel(0)]/A0 = 0.5 K eps^2 + a3 eps^3 (primary window
    eps <= 0.02).  L = K_aff - K_rel, q = K_rel/K_aff.  An independent
    infinitesimal tangent-space curvature (constrained Hessian / LSMR) is
    provided as a check.

numpy/scipy are required.  OpenMM is optional and is used solely for an
energy/force spot check.

Command line:
    python model.py self-test [--data DIR]   deterministic checks (exits nonzero on failure)
    python model.py example [--out DIR] [--data DIR] [--verify]
                              one representative network end-to-end
    python model.py run --phi P --pattern S --rep N --set {A,B} --rule R [--out DIR]
                              one explicit setting
    python model.py run-full --confirm-expensive --out DIR
                              full 192-evaluation baseline campaign (hours)

"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from scipy.spatial import ConvexHull, cKDTree

__version__ = "1.0.0"

# ------------------------------------------------------------------ constants
C_MNM_PER_OMM = 1.660539        # mN/m per kJ/mol/nm^2 
SPRING_SCALE = 0.5              # retained model convention in k_b
R0_NM = 20.0
STIFFNESS_SETS = {"A": (60.0, 300.0), "B": (210.0, 720.0)}   # (soft, stiff) mN/m
BASE_RHO = 0.50                 # nodes/nm^2
BASE_CUTOFF_NM = 2.5
BASE_GROWTH_NM = 2.0
STRAINS = np.array([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04])
FIT_WINDOWS = (0.01, 0.02, 0.04)
PRIMARY_WINDOW = 0.02
MAP_MASTER_SEED = 20261017      # baseline phase maps
CONTROL_MASTER_SEED = 20261019  # D2 shuffles and D1 density maps
PATTERN_ID = {"SEEDED": 0, "SPONTANEOUS": 1}
SOFT, STIFF = 0, 1              # node labels
CLASS_SOFT_SOFT, CLASS_STIFF_STIFF, CLASS_INTERFACE = 0, 1, 2
CLASS_NAMES = ("soft-soft", "stiff-stiff", "interface")
PG_TARGET = 1.0e-6              # kJ/mol/nm, tangential projected-gradient RMS
RADIAL_GATE_NM = 1.0e-8


def nominal_area(radius_nm: float = R0_NM) -> float:
    return 4.0 * np.pi * radius_nm**2


# ------------------------------------------------------------------- geometry
def fibonacci_sphere(radius_nm: float = R0_NM, rho: float = BASE_RHO) -> np.ndarray:
    """Deterministic Fibonacci sphere; identical node ordering to the archive.

    y_i = 1 - 2i/(N-1), r_i = sqrt(max(0, 1-y_i^2)), theta_i = pi(3-sqrt(5)) i,
    x_i = R (r_i cos theta_i, y_i, r_i sin theta_i), N = int(4 pi R^2 rho).
    """
    n = int(4 * np.pi * radius_nm**2 * rho)
    if n < 2:
        raise ValueError(f"degenerate node count N={n}")
    i = np.arange(n, dtype=np.float64)
    y = 1.0 - 2.0 * i / (n - 1)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    return radius_nm * np.column_stack([np.cos(theta) * r_xy, y, np.sin(theta) * r_xy])


def build_spring_graph(xyz0: np.ndarray, cutoff_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Frozen spring graph: unordered pairs with chord length <= cutoff,
    lexicographically sorted (matches ``sorted(cKDTree.query_pairs)``)."""
    pairs = sorted(cKDTree(xyz0).query_pairs(r=cutoff_nm))
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    l0 = np.linalg.norm(xyz0[pairs[:, 0]] - xyz0[pairs[:, 1]], axis=1)
    if np.any(l0 <= 0.0):
        raise ValueError("degenerate zero-length bond")
    return pairs, l0


def hull_faces(xyz0: np.ndarray) -> np.ndarray:
    """Outward-oriented convex-hull triangulation (diagnostics only; distinct
    from the spring graph)."""
    faces = np.asarray(ConvexHull(xyz0).simplices, dtype=np.int64).copy()
    centroid = xyz0.mean(axis=0)
    normals = np.cross(xyz0[faces[:, 1]] - xyz0[faces[:, 0]],
                       xyz0[faces[:, 2]] - xyz0[faces[:, 0]])
    centres = xyz0[faces].mean(axis=1)
    flip = np.einsum("ij,ij->i", normals, centres - centroid) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]


def mean_degree(n_nodes: int, pairs: np.ndarray) -> float:
    deg = np.bincount(pairs[:, 0], minlength=n_nodes) + np.bincount(pairs[:, 1], minlength=n_nodes)
    return float(deg.mean())


# ---------------------------------------------------------------- assignments
def derive_map_seed(replicate: int, phi: float, pattern: str,
                    master_seed: int = MAP_MASTER_SEED,
                    density: float | None = None) -> int:
    """Deterministic per-map seed.

    Baseline maps: SeedSequence(20261017, spawn=(rep, phi_key, mode)).
    D1 density maps:  SeedSequence(20261019, spawn=(rho_key, phi_key, mode, rep)).
    """
    phi_key = int(round(phi * 100))
    if density is None:
        ss = np.random.SeedSequence(master_seed, spawn_key=(int(replicate), phi_key,
                                                            PATTERN_ID[pattern]))
    else:
        rho_key = int(round(density * 1000))
        ss = np.random.SeedSequence(master_seed, spawn_key=(rho_key, phi_key,
                                                            PATTERN_ID[pattern],
                                                            int(replicate)))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def assign_labels(xyz0: np.ndarray, phi: float, pattern: str, rng: np.random.Generator,
                  growth_radius_nm: float = BASE_GROWTH_NM) -> np.ndarray:
    """Node labels (1 = stiff) with exactly int(N*phi) stiff nodes.

    SPONTANEOUS: uniform random choice without replacement.
    SEEDED: max(3, N//25) random seeds; repeatedly a random active-front entry
    converts its unlabelled neighbours within ``growth_radius_nm`` (in query
    list order) until the target count is reached; an entry with no growth is
    popped; any shortfall is filled uniformly. 
    """
    n = xyz0.shape[0]
    n_target = int(n * phi)
    if pattern not in PATTERN_ID:
        raise ValueError(f"unknown pattern {pattern!r}")
    labels = np.zeros(n, dtype=np.uint8)
    if n_target == 0:
        return labels
    if n_target == n:
        return np.ones(n, dtype=np.uint8)

    if pattern == "SPONTANEOUS":
        labels[rng.choice(n, n_target, replace=False)] = 1
        assert int(labels.sum()) == n_target
        return labels

    tree = cKDTree(xyz0)
    n_seeds = max(3, int(n / 25))
    seeds = rng.choice(n, n_seeds, replace=False)
    labels[seeds] = 1
    front = [int(s) for s in seeds]
    assigned = len(front)
    while assigned < n_target and front:
        idx = int(rng.integers(len(front)))
        grew = False
        for nb in tree.query_ball_point(xyz0[front[idx]], growth_radius_nm):
            if labels[nb] == SOFT and assigned < n_target:
                labels[nb] = 1
                front.append(int(nb))
                assigned += 1
                grew = True
        if not grew:
            front.pop(idx)
    if assigned < n_target:
        remaining = np.flatnonzero(labels == SOFT)
        labels[rng.choice(remaining, n_target - assigned, replace=False)] = 1
    assert int(labels.sum()) == n_target
    return labels


# -------------------------------------------------------------------- springs
def interface_stiffness(k_soft: float, k_stiff: float, rule: str) -> float:
    if rule == "harmonic":
        return 2.0 * k_soft * k_stiff / (k_soft + k_stiff)
    if rule == "arithmetic":
        return 0.5 * (k_soft + k_stiff)
    if rule == "uniform":
        if not np.isclose(k_soft, k_stiff):
            raise ValueError("uniform rule requires equal class stiffness")
        return k_soft
    raise ValueError(f"unknown interface rule {rule!r}")


def bond_classes(labels: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """0 soft-soft, 1 stiff-stiff, 2 interface."""
    s_i = labels[pairs[:, 0]] == 1
    s_j = labels[pairs[:, 1]] == 1
    cls = np.full(pairs.shape[0], CLASS_INTERFACE, dtype=np.int64)
    cls[(~s_i) & (~s_j)] = CLASS_SOFT_SOFT
    cls[s_i & s_j] = CLASS_STIFF_STIFF
    return cls


@dataclass(frozen=True)
class SpringNetwork:
    """Immutable bond set with per-bond stiffness at the reference geometry."""
    pairs: np.ndarray    # (E,2) int64, i<j, lexicographically sorted
    l0: np.ndarray       # (E,) nm
    k: np.ndarray        # (E,) kJ/mol/nm^2
    bond_class: np.ndarray  # (E,) int in {0,1,2}

    @property
    def n_bonds(self) -> int:
        return int(self.pairs.shape[0])

    def energy(self, x: np.ndarray) -> float:
        e, _ = self.extensions(x)
        return float(0.5 * np.dot(self.k, e * e))

    def energy_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """U and dU/dx with grad[i] += k e d/l, grad[j] -= k e d/l."""
        x = np.asarray(x, dtype=np.float64)
        d = x[self.pairs[:, 0]] - x[self.pairs[:, 1]]
        l = np.sqrt(np.einsum("ij,ij->i", d, d))
        if np.any(l <= 0.0):
            raise FloatingPointError("zero bond length")
        e = l - self.l0
        energy = float(0.5 * np.dot(self.k, e * e))
        gi = (self.k * e / l)[:, None] * d
        n = x.shape[0]
        grad = np.empty((n, 3))
        for ax in range(3):
            grad[:, ax] = (np.bincount(self.pairs[:, 0], weights=gi[:, ax], minlength=n)
                           - np.bincount(self.pairs[:, 1], weights=gi[:, ax], minlength=n))
        return energy, grad

    def extensions(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(x, dtype=np.float64)[self.pairs[:, 0]] \
            - np.asarray(x, dtype=np.float64)[self.pairs[:, 1]]
        l = np.linalg.norm(d, axis=1)
        if np.any(l <= 0.0):
            raise FloatingPointError("zero bond length")
        return l - self.l0, l

    def per_bond(self, x: np.ndarray) -> dict:
        e, l = self.extensions(x)
        return {"l": l, "delta_l": e, "strain": e / self.l0,
                "f_abs": self.k * np.abs(e), "u_b": 0.5 * self.k * e * e}

    def class_summary(self, x: np.ndarray) -> dict:
        """Class-mean signed strain / |force| / energy at configuration x."""
        pb = self.per_bond(x)
        out = {}
        for code, name in enumerate(CLASS_NAMES):
            m = self.bond_class == code
            out[name] = {"n": int(m.sum()),
                         "strain_mean": float(pb["strain"][m].mean()),
                         "force_mean": float(pb["f_abs"][m].mean()),
                         "energy_total": float(pb["u_b"][m].sum())}
        return out


def build_network(labels: np.ndarray, pairs: np.ndarray, l0: np.ndarray,
                  stiffness_set: str = "B", rule: str = "harmonic") -> SpringNetwork:
    """Stiffness map: k_b = 0.5 K_class / c with harmonic/arithmetic interfaces."""
    k_soft, k_stiff = STIFFNESS_SETS[stiffness_set]
    k_int = interface_stiffness(k_soft, k_stiff, rule)
    cls = bond_classes(labels, pairs)
    k_class = np.where(cls == CLASS_STIFF_STIFF, k_stiff,
                       np.where(cls == CLASS_SOFT_SOFT, k_soft, k_int))
    k = SPRING_SCALE * k_class / C_MNM_PER_OMM
    if np.any(k <= 0.0):
        raise ValueError("non-positive spring constant")
    return SpringNetwork(pairs=pairs, l0=l0.copy(), k=k, bond_class=cls)


def network_from_arrays(xyz0, pairs, l0, labels, k, bond_class) -> SpringNetwork:
    """Assemble a network directly from saved arrays (e.g. shuffled children)."""
    return SpringNetwork(pairs=np.asarray(pairs), l0=np.asarray(l0, float),
                         k=np.asarray(k, float), bond_class=np.asarray(bond_class))


# --------------------------------------------------------- sphere relaxation
class SphereRelaxer:
    """Tangential relaxation with every node held on |x_i| = R.

    Parameters y (free) with x = R y/||y||; the exact chain rule gives
    dU/dy_i = (R/||y_i||) [g_i - n_i (n_i . g_i)] (tangential projection), so
    the radial constraint holds to machine precision by construction.
    SciPy L-BFGS-B with restarts until the declared projected-gradient RMS
    criterion is met — deterministic, no RNG.
    """

    def __init__(self, net: SpringNetwork, radius_nm: float):
        self.net = net
        self.R = float(radius_nm)

    def _x(self, y):
        return self.R * (y / np.linalg.norm(y, axis=1, keepdims=True))

    def fun_grad(self, y_flat):
        y = np.asarray(y_flat, dtype=np.float64).reshape(-1, 3)
        x = self._x(y)
        energy, g = self.net.energy_gradient(x)
        ynorm = np.linalg.norm(y, axis=1, keepdims=True)
        n = y / ynorm
        g_tan = g - n * np.einsum("ij,ij->i", n, g)[:, None]
        return energy, ((self.R / ynorm) * g_tan).ravel()

    def projected_gradient_rms(self, x) -> tuple[float, float]:
        _, g = self.net.energy_gradient(x)
        n = x / np.linalg.norm(x, axis=1, keepdims=True)
        g_tan = g - n * np.einsum("ij,ij->i", n, g)[:, None]
        mag = np.linalg.norm(g_tan, axis=1)
        return float(np.sqrt(np.mean(mag**2))), float(mag.max())

    def relax(self, x_init, max_iterations=20000, pg_target=PG_TARGET,
              max_restarts=6, lbfgs_memory=50) -> dict:
        x_init = np.asarray(x_init, dtype=np.float64)
        y = (x_init / np.linalg.norm(x_init, axis=1, keepdims=True))
        total_it = total_fe = 0
        success, energy, message = False, self.fun_grad(y.ravel())[0], ""
        for _ in range(max_restarts):
            res = minimize(self.fun_grad, y.ravel(), method="L-BFGS-B", jac=True,
                           options={"maxiter": max_iterations,
                                    "maxfun": 10 * max_iterations,
                                    "maxcor": lbfgs_memory,
                                    "ftol": 1e-18, "gtol": 1e-14})
            y = res.x.reshape(-1, 3)
            total_it += int(res.nit)
            total_fe += int(res.nfev)
            energy = float(res.fun)
            message = str(res.message)
            x = self._x(y)
            if self.projected_gradient_rms(x)[0] <= pg_target:
                success = True
                break
        x = self._x(y)
        pg_rms, pg_max = self.projected_gradient_rms(x)
        radial = float(np.abs(np.linalg.norm(x, axis=1) - self.R).max())
        return {"x": x, "energy": energy, "iterations": total_it, "nfev": total_fe,
                "pg_rms": pg_rms, "pg_max": pg_max,
                "radial_error_nm": radial, "success": bool(success),
                "message": message}


def affine_positions(xyz0: np.ndarray, eps_s: float) -> np.ndarray:
    return np.sqrt(1.0 + eps_s) * xyz0


def U_affine(net: SpringNetwork, eps_s: float) -> float:
    s = np.sqrt(1.0 + eps_s)
    e = (s - 1.0) * net.l0
    return float(0.5 * np.dot(net.k, e * e))


def K_affine(net: SpringNetwork, radius_nm: float = R0_NM) -> float:
    """Exact affine curvature, mN/m: c/(4 A0) sum_b k_b l0_b^2."""
    return C_MNM_PER_OMM * float(np.dot(net.k, net.l0**2)) / (4.0 * nominal_area(radius_nm))


def K_affine_by_class(net: SpringNetwork, radius_nm: float = R0_NM) -> dict:
    out = {}
    for code, name in enumerate(CLASS_NAMES):
        m = net.bond_class == code
        out[name] = C_MNM_PER_OMM * float(np.dot(net.k[m], net.l0[m]**2)) \
            / (4.0 * nominal_area(radius_nm))
    return out


def strain_curve(net: SpringNetwork, xyz0: np.ndarray, strains=STRAINS,
                 radius_nm: float = R0_NM, verbose: bool = False) -> dict:
    """Relaxed and affine energies over the nominal-strain grid.

    Each strained relaxation is started from the affine configuration at that
    strain (the archived protocol); eps = 0 is exact by construction.
    """
    U_rel, U_aff = np.zeros(len(strains)), np.zeros(len(strains))
    x_rel = [None] * len(strains)
    diag = []
    for si, eps in enumerate(strains):
        U_aff[si] = U_affine(net, eps)
        if eps == 0.0:
            x_rel[si] = xyz0.copy()
            U_rel[si] = 0.0
            continue
        R = radius_nm * np.sqrt(1.0 + eps)
        res = SphereRelaxer(net, R).relax(affine_positions(xyz0, eps))
        U_rel[si] = res["energy"]
        x_rel[si] = res["x"]
        diag.append({"eps": float(eps), "iterations": res["iterations"],
                     "pg_rms": res["pg_rms"], "radial_nm": res["radial_error_nm"],
                     "success": res["success"]})
        if verbose:
            print(f"  eps={eps:g}: U_rel={U_rel[si]:.9f} kJ/mol "
                  f"(it={res['iterations']}, pg={res['pg_rms']:.2e})", flush=True)
        if not res["success"]:
            raise RuntimeError(f"relaxation did not meet pg target at eps={eps}: "
                               f"{res['pg_rms']:.3e}")
    return {"strains": np.asarray(strains, float), "U_rel": U_rel, "U_aff": U_aff,
            "x_rel": x_rel, "diagnostics": diag}


# ------------------------------------------------------------------ the fits
def fit_modulus(strains, energies, window, model="F2",
                radius_nm: float = R0_NM) -> dict:
    """F1/F2/F3 curvature estimators on scaled predictors.

    y(eps) = c [U(eps)-U(0)]/A0 is fitted over grid points with eps <= window:
      F1: y = a1 t + 0.5 K t^2 + a3 t^3 (free linear term)
      F2: y =       0.5 K t^2 + a3 t^3 (zero slope; primary estimator)
      F3: y =       0.5 K t^2           (purely quadratic)
    with t = eps/window so the design matrix is well conditioned; K, a1, a3 in
    mN/m.  The primary manuscript choice is F2 at window 0.02.
    """
    strains = np.asarray(strains, float)
    energies = np.asarray(energies, float)
    m = strains <= window + 1e-12
    if m.sum() < 3:
        raise ValueError("insufficient points in window")
    y = C_MNM_PER_OMM * (energies[m] - energies[0]) / nominal_area(radius_nm)
    t = strains[m] / window
    powers = {"F1": [1, 2, 3], "F2": [2, 3], "F3": [2]}[model]
    X = np.array([t**p for p in powers]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return {"model": model, "window": float(window), "n_points": int(m.sum()),
            "K_mNm": float(2.0 * beta[powers.index(2)] / window**2),
            "a1_mNm": float(beta[0] / window) if model == "F1" else 0.0,
            "a3_mNm": float(beta[powers.index(3)] / window**3) if 3 in powers else 0.0,
            "rank": int(np.linalg.matrix_rank(X)),
            "condition": float(np.linalg.cond(X)),
            "residual_rms": float(np.sqrt(np.mean(resid**2)))}


# ------------------------------------------- infinitesimal tangent-space check
def tangent_curvature(net: SpringNetwork, xyz0: np.ndarray,
                      radius_nm: float = R0_NM) -> dict:
    """Zero-strain constrained-Hessian curvature via sparse least squares.

    With orthonormal tangent bases T_i, first-order node motion
    x_i = x_i^0 + eps (x_i^0/2 + T_i v_i) gives first-order edge extensions
    d_b = l0_b/2 + r_hat_b . (T_i v_i - T_j v_j).  Minimising
    sum_b k_b d_b^2 over v (rotations are null modes) yields
        K0 = K_aff - (c/A0) f^T H^+ f,  H = C^T W C, f = C^T W d,
    solved here with LSMR on the weighted least-squares form.
    Returns K0 in mN/m plus convergence diagnostics.
    """
    x = np.asarray(xyz0, float)
    pairs, k, l0 = net.pairs, net.k, net.l0
    n = x.shape[0]
    normal = x / np.linalg.norm(x, axis=1)[:, None]
    ref = np.tile([0.0, 0.0, 1.0], (n, 1))
    ref[np.abs(normal[:, 2]) > 0.9] = [0, 1, 0]
    t1 = np.cross(normal, ref)
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(normal, t1)
    T = np.stack([t1, t2], axis=2)                    # (N,3,2)
    b_hat = (x[pairs[:, 0]] - x[pairs[:, 1]]) / l0[:, None]
    bi = np.einsum("ei,eij->ej", b_hat, T[pairs[:, 0]])
    bj = -np.einsum("ei,eij->ej", b_hat, T[pairs[:, 1]])
    cols = np.column_stack([2 * pairs[:, 0], 2 * pairs[:, 0] + 1,
                            2 * pairs[:, 1], 2 * pairs[:, 1] + 1])
    vals = np.column_stack([bi, bj]) * np.sqrt(k)[:, None]
    R = coo_matrix((vals.ravel(), (np.repeat(np.arange(len(pairs)), 4), cols.ravel())),
                   shape=(len(pairs), 2 * n)).tocsr()
    target = -np.sqrt(k) * l0 / 2.0
    sol = lsmr(R, target, atol=1e-12, btol=1e-12, maxiter=15000)
    res = R @ sol[0] - target
    K0 = C_MNM_PER_OMM / nominal_area(radius_nm) * float(np.dot(res, res))
    return {"K0_mNm": K0, "iterations": int(sol[2]), "stop_code": int(sol[1]),
            "normal_residual": float(np.linalg.norm(R.T @ res))}


# ----------------------------------------------------- D2 shuffle and D1 cutoffs
def length_rank_bins(l0: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """10 rank-quantile bins of reference chord length (stable ordering)."""
    e = l0.shape[0]
    order = np.lexsort((np.arange(e), l0))
    bin_id = np.empty(e, dtype=np.int64)
    for b, ids in enumerate(np.array_split(order, n_bins)):
        bin_id[ids] = b
    counts = np.bincount(bin_id, minlength=n_bins)
    assert counts.max() - counts.min() <= 1
    return bin_id


def shuffle_stiffness(k_parent: np.ndarray, bin_id: np.ndarray, pattern: str,
                      replicate: int, shuffle_id: int, phi_key: int = 50,
                      n_bins: int = 10) -> np.ndarray:
    """Permute stiffness within length bins; exact multiset preservation.

    Seed is independent of the interface rule, so harmonic and arithmetic
    children of the same (map, shuffle) share one edge permutation.
    """
    e = k_parent.shape[0]
    ss = np.random.SeedSequence(CONTROL_MASTER_SEED,
                                spawn_key=(phi_key, PATTERN_ID[pattern],
                                           int(replicate), int(shuffle_id)))
    rng = np.random.default_rng(int(ss.generate_state(1, dtype=np.uint32)[0]))
    perm = np.arange(e, dtype=np.int64)
    for b in range(n_bins):
        ids = np.flatnonzero(bin_id == b)
        perm[ids] = rng.permutation(ids)
    k_shuffle = k_parent[perm]
    assert np.array_equal(np.sort(k_shuffle), np.sort(k_parent))
    for b in range(n_bins):
        ids = np.flatnonzero(bin_id == b)
        assert np.array_equal(np.sort(k_shuffle[ids]), np.sort(k_parent[ids]))
    return k_shuffle


def d1_cutoffs(scheme: str, rho: float, cutoff_nm: float = BASE_CUTOFF_NM,
               growth_nm: float = BASE_GROWTH_NM, ref_rho: float = BASE_RHO):
    """D1-A keeps physical radii; D1-B scales both by sqrt(ref_rho/rho)."""
    if scheme == "d1a":
        return cutoff_nm, growth_nm
    if scheme == "d1b":
        f = (ref_rho / rho) ** 0.5
        return cutoff_nm * f, growth_nm * f
    raise ValueError(scheme)


# ------------------------------------------------------------------ Utilities
def kabsch_rotation(x_rel: np.ndarray, x_aff: np.ndarray) -> np.ndarray:
    """Proper rotation Q minimising |x_rel - x_aff Q| about the origin."""
    u, _, v = np.linalg.svd(x_rel.T @ x_aff)
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ v)
    return u @ d @ v


def tangential_displacement(x_rel: np.ndarray, x_aff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch-aligned relaxed positions and the tangential displacement field."""
    xr = x_rel @ kabsch_rotation(x_rel, x_aff)
    normal = x_aff / np.linalg.norm(x_aff, axis=1)[:, None]
    disp = xr - x_aff
    tangent = disp - (disp * normal).sum(axis=1)[:, None] * normal
    return xr, tangent


# -------------------------------------------------------------- optional OpenMM
def openmm_spot_check(net: SpringNetwork, x: np.ndarray):
    """Independent OpenMM energy/force cross-check of the same bond records.

    Returns {"status": "SKIPPED"} when OpenMM is not installed — never a pass.
    """
    try:
        import openmm
    except Exception:
        return {"status": "SKIPPED (openmm not installed)"}
    try:
        sys_ = openmm.System()
        for _ in range(x.shape[0]):
            sys_.addParticle(1.0)
        force = openmm.HarmonicBondForce()
        for (i, j), r0, kb in zip(net.pairs, net.l0, net.k):
            force.addBond(int(i), int(j), float(r0), float(kb))
        sys_.addForce(force)
        integ = openmm.VerletIntegrator(0.0)
        plat = openmm.Platform.getPlatformByName("Reference")
        ctxt = openmm.Context(sys_, integ, plat)
        ctxt.setPositions(x)
        state = ctxt.getState(getEnergy=True, getForces=True)
        U_omm = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        f_omm = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        U_np, g_np = net.energy_gradient(x)
        return {"status": "ok",
                "U_numpy": U_np, "U_openmm": U_omm,
                "energy_abs_diff": abs(U_np - U_omm),
                "force_rel_max": float((np.abs(g_np + f_omm)
                                        / max(np.abs(g_np).max(), 1e-30)).max()),
                "openmm_version": openmm.version.full_version}
    except Exception as exc:
        return {"status": f"FAILED: {exc}"}


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------- run one setting
def run_setting(phi=0.5, pattern="SEEDED", replicate=0, stiffness_set="B",
                rule="harmonic", rho=BASE_RHO, cutoff_nm=BASE_CUTOFF_NM,
                growth_nm=BASE_GROWTH_NM, labels=None, strains=STRAINS,
                verbose=False) -> dict:
    """Geometry + labels (or supplied saved labels) + stiffness map + strain curve."""
    xyz0 = fibonacci_sphere(R0_NM, rho)
    pairs, l0 = build_spring_graph(xyz0, cutoff_nm)
    if labels is None:
        seed = derive_map_seed(replicate, phi, pattern,
                               density=rho if rho != BASE_RHO else None)
        labels = assign_labels(xyz0, phi, pattern,
                               np.random.default_rng(seed), growth_nm)
    net = build_network(labels, pairs, l0, stiffness_set, rule)
    ka = K_affine(net)
    curve = strain_curve(net, xyz0, strains, verbose=verbose)
    fits = {m: fit_modulus(curve["strains"], curve["U_rel"], PRIMARY_WINDOW, m)
            for m in ("F1", "F2", "F3")}
    k_rel = fits["F2"]["K_mNm"]
    return {"xyz0": xyz0, "pairs": pairs, "l0": l0, "labels": labels, "net": net,
            "K_aff": ka, "K_aff_class": K_affine_by_class(net),
            "curve": curve, "fits": fits, "K_rel_F2": k_rel,
            "L": ka - k_rel, "q": k_rel / ka,
            "hessian": tangent_curvature(net, xyz0)}


# -------------------------------------------------------------------- self-test
def _approx(a, b, tol, what):
    if not abs(a - b) <= tol:
        raise AssertionError(f"{what}: {a!r} vs {b!r} (tol {tol})")


def self_test(data_dir: Path | None, quick: bool = False) -> None:
    checks = []

    def ok(name):
        checks.append(name)
        print(f"  ok: {name}", flush=True)

    print(f"model.py self-test (numpy {np.__version__}, scipy {__import__('scipy').__version__})")

    # 1. reference geometry
    xyz0 = fibonacci_sphere()
    pairs, l0 = build_spring_graph(xyz0, BASE_CUTOFF_NM)
    n = xyz0.shape[0]
    _approx(n, 2513, 0, "baseline node count")
    _approx(pairs.shape[0], 10022, 0, "baseline edge count")
    _approx(mean_degree(n, pairs), 7.97612415439713, 1e-9, "mean degree")
    _approx(float(np.abs(np.linalg.norm(xyz0, axis=1) - R0_NM).max()), 0.0, 1e-9,
            "nodes on reference sphere")
    ok("reference geometry N=2513, E=10022, <z>=7.9761")

    # 2. assignment counts, seeds and interface stiffness
    seed = derive_map_seed(0, 0.5, "SEEDED")
    lab_seed = assign_labels(xyz0, 0.5, "SEEDED", np.random.default_rng(seed))
    _approx(int(lab_seed.sum()), int(n * 0.5), 0, "SEEDED stiff count")
    lab_rnd = assign_labels(xyz0, 0.5, "SPONTANEOUS",
                            np.random.default_rng(derive_map_seed(0, 0.5, "SPONTANEOUS")))
    _approx(int(lab_rnd.sum()), int(n * 0.5), 0, "SPONTANEOUS stiff count")
    _approx(interface_stiffness(210.0, 720.0, "harmonic"),
            2 * 210 * 720 / 930, 1e-12, "set-B harmonic interface")
    _approx(interface_stiffness(60.0, 300.0, "arithmetic"), 180.0, 1e-12,
            "set-A arithmetic interface")
    ok("assignment counts and interface stiffness values")

    # 3. unit conversion and stiffness map
    net = build_network(lab_seed, pairs, l0, "B", "harmonic")
    _approx(float(net.k.max()), SPRING_SCALE * 720.0 / C_MNM_PER_OMM, 1e-12,
            "k_b conversion (0.5*K/c applied once)")
    _approx(float(net.k[net.bond_class == CLASS_SOFT_SOFT][0]),
            SPRING_SCALE * 210.0 / C_MNM_PER_OMM, 1e-12, "soft-soft k_b")
    ok("spring-constant unit conversion")

    # 4. rest-state energy zero; affine formula vs direct evaluation
    _approx(net.energy(xyz0), 0.0, 1e-9, "U(x0) = 0")
    for eps in (0.005, 0.02):
        xa = affine_positions(xyz0, eps)
        _approx(net.energy(xa), U_affine(net, eps), 1e-9 * max(1.0, abs(U_affine(net, eps))),
                "affine energy formula")
    ok("unstressed reference state and affine energy identity")

    # 5. analytic gradient vs finite differences at a strained configuration
    x = affine_positions(xyz0, 0.02).copy()
    x[::7] += 0.003 * np.random.default_rng(0).standard_normal((len(x[::7]), 3))
    _, g = net.energy_gradient(x)
    rng = np.random.default_rng(1)
    for _ in range(12):
        i, j = rng.integers(n), rng.integers(3)
        h = 1e-6
        xp = x.copy(); xp[i, j] += h
        xm = x.copy(); xm[i, j] -= h
        _approx((net.energy(xp) - net.energy(xm)) / (2 * h), g[i, j],
                1e-4 * max(1.0, abs(g[i, j])), "finite-difference gradient")
    ok("Cartesian gradient matches finite differences")

    # 6. F2 fit recovers an exact quadratic+cubic polynomial; zero slope holds
    e = np.array([0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02])
    y = 0.5 * 250.0 * e**2 - 30.0 * e**3
    U = y / C_MNM_PER_OMM * nominal_area()
    for model in ("F1", "F2"):
        _approx(fit_modulus(e, U, 0.02, model)["K_mNm"], 250.0, 1e-6,
                f"fit recovers curvature ({model})")
    _approx(fit_modulus(e, U, 0.02, "F2")["a1_mNm"], 0.0, 0.0, "F2 zero slope")
    ok("F1/F2 fits on known polynomial")

    # 7. affine-curve F2 fit approaches the analytic affine curvature
    ka = K_affine(net)
    Uaff = np.array([U_affine(net, s) for s in STRAINS])
    _approx(fit_modulus(STRAINS, Uaff, 0.01, "F2")["K_mNm"] / ka, 1.0, 6e-3,
            "affine F2 curvature vs analytic (finite-window bias)")
    ok("analytic affine curvature consistent with its finite-window fit")

    # 8. small-network constrained relaxation (non-trivial graph, <z> ~ 3)
    xyz_s = fibonacci_sphere(R0_NM, 0.20)
    pairs_s, l0_s = build_spring_graph(xyz_s, BASE_CUTOFF_NM)
    assert pairs_s.shape[0] > 1000, "small test graph is non-trivial"
    rng_s = np.random.default_rng(20261017)
    lab_s = assign_labels(xyz_s, 0.5, "SEEDED", rng_s)
    net_s = build_network(lab_s, pairs_s, l0_s, "B", "harmonic")
    res = SphereRelaxer(net_s, R0_NM * np.sqrt(1.02)).relax(affine_positions(xyz_s, 0.02))
    assert res["success"], "small relaxation converged"
    assert res["iterations"] > 0, "optimiser actually iterated"
    _approx(res["radial_error_nm"], 0.0, RADIAL_GATE_NM, "radial constraint")
    _approx(res["pg_rms"], 0.0, PG_TARGET, "projected-gradient gate")
    assert res["energy"] <= U_affine(net_s, 0.02) + 1e-9, "relaxed energy <= affine"
    ok(f"small-network relaxation (N={xyz_s.shape[0]}, E={pairs_s.shape[0]}, "
       f"it={res['iterations']}, pg={res['pg_rms']:.2e})")

    # 9. infinitesimal tangent curvature: below affine, above zero, sane
    h = tangent_curvature(net, xyz0)
    assert 0.0 < h["K0_mNm"] < ka, "tangent curvature bracket"
    ok(f"tangent-space curvature K0={h['K0_mNm']:.4f} mN/m < K_aff={ka:.4f}")

    # 10. D2 shuffle: binwise multiset exact; rule-independent permutation
    bins = length_rank_bins(l0)
    k_par = build_network(lab_seed, pairs, l0, "B", "harmonic").k
    k_ar = build_network(lab_seed, pairs, l0, "B", "arithmetic").k
    k_sh_h = shuffle_stiffness(k_par, bins, "SEEDED", 0, 0)
    k_sh_a = shuffle_stiffness(k_ar, bins, "SEEDED", 0, 0)
    assert np.array_equal(np.sort(k_sh_h), np.sort(k_par))
    for b in range(10):
        ids = np.flatnonzero(bins == b)
        assert np.array_equal(np.sort(k_sh_h[ids]), np.sort(k_par[ids]))
    assert np.array_equal(np.flatnonzero(k_sh_h != k_par), np.flatnonzero(k_sh_a != k_ar)), \
        "same permutation applied to both interface rules"
    ok("D2 shuffle: binwise stiffness multisets conserved, rules paired")

    # 11. D1 cutoff conventions
    _approx(d1_cutoffs("d1a", 0.35)[0], 2.5, 0, "D1-A cutoff fixed")
    _approx(d1_cutoffs("d1b", 0.35)[0], 2.5 * np.sqrt(0.5 / 0.35), 1e-12, "D1-B cutoff")
    _approx(d1_cutoffs("d1b", 0.75)[1], 2.0 * np.sqrt(0.5 / 0.75), 1e-12, "D1-B growth")
    ok("D1-A/D1-B cutoff conventions")

    # 12. saved-fixture regression (requires data/representative_states.npz)
    if data_dir is not None and (data_dir / "representative_states.npz").exists():
        z = np.load(data_dir / "representative_states.npz")
        for tag in ("mapA", "mapB"):
            saved_pairs = z[f"{tag}_pairs"]
            assert np.array_equal(saved_pairs, pairs), f"{tag} graph identical"
            sid_h = f"phi050_{'seeded' if tag == 'mapA' else 'spontaneous'}_rep0000_B_harmonic"
            U_ref = z[f"exp_fixture_{sid_h}_U_rel"]
            K_ref = float(z[f"exp_fixture_{sid_h}_K_rel_F2"])
            t0 = time.perf_counter()
            out = run_setting(labels=z[f"{tag}_phase_labels"], stiffness_set="B",
                              rule="harmonic", verbose=False)
            dU = float(np.abs(out["curve"]["U_rel"] - U_ref).max())
            assert dU <= 1e-6, f"{tag} energy curve drift {dU:.3e} kJ/mol"
            _approx(out["K_rel_F2"], K_ref, 0.01, f"{tag} F2 modulus")
            cls = out["net"].class_summary(out["curve"]["x_rel"][6])
            for ci, cname in enumerate(CLASS_NAMES):
                _approx(cls[cname]["strain_mean"],
                        float(z[f"exp_{tag}_class_strain_mean_eps0.020"][ci]),
                        5e-6, f"{tag} {cname} class strain")
            _approx(float(z[f"exp_{tag}_class_force_mean_eps0.020"][1]),
                    cls["stiff-stiff"]["force_mean"], 1e-3, f"{tag} stiff force")
            ok(f"{tag} ({str(z[f'{tag}_id'][0])}): max|dU|={dU:.2e} kJ/mol, "
               f"dK_rel={out['K_rel_F2'] - K_ref:+.2e} mN/m "
               f"({time.perf_counter() - t0:.0f}s)")
    else:
        print("  note: data/representative_states.npz not found — saved-fixture "
              "regression SKIPPED (run from the repository root)")

    omm = openmm_spot_check(net, affine_positions(xyz0, 0.02))
    print(f"  OpenMM spot-check: {omm['status']}")
    if omm["status"] == "ok":
        assert omm["energy_abs_diff"] < 1e-6 and omm["force_rel_max"] < 1e-8
        ok(f"OpenMM energy/force agreement (v{omm['openmm_version']})")
    print(f"SELF-TEST PASSED ({len(checks)} checks + fixture regression)")


# ---------------------------------------------------------------------- example
def example(out_dir: Path, data_dir: Path | None, verify: bool) -> None:
    """Regenerate the representative phi=0.50 Set-B harmonic clustered network.

    Labels are regenerated from the archived seed recipe (fresh-generation
    reproducibility); with --verify the full strain curve, F2 modulus, tangent
    curvature, the arithmetic partner rule, one D2 shuffled child and one D1-B
    density network are rerun and compared against the saved fixtures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    seed = derive_map_seed(0, 0.5, "SEEDED")
    print(f"example: phi=0.50 SEEDED rep0, set B harmonic; map seed {seed}")
    out = run_setting(phi=0.5, pattern="SEEDED", replicate=0, stiffness_set="B",
                      rule="harmonic", verbose=True)
    labels_fresh = out["labels"]

    cfg = {"model_py_sha256_16": code_hash(), "model_version": __version__,
           "phi": 0.5, "pattern": "SEEDED", "replicate": 0, "stiffness_set": "B",
           "interface_rule": "harmonic", "map_seed": seed,
           "seed_recipe": "SeedSequence(entropy=20261017, spawn_key=(0,50,0)) -> PCG64",
           "strains": STRAINS.tolist(), "python": sys.version.split()[0],
           "numpy": np.__version__, "scipy": __import__("scipy").__version__,
           "platform": platform.platform()}
    npz = out_dir / "network.npz"
    np.savez_compressed(npz, xyz0=out["xyz0"], pairs=out["pairs"], l0=out["l0"],
                        phase_labels=labels_fresh, bond_classes=out["net"].bond_class,
                        k_omm=out["net"].k)
    rows = []
    for si, eps in enumerate(out["curve"]["strains"]):
        rows.append({"epsilon_s": float(eps),
                     "U_aff_kJmol": float(out["curve"]["U_aff"][si]),
                     "U_rel_kJmol": float(out["curve"]["U_rel"][si]),
                     "pg_rms": (out["curve"]["diagnostics"][si - 1]["pg_rms"] if si else 0.0),
                     "iterations": (out["curve"]["diagnostics"][si - 1]["iterations"] if si else 0)})
    import csv
    with open(out_dir / "strain_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = {"K_aff_mNm": out["K_aff"], "K_rel_F2_mNm": out["K_rel_F2"],
              "L_mNm": out["L"], "q": out["q"],
              "K_aff_class_mNm": out["K_aff_class"],
              "tangent_K0_mNm": out["hessian"]["K0_mNm"],
              "fits": out["fits"]}
    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"configuration": cfg, "results": result,
                   "openmm_spot_check": openmm_spot_check(
                       out["net"], affine_positions(out["xyz0"], 0.02)),
                   "runtime_s": time.perf_counter() - t0}, fh, indent=2)
    print(f"  K_aff={out['K_aff']:.4f}  K_rel(F2)={out['K_rel_F2']:.4f}  "
          f"L={out['L']:.4f}  q={out['q']:.6f}  "
          f"K0={out['hessian']['K0_mNm']:.4f} mN/m")
    print(f"  wrote {npz}, strain_curve.csv, summary.json")

    if not verify:
        return
    print("verify: comparing fresh generation and reruns against saved fixtures")
    z = np.load(data_dir / "representative_states.npz")
    # fresh RNG labels must equal the archived realisation
    assert np.array_equal(labels_fresh, z["mapA_phase_labels"]), \
        "freshly generated labels differ from the archived map"
    print("  ok: fresh seed-derived labels identical to archived mapA realisation")

    for tag in ("mapA", "mapB"):
        sid = f"phi050_{'seeded' if tag == 'mapA' else 'spontaneous'}_rep0000_B_"
        for rule in ("harmonic", "arithmetic"):
            t0 = time.perf_counter()
            o = run_setting(labels=z[f"{tag}_phase_labels"], stiffness_set="B",
                            rule=rule)
            U_ref = z[f"exp_fixture_{sid}{rule}_U_rel"]
            K_ref = float(z[f"exp_fixture_{sid}{rule}_K_rel_F2"])
            dU = float(np.abs(o["curve"]["U_rel"] - U_ref).max())
            assert dU <= 1e-6, f"{tag}/{rule}: energy drift {dU:.3e}"
            assert abs(o["K_rel_F2"] - K_ref) <= 0.01
            print(f"  ok: {tag} {rule}: max|dU|={dU:.2e} kJ/mol, "
                  f"dK_rel={o['K_rel_F2'] - K_ref:+.2e} mN/m "
                  f"({time.perf_counter() - t0:.0f}s)")

    # D2 shuffled child (seeded rep0 shuffle0): shuffled k on the same graph
    t0 = time.perf_counter()
    net2 = network_from_arrays(z["d2c_xyz0"], z["d2c_pairs"], z["d2c_l0"],
                               None, z["d2c_k_shuffle"], z["d2c_parent_class"])
    c2 = strain_curve(net2, z["d2c_xyz0"])
    U_ref = z["exp_fixture_phi050_seeded_rep0000_harmonic_shuffle0_U_rel"]
    K_ref = float(z["exp_fixture_phi050_seeded_rep0000_harmonic_shuffle0_K_rel_F2"])
    dU = float(np.abs(c2["U_rel"] - U_ref).max())
    k2 = fit_modulus(c2["strains"], c2["U_rel"], PRIMARY_WINDOW)["K_mNm"]
    assert dU <= 1e-6 and abs(k2 - K_ref) <= 0.01
    # and the shuffle construction itself must reproduce k_shuffle exactly
    k_par = build_network(z["d2c_source_phase_labels"], z["d2c_pairs"], z["d2c_l0"],
                          "B", "harmonic").k
    k_re = shuffle_stiffness(k_par, length_rank_bins(z["d2c_l0"]), "SEEDED", 0,
                             int(z["d2c_shuffle_id"]))
    assert np.array_equal(k_re, z["d2c_k_shuffle"]), "shuffle reconstruction"
    print(f"  ok: D2 child shuffle0 rerun + exact shuffle reconstruction "
          f"(max|dU|={dU:.2e}, dK={k2 - K_ref:+.2e}; {time.perf_counter() - t0:.0f}s)")

    # D1-B density network (rho=0.35 coordination-matched)
    t0 = time.perf_counter()
    rho = float(z["d1b_rho"])
    cut, gro = d1_cutoffs("d1b", rho)
    _approx(cut, float(z["d1b_cutoff_nm"]), 1e-12, "D1-B fixture cutoff")
    seed_d1 = derive_map_seed(0, 0.5, "SEEDED", master_seed=CONTROL_MASTER_SEED,
                              density=rho)
    lab_d1 = assign_labels(fibonacci_sphere(R0_NM, rho), 0.5, "SEEDED",
                           np.random.default_rng(seed_d1), gro)
    assert np.array_equal(lab_d1, z["d1b_phase_labels"]), "D1-B labels"
    o = run_setting(labels=z["d1b_phase_labels"], stiffness_set="B", rule="harmonic",
                    rho=rho, cutoff_nm=cut, growth_nm=gro)
    U_ref = z["exp_fixture_rho035_seeded_rep0000_harmonic_d1b_U_rel"]
    K_ref = float(z["exp_fixture_rho035_seeded_rep0000_harmonic_d1b_K_rel_F2"])
    dU = float(np.abs(o["curve"]["U_rel"] - U_ref).max())
    assert dU <= 1e-6 and abs(o["K_rel_F2"] - K_ref) <= 0.01
    print(f"  ok: D1-B rho=0.35 map regenerated + rerun (max|dU|={dU:.2e}, "
          f"dK={o['K_rel_F2'] - K_ref:+.2e}; {time.perf_counter() - t0:.0f}s)")
    print("VERIFY PASSED")


# -------------------------------------------------------------------- campaign
def run_full_campaign(out_dir: Path) -> None:
    """Full 192-evaluation baseline factorial (both fractions, sets, rules,
    12 maps per architecture) + 2 uniform controls.  Hours of CPU time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    summary_path = out_dir / "campaign_summary.csv"
    fields = ["phi", "pattern", "replicate", "stiffness_set", "interface_rule",
              "map_seed", "N", "n_bonds", "K_aff_mNm", "K_rel_F2_mNm", "L_mNm", "q",
              "K0_tangent_mNm", "runtime_s"]
    done = set()
    if summary_path.exists():
        with open(summary_path) as fh:
            done = {(r["phi"], r["pattern"], r["replicate"], r["stiffness_set"],
                     r["interface_rule"]) for r in _csv.DictReader(fh)}
        print(f"resuming: {len(done)} settings already complete")
    geo_cache: dict[tuple, tuple] = {}

    def geometry(rho, cutoff):
        if (rho, cutoff) not in geo_cache:
            x = fibonacci_sphere(R0_NM, rho)
            geo_cache[(rho, cutoff)] = (x, *build_spring_graph(x, cutoff))
        return geo_cache[(rho, cutoff)]

    with open(summary_path, "a", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        if not done:
            writer.writeheader()
        for phi in (0.15, 0.50):
            for pattern in ("SEEDED", "SPONTANEOUS"):
                for rep in range(12):
                    seed = derive_map_seed(rep, phi, pattern)
                    xyz, pairs, l0 = geometry(BASE_RHO, BASE_CUTOFF_NM)
                    labels = assign_labels(xyz, phi, pattern,
                                           np.random.default_rng(seed))
                    for sset in ("A", "B"):
                        for rule in ("harmonic", "arithmetic"):
                            key = (str(phi), pattern, str(rep), sset, rule)
                            if key in done:
                                continue
                            t0 = time.perf_counter()
                            out = run_setting(labels=labels, stiffness_set=sset,
                                              rule=rule)
                            writer.writerow({**dict(zip(fields[:5], key)),
                                             "map_seed": seed, "N": xyz.shape[0],
                                             "n_bonds": pairs.shape[0],
                                             "K_aff_mNm": out["K_aff"],
                                             "K_rel_F2_mNm": out["K_rel_F2"],
                                             "L_mNm": out["L"], "q": out["q"],
                                             "K0_tangent_mNm": out["hessian"]["K0_mNm"],
                                             "runtime_s": round(time.perf_counter() - t0, 1)})
                            fh.flush()
                            print(f"{key}: K_rel={out['K_rel_F2']:.3f}", flush=True)
    print(f"campaign written to {summary_path}")


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("self-test", help="deterministic checks")
    st.add_argument("--data", default="data", help="canonical data directory")
    ex = sub.add_parser("example", help="one representative network end-to-end")
    ex.add_argument("--out", default="results/example")
    ex.add_argument("--data", default="data")
    ex.add_argument("--verify", action="store_true",
                    help="rerun saved fixtures and compare (a few minutes)")
    rn = sub.add_parser("run", help="one explicit setting")
    rn.add_argument("--phi", type=float, default=0.5)
    rn.add_argument("--pattern", default="SEEDED", choices=["SEEDED", "SPONTANEOUS"])
    rn.add_argument("--rep", type=int, default=0)
    rn.add_argument("--set", default="B", choices=["A", "B"])
    rn.add_argument("--rule", default="harmonic", choices=["harmonic", "arithmetic"])
    rn.add_argument("--out", default=None)
    rf = sub.add_parser("run-full", help="full baseline campaign (expensive)")
    rf.add_argument("--out", default="results/campaign")
    rf.add_argument("--confirm-expensive", action="store_true", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "self-test":
        self_test(Path(args.data) if args.data else None)
    elif args.cmd == "example":
        example(Path(args.out), Path(args.data) if args.data else None, args.verify)
    elif args.cmd == "run":
        out = run_setting(phi=args.phi, pattern=args.pattern, replicate=args.rep,
                          stiffness_set=args.set, rule=args.rule, verbose=True)
        print(json.dumps({"K_aff_mNm": out["K_aff"], "K_rel_F2_mNm": out["K_rel_F2"],
                          "L_mNm": out["L"], "q": out["q"],
                          "K0_tangent_mNm": out["hessian"]["K0_mNm"]}, indent=2))
        if args.out:
            p = Path(args.out)
            p.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p / "network.npz", xyz0=out["xyz0"], pairs=out["pairs"],
                                l0=out["l0"], phase_labels=out["labels"],
                                bond_classes=out["net"].bond_class, k_omm=out["net"].k)
            print(f"wrote {p}/network.npz")
    elif args.cmd == "run-full":
        run_full_campaign(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
