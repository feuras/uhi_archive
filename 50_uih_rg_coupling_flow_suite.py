"""
50_uih_rg_coupling_flow_suite.py

UIH renormalisation group (RG) for dissipation: coupling flows.

This script implements a mode-space RG map for finite dimensional UIH
models (M, G, J) with a single stationary direction e0. It uses the
same synthetic random UIH ensemble as the hypocoercivity scan, but now
applies a Fisher-preserving coarse graining to the mean-zero sector V0
and tracks how the couplings

    g1 = ||J0||_M / λ_F,
    g2 = ||[G0, J0]||_M / λ_F**2,

and the ratio

    R = λ_hyp / λ_F

flow under successive RG steps.

RG scheme (mode-space, Fisher-preserving):

  1. Start from a finite dimensional UIH model (M, G, J, e0) of size n.
     Restrict to the mean-zero subspace

        V0 = { u : ⟨u, e0⟩_M = 0 },

     obtaining (M0, G0, J0) of size (n-1) × (n-1).

  2. Diagonalise the Fisher operator on V0,

        L_F := -G0,

     and sort its eigenvalues λ_2 ≤ λ_3 ≤ ... ≤ λ_n together with
     orthonormal eigenvectors v_k in the M0 inner product. The Fisher
     gap is

        λ_F := λ_2.

  3. Define the RG projection onto the m_slow slowest Fisher modes:

        B_RG = [v_2, v_3, ..., v_{m_slow+1}],

     and coarse grained operators

        M0' = B_RG^T M0 B_RG,
        G0' = B_RG^T G0 B_RG,
        J0' = B_RG^T J0 B_RG.

     In the present implementation M is the identity, so M0' ≈ I.

  4. Repeat step 2–3 on (M0', G0', J0'), tracking at each step:

        λ_F,
        λ_hyp := - max Re spec(G0 + J0),
        g1, g2,
        R := λ_hyp / λ_F,
        dim(V0).

The script runs an ensemble of random UIH models and reports the
ensemble-averaged flows R_k, g1_k, g2_k as functions of the RG step k.

Usage (from the code archive root):

    python 43_uih_rg_coupling_flow_suite.py

Optional arguments:

    --num-models N      number of independent UIH models (default: 128)
    --dim DIM           full space dimension n (default: 16)
    --j-scale S         scale factor for J relative to λ_F (default: 1.0)
    --rg-steps K        number of RG steps on V0 (default: 5)
    --m-slow M          number of slow Fisher modes to retain (default: 4)
    --workers W         number of worker processes (default: 20; if W<2
                        or parallel fails, falls back to 1)
    --seed SEED         RNG seed (default: 2025)
    --output FILE       output .npz path (default:
                        uih_rg_coupling_flow_results.npz)
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# UIH model and basic constructions (mirrors the hypocoercivity scan script)
# ---------------------------------------------------------------------------

@dataclass
class UIHModel:
    """
    Container for a finite dimensional UIH model in coordinates.

    M : (n, n) SPD metric matrix
    G : (n, n) metric-symmetric dissipative part
    J : (n, n) metric-skew reversible part
    e0: (n,) stationary direction with K e0 = 0
    """
    M: np.ndarray
    G: np.ndarray
    J: np.ndarray
    e0: np.ndarray
    label: str


def generate_random_uih_model(
    n: int,
    lam_F_min: float,
    lam_F_max: float,
    J_scale: float,
    rng: np.random.Generator,
    label: str,
) -> Tuple[UIHModel, float]:
    """
    Generate a random finite dimensional UIH model with metric M = I_n.
    """
    # Random orthogonal matrix Q (QR with sign correction).
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0

    # Draw positive eigenvalues in [lam_F_min, lam_F_max].
    pos_eigs = lam_F_min + (lam_F_max - lam_F_min) * rng.random(n - 1)
    lam_F = float(pos_eigs.min())
    eigs = np.concatenate(([0.0], -pos_eigs))
    G_hat = np.diag(eigs)

    # Random skew part J_hat with J_hat ê0 = 0.
    B = rng.normal(size=(n, n))
    J_hat = 0.5 * (B - B.T)
    J_hat[0, :] = 0.0
    J_hat[:, 0] = 0.0
    J_hat *= J_scale * lam_F

    # Map back.
    G = Q @ G_hat @ Q.T
    J = Q @ J_hat @ Q.T
    e0 = Q[:, 0].copy()
    M = np.eye(n)

    model = UIHModel(M=M, G=G, J=J, e0=e0, label=label)
    return model, lam_F


def build_mean_zero_basis(M: np.ndarray, e0: np.ndarray) -> np.ndarray:
    """
    Construct an M-orthonormal basis for the mean-zero subspace V0.

    Returns a matrix B0 whose columns form an M-orthonormal basis of V0.
    """
    n = M.shape[0]
    Me0 = M @ e0
    norm_e0_sq = float(e0.T @ Me0)
    if norm_e0_sq <= 0.0:
        raise ValueError("Stationary direction e0 has nonpositive M-norm.")
    e0_unit = e0 / np.sqrt(norm_e0_sq)

    B = np.zeros((n, n), dtype=float)
    B[:, 0] = e0_unit

    rng = np.random.default_rng(987654321)
    k = 1
    while k < n:
        v = rng.normal(size=n)
        for j in range(k):
            bj = B[:, j]
            coeff = float(v.T @ (M @ bj))
            v = v - coeff * bj
        Mv = M @ v
        norm_sq = float(v.T @ Mv)
        if norm_sq < 1e-10:
            continue
        B[:, k] = v / np.sqrt(norm_sq)
        k += 1

    B0 = B[:, 1:]
    return B0


def restrict_to_mean_zero(
    model: UIHModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict (M, G, J) to the mean-zero subspace V0 = e0^⊥.

    Returns (M0, G0, J0) of shape (n-1, n-1) each.
    """
    M, G, J, e0 = model.M, model.G, model.J, model.e0
    B0 = build_mean_zero_basis(M, e0)
    M0 = B0.T @ M @ B0
    G0 = B0.T @ G @ B0
    J0 = B0.T @ J @ B0
    return M0, G0, J0


# ---------------------------------------------------------------------------
# Metrics, norms, and UIH diagnostics
# ---------------------------------------------------------------------------

def op_norm_M(A: np.ndarray, M: np.ndarray) -> float:
    """
    Operator norm induced by metric M:

        ||A||_M := sup_{u != 0} ||A u||_M / ||u||_M.
    """
    evals, U = np.linalg.eigh(M)
    if np.any(evals <= 0.0):
        raise ValueError("Metric M is not strictly positive definite.")
    sqrt_e = np.sqrt(evals)
    inv_sqrt_e = 1.0 / sqrt_e
    R = U @ np.diag(sqrt_e) @ U.T
    R_inv = U @ np.diag(inv_sqrt_e) @ U.T

    A_hat = R @ A @ R_inv
    S = A_hat.T @ A_hat
    svals = np.linalg.eigvalsh(S)
    return float(np.sqrt(max(svals.max(), 0.0)))


def fisher_gap(M0: np.ndarray, G0: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Fisher gap λ_F: smallest positive eigenvalue of -G0 on V0.
    Returns (λ_F, evals, evecs) with evals sorted ascending.
    """
    evals, evecs = np.linalg.eigh(-G0)
    evals = np.maximum(evals, 0.0)
    tol = 1e-10
    positive = evals[evals > tol]
    if positive.size == 0:
        raise ValueError("No positive eigenvalues found for -G0.")
    lam_F = float(positive.min())
    return lam_F, evals, evecs


def lambda_hyp(G0: np.ndarray, J0: np.ndarray) -> float:
    """
    Hypocoercive decay rate λ_hyp: minus the largest real part of the
    spectrum of K0 = G0 + J0 on V0.
    """
    K0 = G0 + J0
    eigs = np.linalg.eigvals(K0)
    lam = -float(np.max(eigs.real))
    return lam


def compute_uih_diagnostics_from_MGJ(
    M0: np.ndarray,
    G0: np.ndarray,
    J0: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute (λ_F, λ_hyp, L_J, L_comm, g1, g2, ratio) directly from
    (M0, G0, J0) on V0.
    """
    lam_F, evals_LF, evecs_LF = fisher_gap(M0, G0)
    L_J = op_norm_M(J0, M0)
    comm = G0 @ J0 - J0 @ G0
    L_comm = op_norm_M(comm, M0)
    g1 = L_J / lam_F
    g2 = L_comm / (lam_F * lam_F)
    lam_h = lambda_hyp(G0, J0)
    ratio = lam_h / lam_F

    return {
        "lambda_F": lam_F,
        "lambda_hyp": lam_h,
        "L_J": L_J,
        "L_comm": L_comm,
        "g1": g1,
        "g2": g2,
        "ratio_hyp_over_F": ratio,
        "evals_LF": evals_LF,
        "evecs_LF": evecs_LF,
    }


# ---------------------------------------------------------------------------
# RG map on V0: projection onto slow Fisher modes
# ---------------------------------------------------------------------------

def rg_step_slow_modes(
    M0: np.ndarray,
    G0: np.ndarray,
    J0: np.ndarray,
    m_slow: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One RG step on the mean-zero subspace V0: project onto m_slow
    slowest Fisher modes and coarse-grain (M0, G0, J0).
    """
    evals, evecs = np.linalg.eigh(-G0)
    evals = np.maximum(evals, 0.0)

    tol = 1e-10
    positive_indices = np.where(evals > tol)[0]
    if positive_indices.size == 0:
        # Nothing to coarse-grain in a meaningful way.
        return M0, G0, J0

    m_eff = min(m_slow, positive_indices.size)
    idx_slow = positive_indices[:m_eff]
    B_RG = evecs[:, idx_slow]

    M0_cg = B_RG.T @ M0 @ B_RG
    G0_cg = B_RG.T @ G0 @ B_RG
    J0_cg = B_RG.T @ J0 @ B_RG

    return M0_cg, G0_cg, J0_cg


# ---------------------------------------------------------------------------
# Single-model RG flow (for parallel use)
# ---------------------------------------------------------------------------

def _rg_flow_single_model(
    m_index: int,
    dim: int,
    J_scale: float,
    rg_steps: int,
    m_slow: int,
    seed: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run RG flow for a single model. Returns a tuple:

        (m_index, dims_row, lambda_F_row, lambda_hyp_row,
         g1_row, g2_row, ratio_row)
    """
    # Derive a per-model seed to keep things reproducible and independent.
    rng = np.random.default_rng(seed + 7919 * m_index)

    label = f"rg_model_{m_index}"
    model, lam_F_input = generate_random_uih_model(
        n=dim,
        lam_F_min=0.1,
        lam_F_max=2.0,
        J_scale=J_scale,
        rng=rng,
        label=label,
    )
    M0, G0, J0 = restrict_to_mean_zero(model)

    steps = rg_steps + 1
    dims_row = np.zeros(steps, dtype=int)
    lambda_F_row = np.zeros(steps, dtype=float)
    lambda_hyp_row = np.zeros(steps, dtype=float)
    g1_row = np.zeros(steps, dtype=float)
    g2_row = np.zeros(steps, dtype=float)
    ratio_row = np.zeros(steps, dtype=float)

    # Step 0 diagnostics.
    diag0 = compute_uih_diagnostics_from_MGJ(M0, G0, J0)
    dims_row[0] = M0.shape[0]
    lambda_F_row[0] = diag0["lambda_F"]
    lambda_hyp_row[0] = diag0["lambda_hyp"]
    g1_row[0] = diag0["g1"]
    g2_row[0] = diag0["g2"]
    ratio_row[0] = diag0["ratio_hyp_over_F"]

    # Optional sanity check on λ_F consistency with construction.
    lam_F_input_safe = max(lam_F_input, 1e-12)
    rel_diff = abs(diag0["lambda_F"] - lam_F_input_safe) / lam_F_input_safe
    if rel_diff > 1e-3:
        print(
            f"[UIH-RG] WARNING (model {m_index}): relative λ_F mismatch "
            f"{rel_diff:.3e} (constructed vs extracted)."
        )

    M_current, G_current, J_current = M0, G0, J0

    for k in range(1, steps):
        dim_current = M_current.shape[0]
        if dim_current <= 1:
            dims_row[k:] = dim_current
            lambda_F_row[k:] = lambda_F_row[k - 1]
            lambda_hyp_row[k:] = lambda_hyp_row[k - 1]
            g1_row[k:] = g1_row[k - 1]
            g2_row[k:] = g2_row[k - 1]
            ratio_row[k:] = ratio_row[k - 1]
            break

        m_eff = min(m_slow, dim_current)
        M_cg, G_cg, J_cg = rg_step_slow_modes(
            M_current,
            G_current,
            J_current,
            m_slow=m_eff,
        )
        diag_k = compute_uih_diagnostics_from_MGJ(M_cg, G_cg, J_cg)

        dims_row[k] = M_cg.shape[0]
        lambda_F_row[k] = diag_k["lambda_F"]
        lambda_hyp_row[k] = diag_k["lambda_hyp"]
        g1_row[k] = diag_k["g1"]
        g2_row[k] = diag_k["g2"]
        ratio_row[k] = diag_k["ratio_hyp_over_F"]

        M_current, G_current, J_current = M_cg, G_cg, J_cg

    return (
        m_index,
        dims_row,
        lambda_F_row,
        lambda_hyp_row,
        g1_row,
        g2_row,
        ratio_row,
    )


# ---------------------------------------------------------------------------
# RG flow driver (ensemble)
# ---------------------------------------------------------------------------

def run_rg_flow_ensemble(
    num_models: int,
    dim: int,
    J_scale: float,
    rg_steps: int,
    m_slow: int,
    seed: int,
    workers: int,
) -> Dict[str, np.ndarray]:
    """
    Run a UIH RG coupling flow on an ensemble of random UIH models.

    Supports parallel execution with `workers` processes. If workers < 2
    or parallel execution fails, falls back to sequential evaluation.
    """
    print(
        f"[UIH-RG] Ensemble of {num_models} random UIH models, "
        f"dim={dim}, J_scale={J_scale}, rg_steps={rg_steps}, "
        f"m_slow={m_slow}, workers={workers}, seed={seed}."
    )
    sys.stdout.flush()

    steps = rg_steps + 1
    dims_arr = np.zeros((num_models, steps), dtype=int)
    lambda_F_arr = np.zeros((num_models, steps), dtype=float)
    lambda_hyp_arr = np.zeros((num_models, steps), dtype=float)
    g1_arr = np.zeros((num_models, steps), dtype=float)
    g2_arr = np.zeros((num_models, steps), dtype=float)
    ratio_arr = np.zeros((num_models, steps), dtype=float)

    def _run_sequential():
        for m in range(num_models):
            (
                idx,
                dims_row,
                lambda_F_row,
                lambda_hyp_row,
                g1_row,
                g2_row,
                ratio_row,
            ) = _rg_flow_single_model(
                m_index=m,
                dim=dim,
                J_scale=J_scale,
                rg_steps=rg_steps,
                m_slow=m_slow,
                seed=seed,
            )
            dims_arr[idx, :] = dims_row
            lambda_F_arr[idx, :] = lambda_F_row
            lambda_hyp_arr[idx, :] = lambda_hyp_row
            g1_arr[idx, :] = g1_row
            g2_arr[idx, :] = g2_row
            ratio_arr[idx, :] = ratio_row

            if (m + 1) % max(1, num_models // 10) == 0:
                print(f"[UIH-RG] Completed {m + 1}/{num_models} models (seq).")
                sys.stdout.flush()

    if workers is None:
        workers = 1
    if workers < 2:
        _run_sequential()
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _rg_flow_single_model,
                        m,
                        dim,
                        J_scale,
                        rg_steps,
                        m_slow,
                        seed,
                    )
                    for m in range(num_models)
                ]
                completed = 0
                for fut in as_completed(futures):
                    (
                        idx,
                        dims_row,
                        lambda_F_row,
                        lambda_hyp_row,
                        g1_row,
                        g2_row,
                        ratio_row,
                    ) = fut.result()
                    dims_arr[idx, :] = dims_row
                    lambda_F_arr[idx, :] = lambda_F_row
                    lambda_hyp_arr[idx, :] = lambda_hyp_row
                    g1_arr[idx, :] = g1_row
                    g2_arr[idx, :] = g2_row
                    ratio_arr[idx, :] = ratio_row

                    completed += 1
                    if completed % max(1, num_models // 10) == 0:
                        print(
                            f"[UIH-RG] Completed {completed}/{num_models} "
                            f"models (parallel)."
                        )
                        sys.stdout.flush()
        except Exception as exc:
            print(
                "[UIH-RG] WARNING: parallel execution failed with "
                f"{type(exc).__name__}: {exc}. Falling back to 1 worker."
            )
            sys.stdout.flush()
            # Clear and recompute sequentially for safety.
            dims_arr.fill(0)
            lambda_F_arr.fill(0.0)
            lambda_hyp_arr.fill(0.0)
            g1_arr.fill(0.0)
            g2_arr.fill(0.0)
            ratio_arr.fill(0.0)
            _run_sequential()

    print("[UIH-RG] Ensemble-averaged flows:")
    for k in range(steps):
        mean_dim = dims_arr[:, k].mean()
        mean_ratio = ratio_arr[:, k].mean()
        min_ratio = ratio_arr[:, k].min()
        max_ratio = ratio_arr[:, k].max()
        mean_g1 = g1_arr[:, k].mean()
        mean_g2 = g2_arr[:, k].mean()
        print(
            f"  step {k:2d}: ⟨dim(V0)⟩≈{mean_dim:5.2f}, "
            f"λ_hyp/λ_F in [{min_ratio:.3f}, {max_ratio:.3f}], "
            f"mean={mean_ratio:.3f}; "
            f"⟨g1⟩≈{mean_g1:.3f}, ⟨g2⟩≈{mean_g2:.3f}."
        )
    sys.stdout.flush()

    return {
        "dim_V0": dims_arr,
        "lambda_F": lambda_F_arr,
        "lambda_hyp": lambda_hyp_arr,
        "g1": g1_arr,
        "g2": g2_arr,
        "ratio_hyp_over_F": ratio_arr,
        "J_scale": np.array([J_scale], dtype=float),
        "seed": np.array([seed], dtype=int),
        "workers": np.array([workers], dtype=int),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UIH renormalisation group for dissipation: coupling flows."
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=100000,
        help="number of independent UIH models (default: 128)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=16,
        help="full space dimension n (default: 16)",
    )
    parser.add_argument(
        "--j-scale",
        type=float,
        default=1.0,
        help="scale factor for J relative to λ_F (default: 1.0)",
    )
    parser.add_argument(
        "--rg-steps",
        type=int,
        default=5,
        help="number of RG steps on V0 (default: 5)",
    )
    parser.add_argument(
        "--m-slow",
        type=int,
        default=4,
        help="number of slow Fisher modes to retain at each step "
        "(default: 4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="number of worker processes (default: 20; <2 means sequential)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="RNG seed (default: 2025)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="uih_rg_coupling_flow_results.npz",
        help="output .npz filename "
        "(default: uih_rg_coupling_flow_results.npz)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    results = run_rg_flow_ensemble(
        num_models=args.num_models,
        dim=args.dim,
        J_scale=args.j_scale,
        rg_steps=args.rg_steps,
        m_slow=args.m_slow,
        seed=args.seed,
        workers=args.workers,
    )

    out_path = args.output
    np.savez_compressed(out_path, **results)
    print(f"[UIH-RG] Saved RG coupling flow results to: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
