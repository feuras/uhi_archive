"""
49_uih_hypocoercivity_coupling_scan.py

UIH finite dimensional hypocoercivity coupling scan.

This script generates synthetic finite dimensional UIH models (M, G, J)
with a single stationary direction e0, computes the Fisher gap λ_F of -G
on the mean-zero subspace V0, the metric operator norms

    L_J      = ||J||_M,
    L_comm   = ||[G, J]||_M,

and the associated adimensional UIH couplings

    g1 = L_J / λ_F,
    g2 = L_comm / λ_F**2,

as defined in the finite dimensional UIH hypocoercivity theorem.

For each model it also computes the hypocoercive decay rate

    λ_hyp := - max{ Re(z) : z ∈ spec(K0) },

where K0 = G0 + J0 is the restriction of K = G + J to the mean-zero
subspace V0. The primary diagnostic is the ratio

    λ_hyp / λ_F

as a function of (g1, g2) across a family of random UIH models with
controlled reversible sector strength.

The synthetic UIH models are not tied to any specific Markov, FP or GKLS
discretisation; instead they realise the abstract metric hypotheses of the
finite dimensional theorem in the simplest possible way:

    * the metric M is taken to be the identity,
    * G is constructed to be symmetric negative semidefinite with a
      one-dimensional kernel spanned by e0,
    * J is constructed to be skew-symmetric with J e0 = 0.

In this basis the metric-adjoint coincides with the Euclidean transpose,
and the UIH norms reduce to standard operator 2-norms. This provides a
clean numerical laboratory in which to test how λ_hyp / λ_F behaves as
the couplings g1, g2 are varied.

Usage (from the code archive root):

    python 42_uih_hypocoercivity_coupling_scan.py

Optional arguments:

    --num-samples N       total number of random UIH models (default: 500)
    --dims D1 D2 ...      list of dimensions to sample (default: 4 6 8)
    --j-scales S1 S2 ...  list of J strength factors (default: 0.0 0.2 0.5 1.0 2.0)
    --seed SEED           RNG seed (default: 12345)
    --output FILE         output .npz path (default:
                          uih_hypocoercivity_coupling_scan_results.npz)

The script prints summary statistics grouped by (dimension, J_scale) and
saves the full per-sample data to the specified .npz file for downstream
plotting and analysis.

This script is designed to be self-contained. In a later pass, concrete
Markov / FP / GKLS models from other scripts can be added as additional
families by reusing the generic analysis functions defined below.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np


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

    In an orthonormal eigenbasis we take

        G_hat = diag(0, -λ_2, ..., -λ_n),

    with λ_k in [lam_F_min, lam_F_max] and λ_F := min_k λ_k, and

        J_hat = -J_hat^T,

    with J_hat having vanishing first row/column so that J_hat ê0 = 0
    for ê0 = (1,0,...,0). Mapping back to the original basis with an
    orthogonal matrix Q gives

        G = Q G_hat Q^T,
        J = Q J_hat Q^T,
        e0 = Q ê0.

    On the mean-zero subspace V0 = e0^⊥ the dissipative part G is
    strictly negative definite with Fisher gap λ_F, and J is skew.
    """
    # Random orthogonal matrix Q (QR with sign correction).
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    # Fix orientation to avoid random sign flips on det(Q).
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0

    # Draw positive eigenvalues in [lam_F_min, lam_F_max] for the
    # dissipative sector on V0.
    pos_eigs = lam_F_min + (lam_F_max - lam_F_min) * rng.random(n - 1)
    lam_F = float(pos_eigs.min())
    eigs = np.concatenate(([0.0], -pos_eigs))
    G_hat = np.diag(eigs)

    # Random skew part in eigenbasis with J_hat ê0 = 0.
    B = rng.normal(size=(n, n))
    J_hat = 0.5 * (B - B.T)
    J_hat[0, :] = 0.0
    J_hat[:, 0] = 0.0
    # Scale J relative to λ_F to control g1, g2 regimes.
    J_hat *= J_scale * lam_F

    # Map back to coordinate basis.
    G = Q @ G_hat @ Q.T
    J = Q @ J_hat @ Q.T
    e0 = Q[:, 0].copy()

    M = np.eye(n)

    model = UIHModel(M=M, G=G, J=J, e0=e0, label=label)
    return model, lam_F


def build_mean_zero_basis(M: np.ndarray, e0: np.ndarray) -> np.ndarray:
    """
    Construct an M-orthonormal basis for the mean-zero subspace V0.

    For the present script M is always the identity, so this reduces to
    standard Euclidean Gram-Schmidt. The function is written in metric
    form so that it can be reused later with nontrivial M.

    Returns a matrix B0 whose columns form an M-orthonormal basis of V0.
    """
    n = M.shape[0]
    # Normalise e0 in M-norm.
    Me0 = M @ e0
    norm_e0_sq = float(e0.T @ Me0)
    if norm_e0_sq <= 0.0:
        raise ValueError("Stationary direction e0 has nonpositive M-norm.")
    e0_norm = np.sqrt(norm_e0_sq)
    e0_unit = e0 / e0_norm

    # Start with e0_unit and complete to an M-orthonormal basis.
    B = np.zeros((n, n), dtype=float)
    B[:, 0] = e0_unit

    rng = np.random.default_rng(987654321)
    k = 1
    while k < n:
        v = rng.normal(size=n)
        # Project out previous basis vectors in M-inner product.
        for j in range(k):
            bj = B[:, j]
            coeff = float(v.T @ (M @ bj))
            v = v - coeff * bj
        # Check norm in M-inner product.
        Mv = M @ v
        norm_sq = float(v.T @ Mv)
        if norm_sq < 1e-10:
            continue
        B[:, k] = v / np.sqrt(norm_sq)
        k += 1

    # Columns 1..n-1 span V0.
    B0 = B[:, 1:]
    return B0


def restrict_to_mean_zero(
    model: UIHModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict (M, G, J) to the mean-zero subspace V0 = e0^⊥.

    Returns (M0, G0, J0, B0) where B0 has shape (n, n-1) and satisfies

        M0 = B0^T M B0,
        G0 = B0^T G B0,
        J0 = B0^T J B0.

    In the present script with M = I one has M0 = I_(n-1).
    """
    M, G, J, e0 = model.M, model.G, model.J, model.e0
    B0 = build_mean_zero_basis(M, e0)
    M0 = B0.T @ M @ B0
    G0 = B0.T @ G @ B0
    J0 = B0.T @ J @ B0
    return M0, G0, J0, B0


def op_norm_M(A: np.ndarray, M: np.ndarray) -> float:
    """
    Operator norm induced by metric M:

        ||A||_M := sup_{u != 0} ||A u||_M / ||u||_M.

    In matrix form this is the spectral norm of

        A_hat = R A R^{-1},

    where R is any positive square root of M, M = R^T R. We compute R
    from the eigen decomposition of M and then take

        ||A||_M = sqrt(λ_max(A_hat^T A_hat)).

    For M = I this reduces to the usual operator 2-norm.
    """
    # Eigen decomposition of SPD metric.
    evals, U = np.linalg.eigh(M)
    if np.any(evals <= 0.0):
        raise ValueError("Metric M is not strictly positive definite.")
    sqrt_e = np.sqrt(evals)
    inv_sqrt_e = 1.0 / sqrt_e
    R = U @ np.diag(sqrt_e) @ U.T
    R_inv = U @ np.diag(inv_sqrt_e) @ U.T

    A_hat = R @ A @ R_inv
    # A_hat^T A_hat is symmetric positive semidefinite.
    S = A_hat.T @ A_hat
    svals = np.linalg.eigvalsh(S)
    return float(np.sqrt(max(svals.max(), 0.0)))


def fisher_gap(M0: np.ndarray, G0: np.ndarray) -> float:
    """
    Fisher gap λ_F: smallest positive eigenvalue of the positive operator
    -G0 on V0.

        λ_F := min spec(-G0).

    Here M0 enters only implicitly, via the construction of G0. In a
    general metric one would diagonalise the metric-symmetrised operator,
    but for finite dimensional tests it is sufficient to work in an
    M0-orthonormal basis.
    """
    # In the basis provided by restrict_to_mean_zero, M0 is SPD and
    # symmetric. For M = I, M0 = I.
    # We assume G0 is already M0-symmetric and negative definite.
    evals = np.linalg.eigvalsh(-G0)
    # Filter out numerical zero modes if any.
    tol = 1e-10
    positive = evals[evals > tol]
    if positive.size == 0:
        raise ValueError("No positive eigenvalues found for -G0.")
    return float(positive.min())


def lambda_hyp(G0: np.ndarray, J0: np.ndarray) -> float:
    """
    Hypocoercive decay rate λ_hyp: minus the largest real part of the
    spectrum of K0 = G0 + J0 on V0:

        λ_hyp := - max{ Re z : z ∈ spec(K0) }.

    By construction K0 has spectrum in the left half-plane.
    """
    K0 = G0 + J0
    eigs = np.linalg.eigvals(K0)
    lam = -float(np.max(eigs.real))
    return lam


def compute_uih_diagnostics(model: UIHModel) -> Dict[str, Any]:
    """
    For a given UIHModel compute (λ_F, λ_hyp, L_J, L_comm, g1, g2)
    on the mean-zero subspace V0.
    """
    M0, G0, J0, _ = restrict_to_mean_zero(model)

    lam_F = fisher_gap(M0, G0)
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
    }


def run_scan(
    num_samples: int,
    dims: List[int],
    j_scales: List[float],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Run a coupling scan across the specified dimensions and J scales.

    Returns a dictionary of numpy arrays suitable for saving to .npz.
    """
    rng = np.random.default_rng(seed)

    records: List[Dict[str, Any]] = []

    total_configs = len(dims) * len(j_scales)
    print(
        f"[UIH] Hypocoercivity coupling scan with "
        f"{num_samples} samples, "
        f"{len(dims)} dims, {len(j_scales)} J scales "
        f"(seed={seed})."
    )
    sys.stdout.flush()

    for di, n in enumerate(dims):
        for sj, J_scale in enumerate(j_scales):
            label = f"random_uih_n{n}_Jscale{J_scale:.3g}"
            print(
                f"[UIH] Sampling models for dim={n}, J_scale={J_scale:.3g} "
                f"({di * len(j_scales) + sj + 1}/{total_configs})."
            )
            sys.stdout.flush()

            for k in range(num_samples):
                model, lam_F_input = generate_random_uih_model(
                    n=n,
                    lam_F_min=0.1,
                    lam_F_max=2.0,
                    J_scale=J_scale,
                    rng=rng,
                    label=label,
                )
                diag = compute_uih_diagnostics(model)
                rec = {
                    "dim": n,
                    "J_scale": J_scale,
                    "lambda_F": diag["lambda_F"],
                    "lambda_hyp": diag["lambda_hyp"],
                    "L_J": diag["L_J"],
                    "L_comm": diag["L_comm"],
                    "g1": diag["g1"],
                    "g2": diag["g2"],
                    "ratio_hyp_over_F": diag["ratio_hyp_over_F"],
                }
                # Optionally keep track of the "input" λ_F from construction.
                rec["lambda_F_input"] = lam_F_input
                records.append(rec)

            # Per-configuration summary.
            cfg_ratios = np.array(
                [
                    r["ratio_hyp_over_F"]
                    for r in records
                    if (r["dim"] == n and abs(r["J_scale"] - J_scale) < 1e-12)
                ]
            )
            cfg_g1 = np.array(
                [
                    r["g1"]
                    for r in records
                    if (r["dim"] == n and abs(r["J_scale"] - J_scale) < 1e-12)
                ]
            )
            cfg_g2 = np.array(
                [
                    r["g2"]
                    for r in records
                    if (r["dim"] == n and abs(r["J_scale"] - J_scale) < 1e-12)
                ]
            )
            if cfg_ratios.size > 0:
                print(
                    f"       dim={n:2d}, J_scale={J_scale:5.2f}: "
                    f"λ_hyp/λ_F in [{cfg_ratios.min():.3f}, {cfg_ratios.max():.3f}], "
                    f"mean={cfg_ratios.mean():.3f}; "
                    f"⟨g1⟩≈{cfg_g1.mean():.3f}, ⟨g2⟩≈{cfg_g2.mean():.3f}."
                )
            else:
                print(
                    f"       dim={n:2d}, J_scale={J_scale:5.2f}: no samples?"
                )
            sys.stdout.flush()

    # Pack arrays.
    dims_arr = np.array([r["dim"] for r in records], dtype=int)
    J_scale_arr = np.array([r["J_scale"] for r in records], dtype=float)
    lambda_F_arr = np.array([r["lambda_F"] for r in records], dtype=float)
    lambda_F_in_arr = np.array([r["lambda_F_input"] for r in records], dtype=float)
    lambda_hyp_arr = np.array([r["lambda_hyp"] for r in records], dtype=float)
    L_J_arr = np.array([r["L_J"] for r in records], dtype=float)
    L_comm_arr = np.array([r["L_comm"] for r in records], dtype=float)
    g1_arr = np.array([r["g1"] for r in records], dtype=float)
    g2_arr = np.array([r["g2"] for r in records], dtype=float)
    ratio_arr = np.array([r["ratio_hyp_over_F"] for r in records], dtype=float)

    return {
        "dim": dims_arr,
        "J_scale": J_scale_arr,
        "lambda_F": lambda_F_arr,
        "lambda_F_input": lambda_F_in_arr,
        "lambda_hyp": lambda_hyp_arr,
        "L_J": L_J_arr,
        "L_comm": L_comm_arr,
        "g1": g1_arr,
        "g2": g2_arr,
        "ratio_hyp_over_F": ratio_arr,
        "seed": np.array([seed], dtype=int),
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite dimensional UIH hypocoercivity coupling scan."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="number of random UIH models per (dim, J_scale) combination",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="list of dimensions to sample (default: 4 6 8)",
    )
    parser.add_argument(
        "--j-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5, 1.0, 2.0],
        help="list of J strength factors (default: 0.0 0.2 0.5 1.0 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed (default: 12345)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="uih_hypocoercivity_coupling_scan_results.npz",
        help="output .npz filename "
        "(default: uih_hypocoercivity_coupling_scan_results.npz)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    results = run_scan(
        num_samples=args.num_samples,
        dims=args.dims,
        j_scales=args.j_scales,
        seed=args.seed,
    )

    out_path = args.output
    np.savez_compressed(out_path, **results)
    print(f"[UIH] Saved coupling scan results to: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
