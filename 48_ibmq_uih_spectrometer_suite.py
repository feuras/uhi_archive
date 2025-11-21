#!/usr/bin/env python3
"""
UIH spectrometer for IBM two-qubit K tomography data.

Choices:

  - Use K_reg as the generator in the BKM-orthonormal traceless basis.
  - Define G and J as the symmetric and skew parts of K_reg in that basis.
  - Compute lambda_F from -G, lambda_hyp from K, and g1, g2 from J and [G, J].

Provides:

  - 'spectrometer' command: UIH invariants for each *uih_split_bkm.npz.
  - 'rg' command: one slow-mode RG step on K.
  - 'rg2' command: two successive slow-mode RG steps on K.
  - 'inspect' command: list contents of an npz file.
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from numpy.linalg import eigvals, eigvalsh, norm
from scipy.linalg import expm, logm
from scipy.optimize import least_squares


@dataclass
class UIHInvariants:
    lambda_F: float
    lambda_hyp: float
    g1: float
    g2: float
    L_J: float
    L_comm: float
    dim: int


def _pick_key(npz, candidates: List[str], context: str):
    for key in candidates:
        if key in npz.files:
            return npz[key]
    raise KeyError(f"Could not find any of keys {candidates!r} in npz file ({context})")


# =========================
# Core invariants from K
# =========================

def compute_uih_invariants_from_K(K: np.ndarray, zero_tol: float = 1e-10) -> UIHInvariants:
    """
    K is the generator on the BKM-orthonormal traceless space (n x n).

    We define:
      G = (K + K^T) / 2
      J = (K - K^T) / 2

    Then compute:
      lambda_F from the smallest positive eigenvalue of -G,
      lambda_hyp from the smallest positive -Re(eig(K)),
      g1 from ||J|| / lambda_F,
      g2 from ||[G, J]|| / lambda_F^2.
    """
    K = np.asarray(K, dtype=float)
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")

    G = 0.5 * (K + K.T)
    J = 0.5 * (K - K.T)

    evals_negG = eigvalsh(-G)
    pos_mask = evals_negG > zero_tol
    if not np.any(pos_mask):
        raise RuntimeError("No positive eigenvalues in -G; cannot define Fisher gap.")
    lambda_F = float(evals_negG[pos_mask].min())

    evals_K = eigvals(K)
    rates = -np.real(evals_K)
    pos_rate_mask = rates > zero_tol
    if not np.any(pos_rate_mask):
        raise RuntimeError("No positive decay rates in K; cannot define hypocoercive rate.")
    lambda_hyp = float(rates[pos_rate_mask].min())

    L_J = float(norm(J, 2))
    comm = G @ J - J @ G
    L_comm = float(norm(comm, 2))

    g1 = L_J / lambda_F if lambda_F > 0 else np.inf
    g2 = L_comm / (lambda_F * lambda_F) if lambda_F > 0 else np.inf

    return UIHInvariants(
        lambda_F=lambda_F,
        lambda_hyp=lambda_hyp,
        g1=g1,
        g2=g2,
        L_J=L_J,
        L_comm=L_comm,
        dim=K.shape[0],
    )


def load_K_from_uih_split_bkm(path: str) -> np.ndarray:
    """
    Load K_reg from *uih_split_bkm.npz.

    We treat K_reg as already in the BKM-orthonormal traceless basis.
    """
    npz = np.load(path)
    K = _pick_key(npz, ["K_reg", "K_bkm", "K_BKM", "K"], context="K_reg")
    return np.asarray(K, dtype=float)


# =========================
# Slow-mode RG on K
# =========================

def build_slow_mode_projection(K: np.ndarray, m_slow: int, zero_tol: float = 1e-10) -> np.ndarray:
    """
    Build a projection P onto the m_slow slowest modes of -G.

    K is in BKM-orthonormal basis. We form G = (K + K^T)/2,
    diagonalise -G, and take the eigenvectors with the smallest
    positive eigenvalues of -G.

    Returns P of shape (n, m_slow) with orthonormal columns.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    G = 0.5 * (K + K.T)
    evals, vecs = np.linalg.eigh(-G)
    pos_mask = evals > zero_tol
    evals_pos = evals[pos_mask]
    vecs_pos = vecs[:, pos_mask]

    if evals_pos.size < m_slow:
        raise RuntimeError(
            f"Requested {m_slow} slow modes but only {evals_pos.size} positive eigenvalues of -G."
        )

    idx_sorted = np.argsort(evals_pos)
    idx_keep = idx_sorted[:m_slow]
    P = vecs_pos[:, idx_keep]

    return P


def run_rg_step_slow_modes(
    K: np.ndarray,
    m_slow: int = 4,
    zero_tol: float = 1e-10,
) -> Tuple[UIHInvariants, UIHInvariants]:
    """
    One RG step based on slow modes of -G.

    Returns:
      invariants on full K,
      invariants on coarse K after Fisher gap rescale.
    """
    inv_full = compute_uih_invariants_from_K(K, zero_tol=zero_tol)

    P = build_slow_mode_projection(K, m_slow=m_slow, zero_tol=zero_tol)
    K_c = P.T @ K @ P

    inv_coarse_raw = compute_uih_invariants_from_K(K_c, zero_tol=zero_tol)
    if inv_coarse_raw.lambda_F <= 0:
        raise RuntimeError("Coarse lambda_F is nonpositive; cannot rescale coarse K.")

    alpha = inv_full.lambda_F / inv_coarse_raw.lambda_F
    K_c_rescaled = alpha * K_c

    inv_coarse = compute_uih_invariants_from_K(K_c_rescaled, zero_tol=zero_tol)

    return inv_full, inv_coarse


def run_two_rg_steps_slow_modes(
    K: np.ndarray,
    m_slow: int = 4,
    zero_tol: float = 1e-10,
) -> Tuple[UIHInvariants, UIHInvariants, UIHInvariants]:
    """
    Two successive slow-mode RG steps.

    Step 0: full K, invariants inv0.
    Step 1: apply slow-mode RG to K, rescale Fisher gap, invariants inv1.
    Step 2: apply slow-mode RG again to the step 1 coarse generator, rescale Fisher gap, invariants inv2.

    Returns (inv0, inv1, inv2).
    """
    K0 = np.asarray(K, dtype=float)

    inv0 = compute_uih_invariants_from_K(K0, zero_tol=zero_tol)

    P1 = build_slow_mode_projection(K0, m_slow=m_slow, zero_tol=zero_tol)
    K1 = P1.T @ K0 @ P1
    inv1_raw = compute_uih_invariants_from_K(K1, zero_tol=zero_tol)
    if inv1_raw.lambda_F <= 0:
        raise RuntimeError("Step 1 coarse lambda_F nonpositive; cannot rescale.")
    alpha1 = inv0.lambda_F / inv1_raw.lambda_F
    K1_rescaled = alpha1 * K1
    inv1 = compute_uih_invariants_from_K(K1_rescaled, zero_tol=zero_tol)

    P2 = build_slow_mode_projection(K1_rescaled, m_slow=m_slow, zero_tol=zero_tol)
    K2 = P2.T @ K1_rescaled @ P2
    inv2_raw = compute_uih_invariants_from_K(K2, zero_tol=zero_tol)
    if inv2_raw.lambda_F <= 0:
        raise RuntimeError("Step 2 coarse lambda_F nonpositive; cannot rescale.")
    alpha2 = inv1.lambda_F / inv2_raw.lambda_F
    K2_rescaled = alpha2 * K2
    inv2 = compute_uih_invariants_from_K(K2_rescaled, zero_tol=zero_tol)

    return inv0, inv1, inv2


# =========================
# Optional: multi-depth fit
# =========================

def _flatten_matrix(M: np.ndarray) -> np.ndarray:
    return np.reshape(M, (-1,))


def _unflatten_matrix(v: np.ndarray, n: int) -> np.ndarray:
    return np.reshape(v, (n, n))


def fit_common_generator_from_T_list(
    T_list: List[np.ndarray],
    t_list: List[float],
    K_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit a single real generator K such that T_i ≈ exp(t_i K) for given
    CPTP superoperator matrices T_i at times t_i.

    This is optional, for Markovianity checks.
    """
    if len(T_list) != len(t_list):
        raise ValueError("T_list and t_list must have the same length.")

    n = T_list[0].shape[0]
    for T in T_list:
        if T.shape != (n, n):
            raise ValueError("All T matrices must have the same shape.")

    if K_init is None:
        idx_min = int(np.argmin(t_list))
        T0 = np.asarray(T_list[idx_min], dtype=float)
        t0 = float(t_list[idx_min])
        K0 = logm(T0) / t0
        K0 = np.real(K0)
    else:
        K0 = np.asarray(K_init, dtype=float)

    x0 = _flatten_matrix(K0)

    def residuals(x_vec: np.ndarray) -> np.ndarray:
        K = _unflatten_matrix(x_vec, n)
        res_blocks = []
        for T, t in zip(T_list, t_list):
            T_model = expm(t * K)
            res = T_model - T
            res_blocks.append(_flatten_matrix(res))
        return np.concatenate(res_blocks, axis=0)

    result = least_squares(residuals, x0, method="trf")
    K_fit = _unflatten_matrix(result.x, n)
    return np.real(K_fit)


# =========================
# CLI commands
# =========================

def cmd_spectrometer(pattern: str = "*uih_split_bkm.npz") -> None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[UIH] No files matching pattern {pattern!r}")
        return

    print("[UIH] Spectrometer on BKM split files (K_reg, symmetric/skew split)")
    print("file, dim, lambda_F, lambda_hyp, g1, g2, L_J, L_comm")
    for path in paths:
        try:
            K = load_K_from_uih_split_bkm(path)
            inv = compute_uih_invariants_from_K(K)
            print(
                f"{os.path.basename(path)}, "
                f"{inv.dim:d}, "
                f"{inv.lambda_F:.6e}, "
                f"{inv.lambda_hyp:.6e}, "
                f"{inv.g1:.6e}, "
                f"{inv.g2:.6e}, "
                f"{inv.L_J:.6e}, "
                f"{inv.L_comm:.6e}"
            )
        except Exception as e:
            print(f"[UIH] Error processing {path}: {e}")


def cmd_inspect(path: str) -> None:
    npz = np.load(path)
    print(f"[UIH] Inspecting {path}")
    for key in npz.files:
        arr = npz[key]
        print(f"  key={key!r}, shape={arr.shape}, dtype={arr.dtype}")


def cmd_rg_slow(pattern: str = "*uih_split_bkm.npz", m_slow: int = 4) -> None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[UIH-RG] No files matching pattern {pattern!r}")
        return

    print(f"[UIH-RG] One slow-mode RG step with m_slow={m_slow}")
    print("file, dim_full, dim_coarse, lambda_F_full, lambda_F_coarse, "
          "lambda_hyp_full, lambda_hyp_coarse, g1_full, g1_coarse, g2_full, g2_coarse")

    for path in paths:
        try:
            K = load_K_from_uih_split_bkm(path)
            inv_full, inv_coarse = run_rg_step_slow_modes(K, m_slow=m_slow)
            print(
                f"{os.path.basename(path)}, "
                f"{inv_full.dim:d}, "
                f"{m_slow:d}, "
                f"{inv_full.lambda_F:.6e}, "
                f"{inv_coarse.lambda_F:.6e}, "
                f"{inv_full.lambda_hyp:.6e}, "
                f"{inv_coarse.lambda_hyp:.6e}, "
                f"{inv_full.g1:.6e}, "
                f"{inv_coarse.g1:.6e}, "
                f"{inv_full.g2:.6e}, "
                f"{inv_coarse.g2:.6e}"
            )
        except Exception as e:
            print(f"[UIH-RG] Error processing {path}: {e}")


def cmd_rg_two_steps(pattern: str = "*uih_split_bkm.npz", m_slow: int = 4) -> None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[UIH-RG2] No files matching pattern {pattern!r}")
        return

    print(f"[UIH-RG2] Two slow-mode RG steps with m_slow={m_slow}")
    print("file, dim0, dim1, dim2, "
          "lambda_F0, lambda_F1, lambda_F2, "
          "lambda_hyp0, lambda_hyp1, lambda_hyp2, "
          "g1_0, g1_1, g1_2, "
          "g2_0, g2_1, g2_2")

    for path in paths:
        try:
            K = load_K_from_uih_split_bkm(path)
            inv0, inv1, inv2 = run_two_rg_steps_slow_modes(K, m_slow=m_slow)
            print(
                f"{os.path.basename(path)}, "
                f"{inv0.dim:d}, "
                f"{m_slow:d}, "
                f"{m_slow:d}, "
                f"{inv0.lambda_F:.6e}, "
                f"{inv1.lambda_F:.6e}, "
                f"{inv2.lambda_F:.6e}, "
                f"{inv0.lambda_hyp:.6e}, "
                f"{inv1.lambda_hyp:.6e}, "
                f"{inv2.lambda_hyp:.6e}, "
                f"{inv0.g1:.6e}, "
                f"{inv1.g1:.6e}, "
                f"{inv2.g1:.6e}, "
                f"{inv0.g2:.6e}, "
                f"{inv1.g2:.6e}, "
                f"{inv2.g2:.6e}"
            )
        except Exception as e:
            print(f"[UIH-RG2] Error processing {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UIH spectrometer for IBM K tomography data")
    subparsers = parser.add_subparsers(dest="cmd")

    p_spec = subparsers.add_parser("spectrometer", help="Run spectrometer on *uih_split_bkm.npz")
    p_spec.add_argument(
        "--pattern",
        type=str,
        default="*uih_split_bkm.npz",
        help="Glob pattern (default '*uih_split_bkm.npz')",
    )

    p_inspect = subparsers.add_parser("inspect", help="Inspect an npz file")
    p_inspect.add_argument("path", type=str)

    p_rg = subparsers.add_parser("rg", help="Run one slow-mode RG step on each file")
    p_rg.add_argument(
        "--pattern",
        type=str,
        default="*uih_split_bkm.npz",
        help="Glob pattern (default '*uih_split_bkm.npz')",
    )
    p_rg.add_argument(
        "--m-slow",
        type=int,
        default=4,
        help="Number of slow modes in coarse space (default 4)",
    )

    p_rg2 = subparsers.add_parser("rg2", help="Run two slow-mode RG steps on each file")
    p_rg2.add_argument(
        "--pattern",
        type=str,
        default="*uih_split_bkm.npz",
        help="Glob pattern (default '*uih_split_bkm.npz')",
    )
    p_rg2.add_argument(
        "--m-slow",
        type=int,
        default=4,
        help="Number of slow modes in each coarse space (default 4)",
    )

    args = parser.parse_args()

    if args.cmd == "spectrometer":
        cmd_spectrometer(pattern=args.pattern)
    elif args.cmd == "inspect":
        cmd_inspect(args.path)
    elif args.cmd == "rg":
        cmd_rg_slow(pattern=args.pattern, m_slow=args.m_slow)
    elif args.cmd == "rg2":
        cmd_rg_two_steps(pattern=args.pattern, m_slow=args.m_slow)
    else:
        cmd_spectrometer(pattern="*uih_split_bkm.npz")


if __name__ == "__main__":
    main()
