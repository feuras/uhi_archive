#!/usr/bin/env python3
"""
42_qapi_two_qubit_k_bkm_fisher_split_checks.py

Two-qubit BKM metric and UIH Fisher G/J split for an IBM K generator.

This script:

  * loads a "_uih_split.npz" file produced by
    41_qapi_two_qubit_k_uih_metric_split_checks.py, which contains:
        - K_reg   : 15x15 spectrally regularised generator matrix on su(4),
        - v_stat  : 16-component stationary Pauli expectation vector,
        - lam_stat: corresponding eigenvalue (should be ~1),

  * reconstructs the stationary two-qubit density matrix rho_ss in the
    {I,X,Y,Z}⊗2 Pauli basis using v_stat,

  * builds the Bogoliubov-Kubo-Mori (BKM) Fisher metric on the
    15-dimensional traceless Pauli sector at rho_ss,

  * computes the metric adjoint K_sharp with respect to this BKM metric

        K^sharp = M_bkm^{-1} K_reg^T M_bkm,

  * forms the Fisher G/J split

        G_bkm = 0.5 * (K_reg + K^sharp),
        J_bkm = 0.5 * (K_reg - K^sharp),

    which are symmetric / skew with respect to the BKM metric,

  * diagonalises G_bkm in BKM-orthonormal coordinates to obtain the
    dissipative spectrum and an information-theoretic gap,

  * saves all diagnostic matrices and spectra to a new .npz file for
    later analysis.

This is the finite-dimensional UIH Fisher-Lindblad decomposition on su(4)
for real hardware data.

Usage
-----
    python 42_qapi_two_qubit_k_bkm_fisher_split_checks.py \
        ibmq_results/ibmq_two_qubit_k_tomography_ibm_fez_20251119_225707_uih_split.npz

If no argument is given, a default path from the dataclass config is used.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------

@dataclass
class BKMUISplitConfig:
    uih_npz_path: str = (
        "ibmq_results/"
        "ibmq_two_qubit_k_tomography_ibm_fez_20251120_090502_6kshots_uih_split.npz"
    )
    # Floor for eigenvalues of rho when building the BKM kernel
    rho_eig_floor: float = 1e-12
    # Relative threshold for treating eigenvalues as equal in the kernel
    equal_tol: float = 1e-10
    verbose: bool = True


# ----------------------------------------------------------------------
# Pauli basis construction and rho reconstruction
# ----------------------------------------------------------------------

def single_qubit_paulis() -> List[np.ndarray]:
    """Return [I, X, Y, Z] as 2x2 complex arrays."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]


def two_qubit_paulis() -> List[np.ndarray]:
    """
    Build the 16 two-qubit Pauli matrices in row-major {I,X,Y,Z}⊗{I,X,Y,Z}
    order:

        idx = 4 * i + j  <->  P_idx = P_i ⊗ P_j.
    """
    paulis_1q = single_qubit_paulis()
    ops: List[np.ndarray] = []
    for i in range(4):
        for j in range(4):
            ops.append(np.kron(paulis_1q[i], paulis_1q[j]))
    return ops


def reconstruct_rho_from_vstat(
    v_stat: np.ndarray,
    pauli_ops: List[np.ndarray],
) -> np.ndarray:
    """
    Given a 16-component Pauli expectation vector v_stat with entries

        v_stat[k] = Tr(rho P_k),

    and a list of 16 two-qubit Pauli operators P_k ordered as in the
    tomography scripts, reconstruct the density matrix

        rho = 1/4 * sum_k v_stat[k] P_k.

    For a valid stationary state we expect v_stat[0] ~ 1.
    """
    if v_stat.shape[0] != 16:
        raise ValueError(f"Expected v_stat of length 16, got {v_stat.shape[0]}")
    if len(pauli_ops) != 16:
        raise ValueError(f"Expected 16 Pauli ops, got {len(pauli_ops)}")

    rho = np.zeros((4, 4), dtype=complex)
    for k in range(16):
        rho += v_stat[k] * pauli_ops[k]
    rho *= 0.25
    return rho


# ----------------------------------------------------------------------
# BKM / Kubo-Mori metric construction
# ----------------------------------------------------------------------

def bkm_kernel(p: float, q: float, equal_tol: float = 1e-10) -> float:
    """
    Kubo-Mori / BKM kernel c(p, q) for eigenvalues p, q > 0:

        c(p, q) = (p - q) / (log p - log q)   if p != q,
                = p                           if p == q.

    For nearly equal arguments, we switch to the p branch to avoid
    numerical 0/0.
    """
    if p <= 0.0 or q <= 0.0:
        raise ValueError("BKM kernel requires strictly positive arguments.")
    if abs(p - q) <= equal_tol * max(p, q):
        return float(p)
    return float((p - q) / (np.log(p) - np.log(q)))


def build_bkm_metric_on_traceless(
    rho: np.ndarray,
    pauli_ops: List[np.ndarray],
    eig_floor: float = 1e-12,
    equal_tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the 15x15 BKM / Fisher metric matrix on the traceless Pauli
    sector at the state rho.

    We work in the basis Q_j = P_j / 4 for j=1..15 (i.e. excluding I⊗I),
    with coordinates equal to Pauli expectation values v_j. The BKM
    inner product is

        <A, B>_BKM = sum_{a,b} c(λ_a, λ_b) A_{ab}^* B_{ab},

    where λ_a are the eigenvalues of rho, and A, B are expressed in the
    eigenbasis of rho.

    Returns
    -------
    M_bkm : (15, 15) ndarray
        Real symmetric positive definite metric matrix in the traceless
        Pauli basis.
    evals_rho : (4,) ndarray
        Clipped eigenvalues of rho used in the kernel.
    U : (4, 4) ndarray
        Unitary whose columns are eigenvectors of rho (rho = U diag λ U^†).
    """
    # Diagonalise rho
    evals_rho, U = np.linalg.eigh(rho)

    # Clip eigenvalues to enforce strict positivity for the kernel
    evals_clipped = np.clip(evals_rho.real, eig_floor, None)
    evals_clipped /= np.sum(evals_clipped)

    # Build BKM kernel matrix W_{ab} = c(λ_a, λ_b)
    W = np.zeros((4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            W[a, b] = bkm_kernel(evals_clipped[a], evals_clipped[b], equal_tol=equal_tol)

    # Build Q_j = P_j / 4 for j=1..15, then transform to eigenbasis
    idx_non_id = list(range(1, 16))
    Q_tilde: List[np.ndarray] = []
    for idx in idx_non_id:
        P = pauli_ops[idx]
        Q = P / 4.0
        Q_eig = U.conj().T @ Q @ U
        Q_tilde.append(Q_eig)

    # Metric matrix M_{ij} = <Q_i, Q_j>_BKM
    n = len(idx_non_id)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        Qi = Q_tilde[i]
        for j in range(i, n):
            Qj = Q_tilde[j]
            # sum_{a,b} W_{ab} Qi_{ab}^* Qj_{ab}
            val = 0.0 + 0.0j
            for a in range(4):
                for b in range(4):
                    val += W[a, b] * np.conj(Qi[a, b]) * Qj[a, b]
            # For Hermitian Q_i, Q_j this should be real; take real part
            M[i, j] = float(val.real)
            M[j, i] = M[i, j]

    return M, evals_clipped, U


# ----------------------------------------------------------------------
# Fisher G/J split in the BKM metric
# ----------------------------------------------------------------------

def metric_adjoint(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Compute the metric adjoint K^sharp with respect to the positive
    definite inner product <x, y>_M = x^T M y.

    By definition, K^sharp satisfies M K = (K^sharp)^T M, which yields

        K^sharp = M^{-1} K^T M.
    """
    Minv = np.linalg.inv(M)
    K_sharp = Minv @ K.T @ M
    return K_sharp.real


def fisher_split_bkm(K_reg: np.ndarray, M_bkm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a regularised generator K_reg and a BKM metric matrix M_bkm,
    compute the Fisher G/J split

        K^sharp = M^{-1} K^T M,
        G_bkm   = 0.5 * (K_reg + K^sharp),
        J_bkm   = 0.5 * (K_reg - K^sharp).

    Returns (G_bkm, J_bkm).
    """
    K_sharp = metric_adjoint(K_reg, M_bkm)
    G_bkm = 0.5 * (K_reg + K_sharp)
    J_bkm = 0.5 * (K_reg - K_sharp)
    return G_bkm, J_bkm


def cholesky_with_ridge(M: np.ndarray, max_tries: int = 5) -> Tuple[np.ndarray, float]:
    """
    Attempt a Cholesky factorisation M = L L^T, adding a small ridge
    epsilon * I if necessary to enforce positive definiteness.

    Returns (L, eps_used) where eps_used is zero if no ridge was needed.
    """
    eps = 0.0
    base = float(np.max(np.diag(M)))
    if base <= 0.0:
        base = 1.0
    for k in range(max_tries):
        try:
            L = np.linalg.cholesky(M + eps * np.eye(M.shape[0]))
            return L, eps
        except np.linalg.LinAlgError:
            if eps == 0.0:
                eps = 1e-12 * base
            else:
                eps *= 10.0
    # If we reach here, raise
    raise np.linalg.LinAlgError("Cholesky failed even after ridge regularisation.")


def orthonormalise_operator(G: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the operator G into coordinates where the metric M becomes
    the identity. If M = L L^T with L lower-triangular, and z = L^T x,
    then in z-coordinates the operator is

        G_ortho = L^T G L^{-T},

    which is symmetric for a metric-symmetric G.
    """
    L, eps = cholesky_with_ridge(M)
    if eps != 0.0:
        print(f"[UIH-BKM] Cholesky required ridge eps = {eps:.3e} to enforce PD metric.")
    LT = L.T
    LT_inv = np.linalg.inv(LT)
    G_ortho = LT @ G @ LT_inv
    return G_ortho, L


def analyse_g_spectrum(G_bkm: np.ndarray, M_bkm: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
    """
    Diagonalise G_bkm in BKM-orthonormal coordinates and extract:

      - the sorted real eigenvalues,
      - the slowest dissipative rate (largest negative eigenvalue),
      - the fastest dissipative rate (most negative eigenvalue),
      - the count of positive eigenvalues (if any).

    Returns (evals_sorted, lam_gap, lam_fast, num_pos).
    """
    G_ortho, _ = orthonormalise_operator(G_bkm, M_bkm)
    # Symmetrise explicitly
    G_ortho_sym = 0.5 * (G_ortho + G_ortho.T)
    evals = np.linalg.eigvalsh(G_ortho_sym)
    evals_sorted = np.sort(evals.real)

    negs = evals_sorted[evals_sorted < 0.0]
    if negs.size > 0:
        lam_gap = float(negs.max())
        lam_fast = float(negs.min())
    else:
        lam_gap = 0.0
        lam_fast = 0.0
    num_pos = int(np.sum(evals_sorted > 0.0))
    return evals_sorted, lam_gap, lam_fast, num_pos


# ----------------------------------------------------------------------
# Main analysis driver
# ----------------------------------------------------------------------

def run_bkm_split(cfg: BKMUISplitConfig) -> None:
    t0 = time.time()
    print(f"[UIH-BKM] Loading UIH split data from: {cfg.uih_npz_path}")
    if not os.path.exists(cfg.uih_npz_path):
        raise FileNotFoundError(f"Cannot find npz file at '{cfg.uih_npz_path}'")

    data = np.load(cfg.uih_npz_path)
    required = ["K_reg", "v_stat"]
    for key in required:
        if key not in data:
            raise KeyError(f"UIH split npz file is missing required array '{key}'")

    K_reg = data["K_reg"]
    v_stat = data["v_stat"]

    print(f"[UIH-BKM] K_reg shape = {K_reg.shape}, v_stat shape = {v_stat.shape}")

    # Build Pauli operators and reconstruct rho_ss
    pauli_ops = two_qubit_paulis()
    rho_ss = reconstruct_rho_from_vstat(v_stat, pauli_ops)

    # Basic rho diagnostics
    tr_rho = np.trace(rho_ss)
    rho_herm_dev = np.linalg.norm(rho_ss - rho_ss.conj().T)
    print(f"[UIH-BKM] Tr(rho_ss) = {tr_rho:.6f}")
    print(f"[UIH-BKM] Hermiticity check ||rho - rho^†||_F = {rho_herm_dev:.3e}")

    evals_rho, U_rho = np.linalg.eigh(rho_ss)
    print("[UIH-BKM] Eigenvalues of rho_ss (raw):")
    for lam in evals_rho:
        print(f"    {lam.real:.6e} (imag {lam.imag:.2e})")

    # Build BKM metric on traceless Pauli sector
    print("[UIH-BKM] Building BKM / Fisher metric on su(4)...")
    M_bkm, evals_rho_clipped, U = build_bkm_metric_on_traceless(
        rho_ss,
        pauli_ops,
        eig_floor=cfg.rho_eig_floor,
        equal_tol=cfg.equal_tol,
    )

    # Metric diagnostics
    sym_dev_M = np.linalg.norm(M_bkm - M_bkm.T)
    cond_M = np.linalg.cond(M_bkm)
    print(f"[UIH-BKM] Metric symmetry check ||M - M^T||_F = {sym_dev_M:.3e}")
    print(f"[UIH-BKM] Metric condition number cond(M_bkm) = {cond_M:.3e}")
    print("[UIH-BKM] Eigenvalues of rho_ss (clipped and renormalised):")
    for lam in evals_rho_clipped:
        print(f"    {lam:.6e}")

    # Fisher G/J split in BKM metric
    print("[UIH-BKM] Performing Fisher G/J split with respect to BKM metric...")
    G_bkm, J_bkm = fisher_split_bkm(K_reg, M_bkm)

    # Check metric symmetry / skewness
    MG = M_bkm @ G_bkm
    MJ = M_bkm @ J_bkm
    sym_dev_G = np.linalg.norm(MG - MG.T)
    skew_dev_J = np.linalg.norm(MJ + MJ.T)
    print(f"[UIH-BKM] Metric symmetry check: ||M G - (M G)^T||_F = {sym_dev_G:.3e}")
    print(f"[UIH-BKM] Metric skew check   : ||M J + (M J)^T||_F = {skew_dev_J:.3e}")

    # Spectral analysis of G_bkm
    print("[UIH-BKM] Analysing dissipative spectrum of G_bkm in BKM-orthonormal coordinates...")
    evals_G, lam_gap, lam_fast, num_pos = analyse_g_spectrum(G_bkm, M_bkm)
    print("[UIH-BKM] Eigenvalues of G_bkm (BKM-orthonormal basis, sorted):")
    for lam in evals_G:
        print(f"    {lam:.6f}")
    print(f"[UIH-BKM] Slowest dissipative rate (largest negative eigenvalue): {lam_gap:.6f}")
    print(f"[UIH-BKM] Fastest dissipative rate (most negative eigenvalue)  : {lam_fast:.6f}")
    print(f"[UIH-BKM] Number of positive eigenvalues (if any): {num_pos}")

    # Norm diagnostics
    norm_K = np.linalg.norm(K_reg, ord=2)
    norm_G = np.linalg.norm(G_bkm, ord=2)
    norm_J = np.linalg.norm(J_bkm, ord=2)
    print(f"[UIH-BKM] Operator norms: ||K_reg||_2 = {norm_K:.4f}, ||G_bkm||_2 = {norm_G:.4f}, ||J_bkm||_2 = {norm_J:.4f}")

    # Save everything
    out_base = os.path.splitext(cfg.uih_npz_path)[0] + "_bkm"
    out_npz = out_base + ".npz"

    np.savez_compressed(
        out_npz,
        K_reg=K_reg,
        rho_ss=rho_ss,
        rho_evals_raw=evals_rho,
        rho_evals_clipped=evals_rho_clipped,
        rho_evecs=U,
        M_bkm=M_bkm,
        G_bkm=G_bkm,
        J_bkm=J_bkm,
        G_bkm_evals=evals_G,
    )
    print(f"[UIH-BKM] Saved BKM Fisher split data to {out_npz}")

    t1 = time.time()
    print(f"[UIH-BKM] BKM Fisher analysis complete in {t1 - t0:.1f} seconds")


def main():
    if len(sys.argv) > 1:
        uih_npz_path = sys.argv[1]
    else:
        uih_npz_path = BKMUISplitConfig().uih_npz_path

    cfg = BKMUISplitConfig(uih_npz_path=uih_npz_path)
    run_bkm_split(cfg)


if __name__ == "__main__":
    main()
