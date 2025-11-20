#!/usr/bin/env python3
"""
43_qapi_two_qubit_k_cptp_repair_and_bkm_checks.py

CPTP repair and BKM Fisher analysis for a two-qubit K generator.

This script:

  * loads a .npz file produced by 40_qapi_two_qubit_k_tomography_ibmq_test.py,
    which contains:
        - T    : 16x16 Pauli transfer matrix (including identity),
        - T_tr : 15x15 block on traceless Pauli subspace,
        - K    : 15x15 matrix log of T_tr,
        - K_evals: eigenvalues of K,
        - V_in, V_out: input/output expectation matrices,

  * reconstructs a Choi matrix J_lin from T using the definition

        J = sum_{j,k} |j><k| ⊗ E(|j><k|),

    where E is the channel represented by T in the Pauli basis,

  * projects J_lin onto the CPTP cone by iterating:
        - Hermitisation,
        - eigenvalue clipping (CP projection),
        - partial trace correction to impose Tr_2(J) = I (TP projection),

  * constructs a repaired Pauli transfer matrix T_cp from the CPTP Choi
    using the identity

        T_{mu,nu} = (1/4) Tr[P_mu E_cp(P_nu)],

    for the two-qubit Pauli basis {I,X,Y,Z}⊗2,

  * recomputes a generator K_cp = log(T_cp,tr), performs the HS metric
    G/J split for diagnostics, and extracts the stationary Pauli vector
    v_stat_cp from T_cp,

  * builds the BKM / Kubo-Mori metric at the repaired stationary state,
    computes the metric adjoint K^sharp and the Fisher G_bkm, J_bkm
    split, and diagonalises G_bkm in BKM orthonormal coordinates to
    examine dissipative eigenvalues.

Usage
-----

    python 43_qapi_two_qubit_k_cptp_repair_and_bkm_checks.py \
        ibmq_results/ibmq_two_qubit_k_tomography_ibmq_fezz_YYYYMMDD_HHMMSS.npz

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
class CPTPRepairConfig:
    tomo_npz_path: str = (
        "ibmq_results/"
        "ibmq_two_qubit_k_tomography_ibm_fez_20251119_225707_4kshots.npz"
    )
    rho_eig_floor: float = 1e-6
    equal_tol: float = 1e-10
    cptp_iters: int = 10
    verbose: bool = True


# ----------------------------------------------------------------------
# Pauli basis and basic utilities
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


def matrix_log_diag(A: np.ndarray, lambda_floor: float = 1e-6) -> np.ndarray:
    """
    Matrix logarithm using eigen-decomposition, with a small-magnitude
    eigenvalue floor to avoid log(0) after CPTP projection.

    Any eigenvalue with |lambda| < lambda_floor is replaced by
    lambda_floor * exp(i * arg(lambda)) before taking the log.
    """
    vals, vecs = np.linalg.eig(A.astype(complex))

    mags = np.abs(vals)
    small = mags < lambda_floor
    if np.any(small):
        n_small = int(np.sum(small))
        print(
            f"[UIH-CPTP] Warning: clamping {n_small} eigenvalues "
            f"|lambda| < {lambda_floor} before log."
        )
        angles = np.angle(vals[small])
        vals[small] = lambda_floor * np.exp(1j * angles)

    L_vals = np.log(vals)
    V_inv = np.linalg.inv(vecs)
    L = vecs @ np.diag(L_vals) @ V_inv
    return L.real


# ----------------------------------------------------------------------
# Channel action from Pauli transfer matrix
# ----------------------------------------------------------------------

def apply_channel_from_T(
    T: np.ndarray,
    A: np.ndarray,
    pauli_ops: List[np.ndarray],
) -> np.ndarray:
    """
    Apply the channel represented by a 16x16 Pauli transfer matrix T to
    an arbitrary 4x4 operator A.

    We expand A in the Pauli basis P_nu, using

        A = sum_nu a_nu P_nu,  where a_nu = (1/4) Tr[P_nu A],

    then use E(A) = sum_mu P_mu (sum_nu T_{mu,nu} a_nu).
    """
    if T.shape != (16, 16):
        raise ValueError(f"Expected T of shape (16,16), got {T.shape}")
    if A.shape != (4, 4):
        raise ValueError(f"Expected A of shape (4,4), got {A.shape}")

    a = np.zeros(16, dtype=complex)
    for nu in range(16):
        a[nu] = 0.25 * np.trace(pauli_ops[nu] @ A)

    b = T @ a
    E_A = np.zeros((4, 4), dtype=complex)
    for mu in range(16):
        E_A += b[mu] * pauli_ops[mu]
    return E_A


# ----------------------------------------------------------------------
# Choi construction and partial trace
# ----------------------------------------------------------------------

def build_choi_from_T(T: np.ndarray, pauli_ops: List[np.ndarray]) -> np.ndarray:
    """
    Build the Choi matrix

        J = sum_{j,k} |j><k| ⊗ E(|j><k|),

    for a channel E represented by a Pauli transfer matrix T.

    The resulting J is a 16x16 complex matrix for a two-qubit channel.
    """
    d = 4
    J = np.zeros((d * d, d * d), dtype=complex)
    for j in range(d):
        for k in range(d):
            A_jk = np.zeros((d, d), dtype=complex)
            A_jk[j, k] = 1.0
            E_jk = apply_channel_from_T(T, A_jk, pauli_ops)
            J += np.kron(A_jk, E_jk)
    return J


def partial_trace_second(X: np.ndarray, d: int) -> np.ndarray:
    """
    Partial trace over the second subsystem for a d^2 x d^2 matrix X
    viewed as an operator on H_in ⊗ H_out with dim(H_in) = dim(H_out) = d.

    Returns Tr_2[X], a d x d matrix, with components
        (Tr_2 X)_{ik} = sum_j X_{ij,kj}.
    """
    X4 = X.reshape(d, d, d, d)  # indices: [i, j, k, l]
    out = np.zeros((d, d), dtype=complex)
    for j in range(d):
        out += X4[:, j, :, j]
    return out


# ----------------------------------------------------------------------
# CPTP projection for the Choi matrix
# ----------------------------------------------------------------------

def project_choi_to_cptp(
    J: np.ndarray,
    d: int,
    n_iters: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project a Choi matrix J towards the CPTP cone by alternating:

      - TP step: adjust J so that Tr_2(J) is closer to I_d,
      - CP step: project onto the positive cone by eigenvalue clipping.

    We start by hermitising J, then iterate:

        1) TP:  A = Tr_2(J), Delta = I_d - A,
                J <- J + (Delta/d) ⊗ I_d
        2) Hermitise: J <- (J + J^†)/2
        3) CP: eigen-decomposition, clip eigenvalues below 0 to 0.

    After n_iters, we hermitise and return the result. The final J_cptp
    is positive semidefinite to numerical tolerance and Tr_2(J_cptp)
    should be close to I_d.
    """
    J_cptp = 0.5 * (J + J.conj().T)
    I_d = np.eye(d, dtype=complex)

    evals_raw = None
    evals_clipped = None

    for _ in range(n_iters):
        # TP step: enforce Tr_2(J) ~ I_d
        A = partial_trace_second(J_cptp, d)
        Delta = I_d - A
        J_cptp = J_cptp + np.kron(Delta / d, I_d)

        # Hermitise
        J_cptp = 0.5 * (J_cptp + J_cptp.conj().T)

        # CP step via eigenvalue clipping
        evals_raw, V = np.linalg.eigh(J_cptp)
        evals_clipped = np.clip(evals_raw, 0.0, None)
        J_cptp = V @ np.diag(evals_clipped) @ V.conj().T

    # Final diagnostics
    A_final = partial_trace_second(J_cptp, d)
    J_cptp = 0.5 * (J_cptp + J_cptp.conj().T)

    return J_cptp, evals_raw, evals_clipped, A_final


# ----------------------------------------------------------------------
# Channel from Choi and repaired Pauli transfer matrix
# ----------------------------------------------------------------------

def apply_channel_from_choi(A: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Apply the channel with Choi matrix J to operator A.

    Convention:
      J = sum_{j,k} |j><k| ⊗ E(|j><k|),
      E(A) = Tr_1[(A^T ⊗ I) J].
    """
    d = A.shape[0]
    A_t = A.T
    big = np.kron(A_t, np.eye(d, dtype=complex))
    B_big = big @ J  # operator on H_in ⊗ H_out

    # Reshape to (i, a, j, b) and trace over the first subsystem (i index)
    B4 = B_big.reshape(d, d, d, d)
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        out += B4[i, :, i, :]

    return out


def pauli_transfer_from_choi(J: np.ndarray, pauli_ops: List[np.ndarray]) -> np.ndarray:
    """
    Construct the 16x16 Pauli transfer matrix T from a Choi matrix J
    in the {I,X,Y,Z}⊗{I,X,Y,Z} basis.

    T_{mu,nu} = (1/d) Tr( P_mu E(P_nu) ), with d = 4.
    """
    d = pauli_ops[0].shape[0]
    nP = len(pauli_ops)
    T = np.zeros((nP, nP), dtype=float)

    for nu in range(nP):
        A = pauli_ops[nu]
        B = apply_channel_from_choi(A, J)
        for mu in range(nP):
            T[mu, nu] = (np.trace(pauli_ops[mu] @ B).real) / d

    return T


# ----------------------------------------------------------------------
# Stationary Pauli vector and HS G/J analysis
# ----------------------------------------------------------------------

def stationary_pauli_vector(T: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, complex]:
    """
    Find an approximate stationary Pauli expectation vector for the full
    16x16 Pauli transfer matrix T by locating the eigenvector with
    eigenvalue closest to 1.

    Returns (v_stat, lam_closest), with v_stat normalised so that
    v_stat[0] ~ 1.
    """
    evals, vecs = np.linalg.eig(T)
    idx = np.argmin(np.abs(evals - 1.0))
    lam = evals[idx]
    v = vecs[:, idx]
    if abs(v[0]) > tol:
        v = v / v[0]
    return v.real, lam


def spectral_regularise_K(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral regularisation of a generator K.

    Diagonalise K, clamp positive real parts of eigenvalues to zero,
    reconstruct K_reg and return (K_reg, evals_raw, evals_reg).
    """
    evals, vecs = np.linalg.eig(K.astype(complex))
    evals_reg = evals.copy()
    for i, lam in enumerate(evals_reg):
        if lam.real > 0:
            evals_reg[i] = complex(0.0, lam.imag)
    vecs_inv = np.linalg.inv(vecs)
    K_reg = vecs @ np.diag(evals_reg) @ vecs_inv
    return K_reg.real, evals, evals_reg


def hs_metric_split(K_reg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hilbert Schmidt metric split:

        G = 0.5 (K_reg + K_reg^T),
        J = 0.5 (K_reg - K_reg^T).
    """
    G = 0.5 * (K_reg + K_reg.T)
    J = 0.5 * (K_reg - K_reg.T)
    return G, J


# ----------------------------------------------------------------------
# BKM metric and Fisher G/J split (as in script 42)
# ----------------------------------------------------------------------

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


def bkm_kernel(p: float, q: float, equal_tol: float = 1e-10) -> float:
    """
    BKM kernel c(p, q) for eigenvalues p, q > 0:

        c(p, q) = (p - q) / (log p - log q)   if p != q,
                = p                           if p == q.
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
    Build the 15x15 BKM metric matrix on the traceless Pauli sector at rho.
    """
    evals_rho, U = np.linalg.eigh(rho)
    evals_clipped = np.clip(evals_rho.real, eig_floor, None)
    evals_clipped /= np.sum(evals_clipped)

    W = np.zeros((4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            W[a, b] = bkm_kernel(evals_clipped[a], evals_clipped[b], equal_tol=equal_tol)

    idx_non_id = list(range(1, 16))
    Q_tilde: List[np.ndarray] = []
    for idx in idx_non_id:
        P = pauli_ops[idx]
        Q = P / 4.0
        Q_eig = U.conj().T @ Q @ U
        Q_tilde.append(Q_eig)

    n = len(idx_non_id)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        Qi = Q_tilde[i]
        for j in range(i, n):
            Qj = Q_tilde[j]
            val = 0.0 + 0.0j
            for a in range(4):
                for b in range(4):
                    val += W[a, b] * np.conj(Qi[a, b]) * Qj[a, b]
            M[i, j] = float(val.real)
            M[j, i] = M[i, j]

    return M, evals_clipped, U


def metric_adjoint(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Metric adjoint K^sharp with respect to <x, y>_M = x^T M y.
    """
    Minv = np.linalg.inv(M)
    K_sharp = Minv @ K.T @ M
    return K_sharp.real


def fisher_split_bkm(K_reg: np.ndarray, M_bkm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fisher G/J split in BKM metric:

        K^sharp = M^{-1} K^T M,
        G_bkm   = 0.5 * (K_reg + K^sharp),
        J_bkm   = 0.5 * (K_reg - K^sharp).
    """
    K_sharp = metric_adjoint(K_reg, M_bkm)
    G_bkm = 0.5 * (K_reg + K_sharp)
    J_bkm = 0.5 * (K_reg - K_sharp)
    return G_bkm, J_bkm


def cholesky_with_ridge(M: np.ndarray, max_tries: int = 5) -> Tuple[np.ndarray, float]:
    """
    Cholesky factorisation M = L L^T, adding a small ridge if needed.
    """
    eps = 0.0
    base = float(np.max(np.diag(M)))
    if base <= 0.0:
        base = 1.0
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(M + eps * np.eye(M.shape[0]))
            return L, eps
        except np.linalg.LinAlgError:
            if eps == 0.0:
                eps = 1e-12 * base
            else:
                eps *= 10.0
    raise np.linalg.LinAlgError("Cholesky failed even after ridge regularisation.")


def orthonormalise_operator(G: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform G into coordinates where the metric M is the identity.
    """
    L, eps = cholesky_with_ridge(M)
    if eps != 0.0:
        print(f"[UIH-CPTP] Cholesky required ridge eps = {eps:.3e} to enforce PD metric.")
    LT = L.T
    LT_inv = np.linalg.inv(LT)
    G_ortho = LT @ G @ LT_inv
    return G_ortho, L


def analyse_g_spectrum(G_bkm: np.ndarray, M_bkm: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
    """
    Diagonalise G_bkm in BKM orthonormal coordinates and extract:

      evals_sorted, lam_gap, lam_fast, num_pos.
    """
    G_ortho, _ = orthonormalise_operator(G_bkm, M_bkm)
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
# Main driver
# ----------------------------------------------------------------------

def run_cptp_repair_and_bkm(cfg: CPTPRepairConfig) -> None:
    t0 = time.time()
    print(f"[UIH-CPTP] Loading tomography data from: {cfg.tomo_npz_path}")
    if not os.path.exists(cfg.tomo_npz_path):
        raise FileNotFoundError(f"Cannot find npz file at '{cfg.tomo_npz_path}'")

    data = np.load(cfg.tomo_npz_path)
    if "T" not in data:
        raise KeyError("Tomography npz is missing 'T' array")

    T = data["T"]
    print(f"[UIH-CPTP] T shape = {T.shape}")

    pauli_ops = two_qubit_paulis()

    # Build linear inversion Choi
    print("[UIH-CPTP] Building Choi matrix from T (linear inversion)...")
    J_lin = build_choi_from_T(T, pauli_ops)

    # Diagnostics on J_lin
    J_lin_herm_dev = np.linalg.norm(J_lin - J_lin.conj().T)
    evals_J_lin = np.linalg.eigvalsh(0.5 * (J_lin + J_lin.conj().T))
    print(f"[UIH-CPTP] Hermiticity check ||J_lin - J_lin^†||_F = {J_lin_herm_dev:.3e}")
    print("[UIH-CPTP] Eigenvalues of Herm(J_lin) (first few):")
    for lam in evals_J_lin[:8]:
        print(f"    {lam:.6e}")
    print(f"[UIH-CPTP] Minimal eigenvalue of Herm(J_lin): {evals_J_lin.min():.6e}")

    # CPTP projection
    print("[UIH-CPTP] Projecting Choi to CPTP with alternating CP + TP steps...")
    d = 4
    J_cptp, evals_raw, evals_clipped, A_final = project_choi_to_cptp(
        J_lin, d, n_iters=cfg.cptp_iters
    )

    J_cptp_herm_dev = np.linalg.norm(J_cptp - J_cptp.conj().T)
    evals_J_cptp = np.linalg.eigvalsh(J_cptp)
    print(f"[UIH-CPTP] Hermiticity check ||J_cptp - J_cptp^†||_F = {J_cptp_herm_dev:.3e}")
    print("[UIH-CPTP] Eigenvalues of J_cptp (first few):")
    for lam in evals_J_cptp[:8]:
        print(f"    {lam:.6e}")
    print(f"[UIH-CPTP] Minimal eigenvalue of J_cptp: {evals_J_cptp.min():.6e}")
    print("[UIH-CPTP] Final partial trace Tr_2(J_cptp) compared to identity:")
    print(f"    A_final =\n{A_final}")
    print(f"    ||A_final - I||_F = {np.linalg.norm(A_final - np.eye(d)):.3e}")

    # Build repaired Pauli transfer matrix from J_cptp
    print("[UIH-CPTP] Building repaired Pauli transfer matrix T_cp from J_cptp...")
    T_cp = pauli_transfer_from_choi(J_cptp, pauli_ops)

    # Basic diagnostics comparing T_cp to original T
    diff_T = np.linalg.norm(T_cp - T)
    print(f"[UIH-CPTP] ||T_cp - T||_F = {diff_T:.3e}")
    print(f"[UIH-CPTP] T_cp[0,0] = {T_cp[0,0]:.6f} (should be ~1)")
    print(f"[UIH-CPTP] Row 0 of T_cp (first few): {T_cp[0, :4]}")

    # Generator and HS G/J split from T_cp
    print("[UIH-CPTP] Computing K_cp = log(T_cp,tr) and HS G/J split...")
    idx_non_id = list(range(1, 16))
    T_tr_cp = T_cp[np.ix_(idx_non_id, idx_non_id)]
    K_cp = matrix_log_diag(T_tr_cp)
    K_reg_cp, evals_K_raw, evals_K_reg = spectral_regularise_K(K_cp)

    print("[UIH-CPTP] Raw K_cp eigenvalues (first few):")
    for lam in evals_K_raw[:10]:
        print(f"    {lam}")
    num_pos_before = int(np.sum(evals_K_raw.real > 0))
    num_pos_after = int(np.sum(evals_K_reg.real > 0))
    print(f"[UIH-CPTP] Positive real eigenvalues of K_cp: before = {num_pos_before}, after = {num_pos_after}")

    G_hs_cp, J_hs_cp = hs_metric_split(K_reg_cp)
    sym_dev_G = np.linalg.norm(G_hs_cp - G_hs_cp.T)
    skew_dev_J = np.linalg.norm(J_hs_cp + J_hs_cp.T)
    print(f"[UIH-CPTP] HS symmetry check: ||G - G^T||_F = {sym_dev_G:.3e}")
    print(f"[UIH-CPTP] HS skew check    : ||J + J^T||_F = {skew_dev_J:.3e}")

    evals_G_hs = np.linalg.eigvals(G_hs_cp)
    evG_real = np.sort(evals_G_hs.real)
    negs_hs = evG_real[evG_real < 0.0]
    if negs_hs.size > 0:
        lam_gap_hs = float(negs_hs.max())
        lam_fast_hs = float(negs_hs.min())
    else:
        lam_gap_hs = 0.0
        lam_fast_hs = 0.0
    num_pos_hs = int(np.sum(evG_real > 0.0))

    print("[UIH-CPTP] HS G eigenvalues (real parts, sorted):")
    for lam in evG_real:
        print(f"    {lam:.6f}")
    print(f"[UIH-CPTP] HS slowest dissipative rate (largest negative): {lam_gap_hs:.6f}")
    print(f"[UIH-CPTP] HS fastest dissipative rate (most negative)  : {lam_fast_hs:.6f}")
    print(f"[UIH-CPTP] HS number of positive eigenvalues: {num_pos_hs}")

    # Stationary Pauli vector and rho_ss_cp
    print("[UIH-CPTP] Extracting stationary Pauli expectation vector for T_cp...")
    v_stat_cp, lam_stat_cp = stationary_pauli_vector(T_cp)
    print(f"[UIH-CPTP] Stationary eigenvalue closest to 1: {lam_stat_cp}")
    print("[UIH-CPTP] First few components of v_stat_cp:")
    for i in range(8):
        print(f"    v_stat_cp[{i}] = {v_stat_cp[i]: .6f}")

    rho_ss_cp = reconstruct_rho_from_vstat(v_stat_cp, pauli_ops)
    tr_rho = np.trace(rho_ss_cp)
    rho_herm_dev = np.linalg.norm(rho_ss_cp - rho_ss_cp.conj().T)
    print(f"[UIH-CPTP] Tr(rho_ss_cp) = {tr_rho:.6f}")
    print(f"[UIH-CPTP] Hermiticity check ||rho - rho^†||_F = {rho_herm_dev:.3e}")
    evals_rho_cp, _ = np.linalg.eigh(rho_ss_cp)
    print("[UIH-CPTP] Eigenvalues of rho_ss_cp (raw):")
    for lam in evals_rho_cp:
        print(f"    {lam.real:.6e} (imag {lam.imag:.2e})")

    # BKM metric and Fisher G/J split on repaired channel
    print("[UIH-CPTP] Building BKM / Fisher metric at rho_ss_cp...")
    M_bkm, evals_rho_clipped, _ = build_bkm_metric_on_traceless(
        rho_ss_cp,
        pauli_ops,
        eig_floor=cfg.rho_eig_floor,
        equal_tol=cfg.equal_tol,
    )
    sym_dev_M = np.linalg.norm(M_bkm - M_bkm.T)
    cond_M = np.linalg.cond(M_bkm)
    print(f"[UIH-CPTP] Metric symmetry check ||M - M^T||_F = {sym_dev_M:.3e}")
    print(f"[UIH-CPTP] Metric condition number cond(M_bkm) = {cond_M:.3e}")
    print("[UIH-CPTP] Eigenvalues of rho_ss_cp (clipped and renormalised):")
    for lam in evals_rho_clipped:
        print(f"    {lam:.6e}")

    print("[UIH-CPTP] Performing Fisher G/J split with respect to BKM metric...")
    G_bkm, J_bkm = fisher_split_bkm(K_reg_cp, M_bkm)

    MG = M_bkm @ G_bkm
    MJ = M_bkm @ J_bkm
    sym_dev_G_bkm = np.linalg.norm(MG - MG.T)
    skew_dev_J_bkm = np.linalg.norm(MJ + MJ.T)
    print(f"[UIH-CPTP] Metric symmetry check: ||M G - (M G)^T||_F = {sym_dev_G_bkm:.3e}")
    print(f"[UIH-CPTP] Metric skew check   : ||M J + (M J)^T||_F = {skew_dev_J_bkm:.3e}")

    print("[UIH-CPTP] Analysing dissipative spectrum of G_bkm in BKM orthonormal coordinates...")
    evals_G_bkm, lam_gap_bkm, lam_fast_bkm, num_pos_bkm = analyse_g_spectrum(G_bkm, M_bkm)
    print("[UIH-CPTP] Eigenvalues of G_bkm (sorted):")
    for lam in evals_G_bkm:
        print(f"    {lam:.6f}")
    print(f"[UIH-CPTP] BKM slowest dissipative rate (largest negative): {lam_gap_bkm:.6f}")
    print(f"[UIH-CPTP] BKM fastest dissipative rate (most negative)  : {lam_fast_bkm:.6f}")
    print(f"[UIH-CPTP] BKM number of positive eigenvalues: {num_pos_bkm}")

    norm_K = np.linalg.norm(K_reg_cp, ord=2)
    norm_G = np.linalg.norm(G_bkm, ord=2)
    norm_J = np.linalg.norm(J_bkm, ord=2)
    print(f"[UIH-CPTP] Operator norms: ||K_reg_cp||_2 = {norm_K:.4f}, ||G_bkm||_2 = {norm_G:.4f}, ||J_bkm||_2 = {norm_J:.4f}")

    # Save repaired objects
    out_base = os.path.splitext(cfg.tomo_npz_path)[0] + "_cptp"
    out_npz = out_base + ".npz"
    np.savez_compressed(
        out_npz,
        T_cp=T_cp,
        J_lin=J_lin,
        J_cptp=J_cptp,
        K_cp=K_cp,
        K_reg_cp=K_reg_cp,
        G_hs_cp=G_hs_cp,
        J_hs_cp=J_hs_cp,
        v_stat_cp=v_stat_cp,
        rho_ss_cp=rho_ss_cp,
        M_bkm=M_bkm,
        G_bkm=G_bkm,
        J_bkm=J_bkm,
        G_bkm_evals=evals_G_bkm,
    )
    print(f"[UIH-CPTP] Saved repaired channel and Fisher data to {out_npz}")

    t1 = time.time()
    print(f"[UIH-CPTP] CPTP repair and BKM analysis complete in {t1 - t0:.1f} seconds")


def main():
    if len(sys.argv) > 1:
        tomo_npz_path = sys.argv[1]
    else:
        tomo_npz_path = CPTPRepairConfig().tomo_npz_path

    cfg = CPTPRepairConfig(tomo_npz_path=tomo_npz_path)
    run_cptp_repair_and_bkm(cfg)


if __name__ == "__main__":
    main()
