#!/usr/bin/env python3
"""
41_qapi_two_qubit_k_uih_metric_split_checks.py

UIH-style metric, G/J split and gap analysis for a two-qubit K generator
obtained from IBM process tomography.

This script:

  * loads a .npz file produced by 40_qapi_two_qubit_k_tomography_ibmq_test.py,
    which contains:
        - T      : 16x16 Pauli transfer matrix (including identity),
        - T_tr   : 15x15 block on traceless Pauli subspace,
        - K      : 15x15 matrix log of T_tr,
        - K_evals: eigenvalues of K,
        - V_in, V_out: input/output expectation matrices,

  * performs a simple spectral regularisation of K by clamping
        Re(lambda) -> min(Re(lambda), 0)
    on its eigenvalues, producing a modified generator K_reg,

  * carries out a G/J split with respect to the Hilbert–Schmidt metric:
        G = 0.5 (K_reg + K_reg^T)
        J = 0.5 (K_reg - K_reg^T),

  * analyses the spectrum of G, identifies the slowest dissipative modes,
    and reports an effective dissipative gap on su(4),

  * extracts the eigenvector of T with eigenvalue closest to 1 to serve
    as an approximate stationary state in the Pauli basis, which can be
    used in later work to build the true BKM/Fisher metric on operators.

This is a post-processing step for real hardware data, intended as a
bridge between raw tomography and the full Fisher–Lindblad story in
Universal Information Hydrodynamics.
"""

import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UIHMetricSplitConfig:
    npz_path: str = "ibmq_results/ibmq_two_qubit_k_tomography_ibm_fez_20251120_090502_6kshots.npz"
    clamp_positive_re: bool = True
    verbose: bool = True


def load_tomography_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find npz file at '{path}'")
    data = np.load(path)
    required = ["T", "T_tr", "K", "K_evals", "V_in", "V_out"]
    for key in required:
        if key not in data:
            raise KeyError(f"npz file is missing required array '{key}'")
    return data


def spectral_regularise_K(K: np.ndarray, clamp_positive_re: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral regularisation of a generator K.

    Diagonalise K, optionally clamp the real parts of its eigenvalues to
    be non-positive, and reconstruct a modified generator K_reg.

    Returns (K_reg, evals_raw, evals_reg).
    """
    evals, vecs = np.linalg.eig(K)
    evals_reg = evals.copy()

    if clamp_positive_re:
        for i, lam in enumerate(evals_reg):
            if lam.real > 0:
                evals_reg[i] = complex(0.0, lam.imag)

    vecs_inv = np.linalg.inv(vecs)
    K_reg = vecs @ np.diag(evals_reg) @ vecs_inv

    return K_reg.real, evals, evals_reg


def hs_metric_split(K_reg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hilbert–Schmidt metric split.

    With respect to the HS metric, the adjoint of K is just its transpose.
    The symmetric and antisymmetric parts of K_reg are therefore

        G = 0.5 (K_reg + K_reg^T),
        J = 0.5 (K_reg - K_reg^T).

    Returns (G, J).
    """
    G = 0.5 * (K_reg + K_reg.T)
    J = 0.5 * (K_reg - K_reg.T)
    return G, J


def stationary_pauli_vector(T: np.ndarray, tol: float = 1e-3) -> Tuple[np.ndarray, float]:
    """
    Find an approximate stationary Pauli expectation vector for the full
    16x16 Pauli transfer matrix T by locating the eigenvector with
    eigenvalue closest to 1.

    Returns (v_stat, lam_closest), where v_stat is normalised so that
    v_stat[0] ~ 1 (the I⊗I component).
    """
    evals, vecs = np.linalg.eig(T)
    idx = np.argmin(np.abs(evals - 1.0))
    lam = evals[idx]
    v = vecs[:, idx]

    # normalise so that the identity component is 1 in magnitude
    if abs(v[0]) > tol:
        v = v / v[0]
    return v.real, lam


def analyse_gap(evals_G: np.ndarray) -> Tuple[float, float]:
    """
    Given eigenvalues of G (real symmetric in HS metric), compute:
      - largest negative eigenvalue (closest to zero from below),
      - most negative eigenvalue (fastest decay direction).
    """
    real_parts = evals_G.real
    negs = real_parts[real_parts < 0]
    if negs.size == 0:
        return 0.0, 0.0
    lam_min = negs.max()   # closest to zero
    lam_fast = negs.min()  # most negative
    return float(lam_min), float(lam_fast)


def run_analysis(cfg: UIHMetricSplitConfig) -> None:
    t0 = time.time()
    print(f"[UIH] Loading tomography data from: {cfg.npz_path}")
    data = load_tomography_npz(cfg.npz_path)

    T = data["T"]
    T_tr = data["T_tr"]
    K = data["K"]
    K_evals = data["K_evals"]

    print("[UIH] Shapes: T =", T.shape, ", T_tr =", T_tr.shape, ", K =", K.shape)
    print("[UIH] Raw K eigenvalues (first few):")
    for lam in K_evals[:10]:
        print(f"    {lam}")

    # Spectral regularisation of K
    print("[UIH] Performing spectral regularisation of K...")
    K_reg, evals_raw, evals_reg = spectral_regularise_K(K, clamp_positive_re=cfg.clamp_positive_re)

    num_pos_before = np.sum(evals_raw.real > 0)
    num_pos_after = np.sum(evals_reg.real > 0)
    print(f"[UIH] Positive real eigenvalues of K: before = {int(num_pos_before)}, after = {int(num_pos_after)}")

    # Hilbert–Schmidt metric split
    print("[UIH] Performing HS metric G/J split...")
    G, J = hs_metric_split(K_reg)

    sym_dev = np.linalg.norm(G - G.T)
    skew_dev = np.linalg.norm(J + J.T)
    print(f"[UIH] Symmetry check: ||G - G^T||_F = {sym_dev:.3e}, ||J + J^T||_F = {skew_dev:.3e}")

    # Spectra
    evals_G = np.linalg.eigvals(G)
    evG_real = np.sort(evals_G.real)
    lam_gap, lam_fast = analyse_gap(evals_G)

    print("[UIH] Eigenvalues of G (real parts, sorted):")
    for lam in evG_real:
        print(f"    {lam:.6f}")

    print(f"[UIH] Slowest dissipative rate (largest negative real part of G): {lam_gap:.6f}")
    print(f"[UIH] Fastest dissipative rate (most negative real part of G):   {lam_fast:.6f}")

    # Norm diagnostics
    norm_K = np.linalg.norm(K_reg, ord=2)
    norm_G = np.linalg.norm(G, ord=2)
    norm_J = np.linalg.norm(J, ord=2)
    print(f"[UIH] Operator norms: ||K_reg||_2 = {norm_K:.4f}, ||G||_2 = {norm_G:.4f}, ||J||_2 = {norm_J:.4f}")

    # Stationary Pauli vector for future BKM metric work
    print("[UIH] Extracting approximate stationary Pauli expectation vector...")
    v_stat, lam_stat = stationary_pauli_vector(T)
    print(f"[UIH] Stationary eigenvalue closest to 1: {lam_stat}")
    print("[UIH] First few components of v_stat (Pauli basis {I,X,Y,Z}⊗2):")
    for i in range(8):
        print(f"    v_stat[{i}] = {v_stat[i]: .6f}")

    # Save post-processed generator and G/J if desired
    out_base = os.path.splitext(cfg.npz_path)[0] + "_uih_split"
    out_npz = out_base + ".npz"

    np.savez_compressed(
        out_npz,
        K_reg=K_reg,
        K_evals_raw=evals_raw,
        K_evals_reg=evals_reg,
        G_hs=G,
        J_hs=J,
        G_evals=evals_G,
        v_stat=v_stat,
        lam_stat=lam_stat,
    )
    print(f"[UIH] Saved UIH split data to {out_npz}")

    t1 = time.time()
    print(f"[UIH] Analysis complete in {t1 - t0:.1f} seconds")


def main():
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        npz_path = UIHMetricSplitConfig().npz_path

    cfg = UIHMetricSplitConfig(npz_path=npz_path)
    run_analysis(cfg)


if __name__ == "__main__":
    main()
