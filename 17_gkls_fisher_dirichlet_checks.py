#!/usr/bin/env python3
"""
17_gkls_fisher_dirichlet_checks.py

Fisher–Dirichlet matching for diagonal GKLS models with detailed balance.

For each random reversible Markov model (pi, Q):

  1. Build a GKLS generator (purely dissipative) with jump operators
     L_ij = sqrt(w[j,i]) |i><j|, where w[i,j] are classical rates.

  2. Construct the BKM (Kubo–Mori) metric M on the operator space, which is
     the Hessian at rho_ss = diag(pi) of the quantum relative entropy
     S(rho || rho_ss).

  3. Form the metric adjoint K^sharp = M^{-1} K^† M and the symmetric part
     G = (K + K^sharp)/2.

  4. Restrict to diagonal perturbations delta rho = diag(delta p), and
     compare:

        E_classical(delta p) =
            1/2 sum_{i != j} pi_i w[i,j] (delta p_j/pi_j - delta p_i/pi_i)^2

        E_GKLS(delta p) =
            - <delta u, M G delta u>   (delta u = vec(delta rho))

     for many random mass-conserving delta p.

The script reports the maximum absolute and relative difference between the
two energies across the ensemble.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng


# --------------------------------------------------------------------
# Basic utilities: pi, flux, Markov rates, GKLS generator
# --------------------------------------------------------------------

def make_random_pi(N, rng):
    """Random stationary distribution with all entries bounded away from 0."""
    x = rng.random(N)
    pi = x / x.sum()
    eps = 1e-3 / N
    pi = np.clip(pi, eps, None)
    pi = pi / pi.sum()
    return pi


def make_random_flux(N, rng, rate_scale=1.0):
    """Random symmetric flux matrix F_ij >= 0 with zero diagonal."""
    F = rng.random((N, N))
    F = 0.5 * (F + F.T)
    np.fill_diagonal(F, 0.0)
    return F * rate_scale


def build_w_and_Q(pi, F):
    """
    Given stationary distribution pi and symmetric flux F_ij, build:

      - w[i, j]: classical rate i -> j
      - Q: generator for populations p (column vector) with dp/dt = Q p

    Detailed balance: pi[i] * w[i, j] = F_ij = pi[j] * w[j, i].
    """
    pi = np.asarray(pi, dtype=float)
    F = np.asarray(F, dtype=float)
    N = len(pi)
    w = np.zeros((N, N), dtype=float)
    Q = np.zeros((N, N), dtype=float)

    # Off diagonal rates from flux
    for i in range(N):
        for j in range(N):
            if i != j:
                if pi[i] <= 0.0:
                    raise ValueError("pi has non positive entry")
                w[i, j] = F[i, j] / pi[i]

    # Generator Q: Q[i, j] = w[j, i] for i != j
    for i in range(N):
        for j in range(N):
            if i != j:
                Q[i, j] = w[j, i]

    # Diagonal entries: Q[i, i] = - sum_{j != i} w[i, j]
    for i in range(N):
        Q[i, i] = -np.sum(w[i, :]) + w[i, i]

    return w, Q


def lindblad_superoperator(H, w):
    """
    Build the Lindblad superoperator K for diagonal jump model with Hamiltonian H and rates w.

    H is an N x N Hermitian matrix.
    w[i, j] is the classical rate i -> j.
    The jump operators are L_ij = sqrt(w[j, i]) |i><j| for i != j.
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    dim = N * N
    K = np.zeros((dim, dim), dtype=complex)

    # Hamiltonian part: -i [H, rho]
    if np.any(np.abs(H) > 0):
        I = np.eye(N, dtype=complex)
        K_H = -1j * (np.kron(I, H) - np.kron(H.T, I))
        K += K_H

    # Jump operators
    I_N = np.eye(N, dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rate = w[j, i]  # j -> i
            if rate <= 0.0:
                continue
            L = np.zeros((N, N), dtype=complex)
            L[i, j] = np.sqrt(rate)
            Ld = L.conj().T
            LdL = Ld @ L
            # vec(L rho L^dagger) = (L_bar \otimes L) vec(rho)
            term_jump = np.kron(L.conj(), L)
            # vec(L^dagger L rho) = (I \otimes L^dagger L) vec(rho)
            term_left = np.kron(I_N, LdL)
            # vec(rho L^dagger L) = ((L^dagger L)^T \otimes I) vec(rho)
            term_right = np.kron(LdL.T, I_N)
            K += term_jump - 0.5 * (term_left + term_right)

    return K


def vec(rho):
    """Vectorise density matrix rho in column stacking convention."""
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


# --------------------------------------------------------------------
# BKM metric and Dirichlet energies
# --------------------------------------------------------------------

def bkm_weights(pi):
    """
    BKM (Kubo–Mori) weights c_ij for Hessian of S(rho || diag(pi)) at rho = diag(pi).

    In the eigenbasis of rho_ss = diag(pi), the Hessian metric is diagonal in the
    matrix unit basis E_ij, with

          c_ij = (log pi_i - log pi_j) / (pi_i - pi_j)     for i != j
          c_ii = 1 / pi_i                                  (continuous limit)

    This yields the BKM information metric:
        g(A, B) = sum_{i, j} conj(A_ij) B_ij c_ij.
    """
    pi = np.asarray(pi, dtype=float)
    N = len(pi)
    C = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if abs(pi[i] - pi[j]) < 1e-14:
                C[i, j] = 1.0 / pi[i]
            else:
                C[i, j] = (np.log(pi[i]) - np.log(pi[j])) / (pi[i] - pi[j])
    return C


def classical_dirichlet(delta_p, pi, w):
    """
    Classical Fisher–Dirichlet energy for a reversible Markov chain
    with stationary distribution pi and rates w[i, j] (i -> j).

      delta_p: perturbation of probabilities (sum_i delta_p_i = 0 recommended)
    """
    pi = np.asarray(pi, dtype=float)
    w = np.asarray(w, dtype=float)
    N = len(pi)
    delta_p = np.asarray(delta_p, dtype=float)
    phi = delta_p / pi  # tangent in log tilt coordinates

    E = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            E += 0.5 * pi[i] * w[i, j] * (phi[j] - phi[i]) ** 2
    return float(E)


def gkls_dirichlet(delta_p, G, M_diag):
    """
    GKLS Fisher–Dirichlet energy using symmetric part G and BKM metric M.

      E_GKLS = - <delta u, M G delta u>,    delta u = vec(diag(delta_p)).
    """
    N = len(delta_p)
    delta_rho = np.diag(delta_p)
    delta_u = vec(delta_rho)  # complex vector
    MG_du = M_diag * (G @ delta_u)
    E = -np.vdot(delta_u, MG_du).real
    return float(E)


# --------------------------------------------------------------------
# Per-model worker
# --------------------------------------------------------------------

def run_single_model(model_index, N, n_tests, rate_scale, seed):
    rng = default_rng(seed + 1000 * model_index)

    # Random reversible Markov model
    pi = make_random_pi(N, rng)
    F = make_random_flux(N, rng, rate_scale=rate_scale)
    w, Q = build_w_and_Q(pi, F)

    # Stationarity check for Q
    stat_resid = float(np.linalg.norm(Q @ pi))

    # GKLS generator with H = 0 (purely diagonal jumps in energy basis)
    H = np.zeros((N, N), dtype=complex)
    K = lindblad_superoperator(H, w)

    # BKM metric in E_ij basis
    C = bkm_weights(pi)
    # Vectorise with the same (i, j) -> index mapping as vec: column-major
    M_diag = C.reshape(-1, order="F")
    M_inv = 1.0 / M_diag

    # Metric adjoint and symmetric part
    Kdag = K.conj().T
    # K^sharp_ij = M_inv[i] * Kdag_ij * M_diag[j]
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G = 0.5 * (K + Ksharp)

    max_abs_err = 0.0
    max_rel_err = 0.0
    min_Ec = float("inf")
    min_Eg = float("inf")
    max_Ec = 0.0
    max_Eg = 0.0

    for _ in range(n_tests):
        # Random mass-conserving perturbation
        x = rng.normal(size=N)
        x -= x.mean()

        Ec = classical_dirichlet(x, pi, w)
        Eg = gkls_dirichlet(x, G, M_diag)

        max_abs_err = max(max_abs_err, abs(Eg - Ec))
        if abs(Ec) > 1e-14:
            rel = abs(Eg - Ec) / abs(Ec)
            max_rel_err = max(max_rel_err, rel)

        min_Ec = min(min_Ec, Ec)
        min_Eg = min(min_Eg, Eg)
        max_Ec = max(max_Ec, Ec)
        max_Eg = max(max_Eg, Eg)

    return {
        "index": model_index,
        "stat_resid": stat_resid,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "min_Ec": min_Ec,
        "min_Eg": min_Eg,
        "max_Ec": max_Ec,
        "max_Eg": max_Eg,
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fisher–Dirichlet checks: GKLS density sector vs classical reversible Markov chain."
    )
    parser.add_argument("--N", type=int, default=3, help="Hilbert/Markov dimension")
    parser.add_argument("--n_models", type=int, default=10, help="Number of random models in the ensemble")
    parser.add_argument("--n_tests", type=int, default=20, help="Number of random perturbations per model")
    parser.add_argument("--rate_scale", type=float, default=1.0, help="Overall scale for random rates")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed")
    parser.add_argument("--n_jobs", type=int, default=0, help="Number of parallel workers (0 for auto)")
    args = parser.parse_args()

    N = args.N
    n_models = args.n_models
    n_tests = args.n_tests
    rate_scale = args.rate_scale
    seed = args.seed

    if args.n_jobs > 0:
        n_jobs = args.n_jobs
    else:
        cpu_count = mp.cpu_count()
        n_jobs = min(20, cpu_count)

    print("gkls_fisher_dirichlet_checks.py")
    print("--------------------------------")
    print(f"Dimension N                   = {N}")
    print(f"Ensemble size n_models        = {n_models}")
    print(f"Perturbations per model       = {n_tests}")
    print(f"Rate scale                    = {rate_scale}")
    print(f"Base random seed              = {seed}")
    print(f"Parallel workers n_jobs       = {n_jobs}")
    print()

    start_time = time.time()

    worker_args = [(m, N, n_tests, rate_scale, seed) for m in range(n_models)]

    if n_jobs == 1:
        results = [run_single_model(*wa) for wa in worker_args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_model, worker_args)

    elapsed = time.time() - start_time

    max_abs_err_all = max(r["max_abs_err"] for r in results)
    max_rel_err_all = max(r["max_rel_err"] for r in results)
    max_stat_resid_all = max(r["stat_resid"] for r in results)
    min_Ec_all = min(r["min_Ec"] for r in results)
    min_Eg_all = min(r["min_Eg"] for r in results)
    max_Ec_all = max(r["max_Ec"] for r in results)
    max_Eg_all = max(r["max_Eg"] for r in results)

    print("Summary over ensemble:")
    print(f"  Max |E_GKLS - E_classical|  = {max_abs_err_all:.3e}")
    print(f"  Max relative error          = {max_rel_err_all:.3e}")
    print(f"  Max stationarity residual   = {max_stat_resid_all:.3e}")
    print(f"  Min classical Dirichlet     = {min_Ec_all:.3e}")
    print(f"  Min GKLS Dirichlet          = {min_Eg_all:.3e}")
    print(f"  Max classical Dirichlet     = {max_Ec_all:.3e}")
    print(f"  Max GKLS Dirichlet          = {max_Eg_all:.3e}")
    print()
    print(f"Elapsed wall clock time       = {elapsed:.2f} s")


if __name__ == "__main__":
    main()
