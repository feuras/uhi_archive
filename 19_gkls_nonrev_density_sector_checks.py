#!/usr/bin/env python3
"""
19_gkls_nonrev_density_sector_checks.py

Non reversible Markov chains and diagonal GKLS lifts:

  - Build random non reversible Markov generators Q with column sums zero.
  - Compute stationary distribution pi from Q pi = 0.
  - Form diagonal GKLS with jump operators L_ij = sqrt(w[j,i]) |i><j|
    where w[i,j] = Q[j,i] are classical rates.
  - Construct BKM (Kubo–Mori) metric M at rho_ss = diag(pi).
  - Form metric adjoint K^sharp and symmetric part G = (K + K^sharp)/2.
  - For random mass conserving perturbations delta p, compare

        E_classical(delta p) =
            1/2 sum_{i != j} pi_i w[i,j] (delta p_j/pi_j - delta p_i/pi_i)^2

        E_GKLS(delta p) =
            - <delta u, M G delta u>,
            delta u = vec(diag(delta p)).

This tests that the density sector Fisher–Dirichlet geometry from GKLS matches
the classical Fisher Dirichlet form even when Q is not reversible.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng
from numpy.linalg import eig


# --------------------------------------------------------------------
# Utilities for Markov generators and GKLS lift
# --------------------------------------------------------------------

def make_random_nonrev_Q(N, rng, rate_scale=1.0):
    """
    Build a random non reversible generator Q with column sums zero.

    Construction:
      - For each pair i != j, draw a positive rate r_ij.
      - Set Q[i, j] = r_ij for i != j.
      - Set Q[j, j] = - sum_{i != j} Q[i, j] so that column sums vanish.

    With generic random rates this is non reversible with respect to its
    stationary distribution.
    """
    R = rate_scale * rng.random((N, N))
    np.fill_diagonal(R, 0.0)
    Q = np.zeros((N, N), dtype=float)
    for j in range(N):
        for i in range(N):
            if i != j:
                Q[i, j] = R[i, j]
        Q[j, j] = -np.sum(Q[:, j]) + Q[j, j]
    return Q


def stationary_distribution(Q, tol=1e-12):
    """
    Compute stationary distribution pi solving Q pi = 0 with pi_i >= 0, sum pi_i = 1.

    We find the eigenvector of Q with eigenvalue closest to 0.
    """
    evals, vecs = eig(Q)
    idx = np.argmin(np.abs(evals))
    v = vecs[:, idx]
    v = v.real
    # Ensure non negativity
    v = np.abs(v)
    if v.sum() == 0.0:
        raise RuntimeError("Failed to obtain non trivial stationary vector")
    pi = v / v.sum()
    resid = np.linalg.norm(Q @ pi)
    if resid > tol:
        raise RuntimeError(f"Stationarity residual too large: {resid}")
    return pi, resid


def build_w_from_Q(Q):
    """
    Given Q with dp/dt = Q p and column sums zero, define classical rates w.

    We use the convention
        Q[i, j] = w[j, i]   for i != j
    so that w[i, j] is the rate i -> j.
    """
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    w = np.zeros_like(Q)
    for i in range(N):
        for j in range(N):
            if i != j:
                w[i, j] = Q[j, i]
    return w


def lindblad_superoperator_diagonal(H, w):
    """
    Diagonal jump GKLS superoperator for Hamiltonian H and rates w[i, j] (i -> j).

    Jump operators:
      L_ij = sqrt(w[j,i]) |i><j|   for i != j.

    For non reversible Q the resulting GKLS is not in detailed balance, but
    diag(pi) with Q pi = 0 is still stationary and diagonal.
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    dim = N * N
    K = np.zeros((dim, dim), dtype=complex)

    # Hamiltonian part (we take H = 0 here by default)
    if np.any(np.abs(H) > 0):
        I = np.eye(N, dtype=complex)
        K_H = -1j * (np.kron(I, H) - np.kron(H.T, I))
        K += K_H

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
            term_jump = np.kron(L.conj(), L)
            term_left = np.kron(I_N, LdL)
            term_right = np.kron(LdL.T, I_N)
            K += term_jump - 0.5 * (term_left + term_right)

    return K


def vec(rho):
    """Vectorise density matrix rho in column stacking convention."""
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    """Unvectorise v back to N x N matrix in column stacking convention."""
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


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
    Classical Fisher–Dirichlet energy for a (possibly non reversible) chain
    with stationary distribution pi and rates w[i, j] (i -> j).

      delta_p: perturbation of probabilities (sum delta_p_i = 0 recommended)

    This is the standard symmetric Dirichlet form built from edge weights
    pi_i w[i,j] and the tilt phi_i = delta_p_i / pi_i.
    """
    pi = np.asarray(pi, dtype=float)
    w = np.asarray(w, dtype=float)
    N = len(pi)
    delta_p = np.asarray(delta_p, dtype=float)
    phi = delta_p / pi

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
    delta_u = vec(delta_rho)
    MG_du = M_diag * (G @ delta_u)
    E = -np.vdot(delta_u, MG_du).real
    return float(E)


# --------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------

def run_single_model(model_index, N, n_tests, rate_scale, seed):
    rng = default_rng(seed + 1000 * model_index)

    # Random non reversible Q and its stationary distribution
    Q = make_random_nonrev_Q(N, rng, rate_scale=rate_scale)
    pi, stat_resid_Q = stationary_distribution(Q)
    col_sums = Q.sum(axis=0)
    col_resid = float(np.linalg.norm(col_sums, ord=np.inf))

    # Classical rates from Q
    w = build_w_from_Q(Q)

    # Diagonal GKLS with H = 0
    H = np.zeros((N, N), dtype=complex)
    K = lindblad_superoperator_diagonal(H, w)

    # Check that rho_ss = diag(pi) is stationary under GKLS
    rho_ss = np.diag(pi)
    stat_vec = K @ vec(rho_ss)
    stat_resid_K = float(np.linalg.norm(stat_vec))

    # Check population generator from GKLS matches Q
    n_gen_tests = 5
    max_gen_error = 0.0
    for _ in range(n_gen_tests):
        p0 = rng.random(N)
        p0 = p0 / p0.sum()
        rho0 = np.diag(p0)
        rhs = K @ vec(rho0)
        rhs_rho = unvec(rhs, N)
        dp_from_gkls = np.diag(rhs_rho).real
        dp_from_Q = Q @ p0
        err = np.linalg.norm(dp_from_gkls - dp_from_Q, ord=np.inf)
        max_gen_error = max(max_gen_error, err)

    # BKM metric and symmetric part G
    C = bkm_weights(pi)
    M_diag = C.reshape(-1, order="F")
    M_inv = 1.0 / M_diag

    Kdag = K.conj().T
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G = 0.5 * (K + Ksharp)

    # Fisher–Dirichlet comparison
    max_abs_err = 0.0
    max_rel_err = 0.0
    min_Ec = float("inf")
    min_Eg = float("inf")
    max_Ec = 0.0
    max_Eg = 0.0

    for _ in range(n_tests):
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
        "stat_resid_Q": stat_resid_Q,
        "col_resid": col_resid,
        "stat_resid_K": stat_resid_K,
        "max_gen_error": max_gen_error,
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
        description="Non reversible GKLS density-sector Fisher–Dirichlet checks."
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

    print("gkls_nonrev_density_sector_checks.py")
    print("------------------------------------")
    print(f"Dimension N                    = {N}")
    print(f"Ensemble size n_models         = {n_models}")
    print(f"Perturbations per model        = {n_tests}")
    print(f"Rate scale                     = {rate_scale}")
    print(f"Base random seed               = {seed}")
    print(f"Parallel workers n_jobs        = {n_jobs}")
    print()

    start_time = time.time()

    worker_args = [(m, N, n_tests, rate_scale, seed) for m in range(n_models)]

    if n_jobs == 1:
        results = [run_single_model(*wa) for wa in worker_args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_model, worker_args)

    elapsed = time.time() - start_time

    max_stat_resid_Q = max(r["stat_resid_Q"] for r in results)
    max_col_resid = max(r["col_resid"] for r in results)
    max_stat_resid_K = max(r["stat_resid_K"] for r in results)
    max_gen_error_all = max(r["max_gen_error"] for r in results)
    max_abs_err_all = max(r["max_abs_err"] for r in results)
    max_rel_err_all = max(r["max_rel_err"] for r in results)
    min_Ec_all = min(r["min_Ec"] for r in results)
    min_Eg_all = min(r["min_Eg"] for r in results)
    max_Ec_all = max(r["max_Ec"] for r in results)
    max_Eg_all = max(r["max_Eg"] for r in results)

    print("Summary over ensemble:")
    print(f"  Max stationarity residual Q   = {max_stat_resid_Q:.3e}")
    print(f"  Max column sum residual Q     = {max_col_resid:.3e}")
    print(f"  Max stationarity residual GKLS= {max_stat_resid_K:.3e}")
    print(f"  Max generator error (p-dot)   = {max_gen_error_all:.3e}")
    print(f"  Max |E_GKLS - E_classical|    = {max_abs_err_all:.3e}")
    print(f"  Max relative error            = {max_rel_err_all:.3e}")
    print(f"  Min classical Dirichlet       = {min_Ec_all:.3e}")
    print(f"  Min GKLS Dirichlet            = {min_Eg_all:.3e}")
    print(f"  Max classical Dirichlet       = {max_Ec_all:.3e}")
    print(f"  Max GKLS Dirichlet            = {max_Eg_all:.3e}")
    print()
    print(f"Elapsed wall clock time         = {elapsed:.2f} s")


if __name__ == "__main__":
    main()
