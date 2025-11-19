#!/usr/bin/env python3
"""
gkls_diagonal_to_markov_checks.py

Diagonal GKLS models with detailed balance, mapped to reversible Markov chains.

Checks:
  - Population generator extracted from GKLS matches the constructed Markov generator Q.
  - Time evolution of populations under GKLS matches exp(t Q) for random initial states.
  - Off diagonal entries of rho(t) remain negligible for diagonal jump models.

The script can run an ensemble of random models in parallel.
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
from numpy.random import default_rng
from scipy.linalg import expm, eigvals


def make_random_pi(N, rng):
    """Random stationary distribution with N components."""
    x = rng.random(N)
    pi = x / x.sum()
    # Guard against extremely small entries
    eps = 1e-3 / N
    pi = np.clip(pi, eps, None)
    pi = pi / pi.sum()
    return pi


def make_random_flux(N, rng, rate_scale=1.0):
    """Random symmetric flux matrix F_ij >= 0 with zero diagonal."""
    F = rng.random((N, N))
    F = 0.5 * (F + F.T)
    np.fill_diagonal(F, 0.0)
    F *= rate_scale
    return F


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
    assert F.shape == (N, N)
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
    Build the Lindblad superoperator L for diagonal jump model with Hamiltonian H and rates w.

    H is an N x N Hermitian matrix.
    w[i, j] is the classical rate i -> j.
    The jump operators are L_ij = sqrt(w[j, i]) |i><j| for i != j.
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    # Superoperator acts on vec(rho) in column stacking convention
    dim = N * N
    L_super = np.zeros((dim, dim), dtype=complex)

    # Hamiltonian part: -i [H, rho]
    if np.any(np.abs(H) > 0):
        I = np.eye(N, dtype=complex)
        L_H = -1j * (np.kron(I, H) - np.kron(H.T, I))
        L_super += L_H

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
            L_super += term_jump - 0.5 * (term_left + term_right)

    return L_super


def vec(rho):
    """Vectorise density matrix rho in column stacking convention."""
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    """Unvectorise v back to N x N matrix in column stacking convention."""
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def random_density_matrix(N, rng):
    """Random full rank density matrix of size N x N."""
    X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    rho = X @ X.conj().T
    rho = rho / np.trace(rho)
    return rho


def quantum_relative_entropy(rho, rho_ss, eps=1e-14):
    """
    Quantum relative entropy S(rho || rho_ss) = Tr[rho (log rho - log rho_ss)].

    Implemented via eigen decomposition. Assumes rho_ss is full rank.
    """
    rho = np.asarray(rho, dtype=complex)
    rho_ss = np.asarray(rho_ss, dtype=complex)

    # Diagonalise rho
    vals_rho, vecs_rho = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    vals_rho = np.clip(vals_rho.real, eps, None)
    log_rho = vecs_rho @ np.diag(np.log(vals_rho)) @ vecs_rho.conj().T

    # Diagonalise rho_ss
    vals_ss, vecs_ss = np.linalg.eigh(0.5 * (rho_ss + rho_ss.conj().T))
    vals_ss = np.clip(vals_ss.real, eps, None)
    log_rho_ss = vecs_ss @ np.diag(np.log(vals_ss)) @ vecs_ss.conj().T

    diff = log_rho - log_rho_ss
    return float(np.trace(rho @ diff).real)


def classical_kl(p, pi, eps=1e-14):
    """Classical KL divergence sum_i p_i (log p_i - log pi_i)."""
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    p = np.clip(p, eps, None)
    pi = np.clip(pi, eps, None)
    return float(np.sum(p * (np.log(p) - np.log(pi))))


def run_single_model(model_index, N, n_traj, t_final, n_steps, rate_scale, seed):
    """
    Run a single random model:

      - Sample pi, F, w, Q.
      - Build diagonal GKLS superoperator.
      - Check generator mapping and time evolution on random diagonal trajectories.
    """
    rng = default_rng(seed + 1000 * model_index)
    pi = make_random_pi(N, rng)
    F = make_random_flux(N, rng, rate_scale=rate_scale)
    w, Q = build_w_and_Q(pi, F)

    # Sanity checks on Q
    col_sums = Q.sum(axis=0)
    stat_resid = np.linalg.norm(Q @ pi)
    # Spectrum of Q, exclude trivial zero
    evals_Q = eigvals(Q)
    evals_Q_sorted = np.sort(evals_Q.real)
    if evals_Q_sorted.size > 1:
        gap_Q = -evals_Q_sorted[-2]
    else:
        gap_Q = 0.0

    # Build GKLS superoperator with H = 0 (purely diagonal jumps)
    N_hilb = N
    H = np.zeros((N_hilb, N_hilb), dtype=complex)
    L_super = lindblad_superoperator(H, w)

    # Check generator mapping on random diagonal densities
    n_gen_tests = 5
    max_gen_error = 0.0
    for _ in range(n_gen_tests):
        p0 = make_random_pi(N, rng)
        rho0 = np.diag(p0)
        rhs = L_super @ vec(rho0)
        rhs_rho = unvec(rhs, N_hilb)
        dp_from_gkls = np.diag(rhs_rho).real
        dp_from_Q = Q @ p0
        err = np.linalg.norm(dp_from_gkls - dp_from_Q, ord=np.inf)
        max_gen_error = max(max_gen_error, err)

    # Time evolution checks for diagonal initial states
    times = np.linspace(0.0, t_final, n_steps)
    max_traj_error = 0.0
    max_offdiag_norm = 0.0
    max_rel_entropy_mismatch = 0.0

    rho_ss = np.diag(pi)

    dt = times[1] - times[0] if len(times) > 1 else 0.0
    expL_dt = expm(L_super * dt) if dt > 0.0 else np.eye(N_hilb * N_hilb, dtype=complex)
    expQ_dt = expm(Q * dt) if dt > 0.0 else np.eye(N, dtype=float)

    for _ in range(n_traj):
        # Diagonal initial state
        p0 = make_random_pi(N, rng)
        rho0 = np.diag(p0)

        v_rho = vec(rho0)
        p = p0.copy()

        # Check at t = 0
        diag_rho0 = np.diag(rho0).real
        err0 = np.linalg.norm(diag_rho0 - p, ord=np.inf)
        max_traj_error = max(max_traj_error, err0)
        off_diag0 = rho0.copy()
        np.fill_diagonal(off_diag0, 0.0)
        max_offdiag_norm = max(max_offdiag_norm, np.linalg.norm(off_diag0))
        S_q0 = quantum_relative_entropy(rho0, rho_ss)
        S_c0 = classical_kl(p, pi)
        max_rel_entropy_mismatch = max(max_rel_entropy_mismatch, abs(S_q0 - S_c0))

        # Evolve in time
        for k in range(1, n_steps):
            v_rho = expL_dt @ v_rho
            rho_t = unvec(v_rho, N_hilb)
            p = expQ_dt @ p

            diag_rho = np.diag(rho_t).real
            err = np.linalg.norm(diag_rho - p, ord=np.inf)
            max_traj_error = max(max_traj_error, err)

            off_diag = rho_t.copy()
            np.fill_diagonal(off_diag, 0.0)
            off_norm = np.linalg.norm(off_diag)
            max_offdiag_norm = max(max_offdiag_norm, off_norm)

            S_q = quantum_relative_entropy(rho_t, rho_ss)
            S_c = classical_kl(diag_rho, pi)
            ent_mis = abs(S_q - S_c)
            max_rel_entropy_mismatch = max(max_rel_entropy_mismatch, ent_mis)

    return {
        "index": model_index,
        "max_gen_error": max_gen_error,
        "max_traj_error": max_traj_error,
        "max_offdiag_norm": max_offdiag_norm,
        "max_rel_entropy_mismatch": max_rel_entropy_mismatch,
        "stat_resid": float(stat_resid),
        "col_sums_norm": float(np.linalg.norm(col_sums, ord=np.inf)),
        "gap_Q": float(gap_Q),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagonal GKLS to Markov checks: density sector equivalence tests."
    )
    parser.add_argument("--N", type=int, default=3, help="Hilbert space dimension")
    parser.add_argument("--n_models", type=int, default=20, help="Number of random models in the ensemble")
    parser.add_argument("--n_traj", type=int, default=10, help="Number of random trajectories per model")
    parser.add_argument("--t_final", type=float, default=5.0, help="Final time for trajectory tests")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of time steps per trajectory")
    parser.add_argument("--rate_scale", type=float, default=1.0, help="Overall scale for random rates")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed")
    parser.add_argument("--n_jobs", type=int, default=0, help="Number of parallel workers (0 for auto)")
    args = parser.parse_args()

    N = args.N
    n_models = args.n_models
    n_traj = args.n_traj
    t_final = args.t_final
    n_steps = args.n_steps
    rate_scale = args.rate_scale
    seed = args.seed

    if args.n_jobs > 0:
        n_jobs = args.n_jobs
    else:
        cpu_count = mp.cpu_count()
        n_jobs = min(20, cpu_count)

    print("gkls_diagonal_to_markov_checks.py")
    print("---------------------------------")
    print(f"Hilbert dimension N          = {N}")
    print(f"Ensemble size n_models       = {n_models}")
    print(f"Trajectories per model       = {n_traj}")
    print(f"Final time t_final           = {t_final}")
    print(f"Time steps per trajectory    = {n_steps}")
    print(f"Rate scale                   = {rate_scale}")
    print(f"Base random seed             = {seed}")
    print(f"Parallel workers n_jobs      = {n_jobs}")
    print()

    start_time = time.time()

    worker_args = [(m, N, n_traj, t_final, n_steps, rate_scale, seed) for m in range(n_models)]

    if n_jobs == 1:
        results = [run_single_model(*wa) for wa in worker_args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_model, worker_args)

    elapsed = time.time() - start_time

    # Aggregate diagnostics
    max_gen_error_all = max(r["max_gen_error"] for r in results)
    max_traj_error_all = max(r["max_traj_error"] for r in results)
    max_offdiag_norm_all = max(r["max_offdiag_norm"] for r in results)
    max_ent_mismatch_all = max(r["max_rel_entropy_mismatch"] for r in results)
    max_stat_resid_all = max(r["stat_resid"] for r in results)
    max_col_sum_norm_all = max(r["col_sums_norm"] for r in results)
    gaps = [r["gap_Q"] for r in results]

    print("Summary over ensemble:")
    print(f"  Max generator error        = {max_gen_error_all:.3e}")
    print(f"  Max trajectory error       = {max_traj_error_all:.3e}")
    print(f"  Max off diagonal norm      = {max_offdiag_norm_all:.3e}")
    print(f"  Max rel entropy mismatch   = {max_ent_mismatch_all:.3e}")
    print(f"  Max stationarity residual  = {max_stat_resid_all:.3e}")
    print(f"  Max column sums norm       = {max_col_sum_norm_all:.3e}")
    if gaps:
        print(f"  Min Markov spectral gap    = {min(gaps):.3e}")
        print(f"  Max Markov spectral gap    = {max(gaps):.3e}")
    print()
    print(f"Elapsed wall clock time      = {elapsed:.2f} s")


if __name__ == "__main__":
    main()
