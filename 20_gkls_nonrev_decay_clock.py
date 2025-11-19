#!/usr/bin/env python3
"""
20_gkls_nonrev_decay_clock.py

Non reversible Markov chains and GKLS lifts: Fisher decay clock tests.

For each random model:
  - Build non reversible Q with column sums zero.
  - Compute stationary pi from Q pi = 0 and rates w[i,j] = Q[j,i].
  - Build diagonal GKLS with jumps L_ij = sqrt(w[j,i]) |i><j|.
  - Construct symmetric Fisher Laplacian L_sym on tilt phi = delta_p / pi
    from edge weights a_ij = 0.5 (pi_i w[i,j] + pi_j w[j,i]).
  - Compute its spectral gap lambda_F (smallest non zero eigenvalue).
  - For random diagonal initial states evolve:
        dp/dt   = Q p
        d(rho)/dt = GKLS(rho)
    and track
        S_c(t) = KL(p(t) || pi)
        S_q(t) = S(rho(t) || rho_ss)
        E_c(t) = classical Dirichlet
        E_q(t) = GKLS Dirichlet
  - Fit an effective decay rate from log S_c(t) and log S_q(t) on a late
    time window and compare to lambda_F.

This probes whether the Fisher symmetric gap still controls entropy decay
even when detailed balance is broken.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng
from numpy.linalg import eig, eigh
from scipy.linalg import expm


# Reuse core utilities from script 19 (copied for self containment)

def make_random_nonrev_Q(N, rng, rate_scale=1.0):
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
    evals, vecs = eig(Q)
    idx = np.argmin(np.abs(evals))
    v = vecs[:, idx]
    v = v.real
    v = np.abs(v)
    if v.sum() == 0.0:
        raise RuntimeError("Failed to obtain non trivial stationary vector")
    pi = v / v.sum()
    resid = np.linalg.norm(Q @ pi)
    if resid > tol:
        raise RuntimeError(f"Stationarity residual too large: {resid}")
    return pi, resid


def build_w_from_Q(Q):
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    w = np.zeros_like(Q)
    for i in range(N):
        for j in range(N):
            if i != j:
                w[i, j] = Q[j, i]
    return w


def lindblad_superoperator_diagonal(H, w):
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    dim = N * N
    K = np.zeros((dim, dim), dtype=complex)
    if np.any(np.abs(H) > 0):
        I = np.eye(N, dtype=complex)
        K_H = -1j * (np.kron(I, H) - np.kron(H.T, I))
        K += K_H
    I_N = np.eye(N, dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rate = w[j, i]
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
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def bkm_weights(pi):
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
    N = len(delta_p)
    delta_rho = np.diag(delta_p)
    delta_u = vec(delta_rho)
    MG_du = M_diag * (G @ delta_u)
    E = -np.vdot(delta_u, MG_du).real
    return float(E)


def quantum_relative_entropy(rho, rho_ss, eps=1e-14):
    rho = np.asarray(rho, dtype=complex)
    rho_ss = np.asarray(rho_ss, dtype=complex)
    vals_rho, vecs_rho = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    vals_rho = np.clip(vals_rho.real, eps, None)
    log_rho = vecs_rho @ np.diag(np.log(vals_rho)) @ vecs_rho.conj().T
    vals_ss, vecs_ss = np.linalg.eigh(0.5 * (rho_ss + rho_ss.conj().T))
    vals_ss = np.clip(vals_ss.real, eps, None)
    log_rho_ss = vecs_ss @ np.diag(np.log(vals_ss)) @ vecs_ss.conj().T
    diff = log_rho - log_rho_ss
    return float(np.trace(rho @ diff).real)


def classical_kl(p, pi, eps=1e-14):
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    p = np.clip(p, eps, None)
    pi = np.clip(pi, eps, None)
    return float(np.sum(p * (np.log(p) - np.log(pi))))


def fisher_laplacian(phi_dim, pi, w):
    """
    Build the symmetric Fisher Laplacian L_sym on tilt variables phi.

    Edge weights:
        a_ij = 0.5 * (pi_i w[i,j] + pi_j w[j,i])

    Dirichlet:
        E(phi) = 0.5 sum_{i,j} a_ij (phi_j - phi_i)^2 = phi^T L_sym phi
    """
    pi = np.asarray(pi, dtype=float)
    w = np.asarray(w, dtype=float)
    N = len(pi)
    assert N == phi_dim
    a = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a[i, j] = 0.5 * (pi[i] * w[i, j] + pi[j] * w[j, i])
    L = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            L[i, i] += a[i, j]
            L[i, j] -= a[i, j]
    return L


def effective_decay_rate(times, values, t_min_frac=0.3, t_max_frac=0.9):
    """
    Fit a decay rate r from values(t) ~ C exp(-r t) on a late time window.

    We fit log(values) vs t using simple least squares, restricting to
    times where values are positive and above numerical floor.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask_pos = values > 1e-14
    times = times[mask_pos]
    values = values[mask_pos]
    if len(times) < 3:
        return np.nan
    t_min = times[0] + t_min_frac * (times[-1] - times[0])
    t_max = times[0] + t_max_frac * (times[-1] - times[0])
    mask_window = (times >= t_min) & (times <= t_max)
    t_fit = times[mask_window]
    v_fit = values[mask_window]
    if len(t_fit) < 3:
        return np.nan
    y = np.log(v_fit)
    A = np.vstack([t_fit, np.ones_like(t_fit)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = coeffs[0]
    r = -slope
    return float(r)


def run_single_model(model_index, N, n_traj, t_final, n_steps, rate_scale, seed):
    rng = default_rng(seed + 1000 * model_index)

    # Build model
    Q = make_random_nonrev_Q(N, rng, rate_scale=rate_scale)
    pi, stat_resid_Q = stationary_distribution(Q)
    w = build_w_from_Q(Q)

    # GKLS
    H = np.zeros((N, N), dtype=complex)
    K = lindblad_superoperator_diagonal(H, w)
    rho_ss = np.diag(pi)
    stat_vec = K @ vec(rho_ss)
    stat_resid_K = float(np.linalg.norm(stat_vec))

    # BKM metric and symmetric part G
    C = bkm_weights(pi)
    M_diag = C.reshape(-1, order="F")
    M_inv = 1.0 / M_diag
    Kdag = K.conj().T
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G = 0.5 * (K + Ksharp)

    # Fisher Laplacian and gap
    L = fisher_laplacian(N, pi, w)
    evals_L, _ = eigh(L)
    evals_sorted = np.sort(evals_L)
    if len(evals_sorted) > 1:
        lambda_F = evals_sorted[1]
    else:
        lambda_F = 0.0

    # Time stepping
    times = np.linspace(0.0, t_final, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0
    expQ_dt = expm(Q * dt) if dt > 0.0 else np.eye(N)
    expK_dt = expm(K * dt) if dt > 0.0 else np.eye(N * N, dtype=complex)

    # Collect decay rates across trajectories
    rates_Sc = []
    rates_Sq = []
    rates_Ec = []
    rates_Eq = []

    for _ in range(n_traj):
        # Random diagonal initial state
        p0 = rng.random(N)
        p0 = p0 / p0.sum()
        rho0 = np.diag(p0)

        p = p0.copy()
        v_rho = vec(rho0)

        S_c_vals = []
        S_q_vals = []
        E_c_vals = []
        E_q_vals = []

        for k in range(n_steps):
            if k > 0:
                p = expQ_dt @ p
                v_rho = expK_dt @ v_rho
            rho_t = unvec(v_rho, N)

            delta_p = p - pi
            S_c = classical_kl(p, pi)
            S_q = quantum_relative_entropy(rho_t, rho_ss)
            E_c = classical_dirichlet(delta_p, pi, w)
            E_q = gkls_dirichlet(delta_p, G, M_diag)

            S_c_vals.append(S_c)
            S_q_vals.append(S_q)
            E_c_vals.append(E_c)
            E_q_vals.append(E_q)

        r_Sc = effective_decay_rate(times, np.array(S_c_vals))
        r_Sq = effective_decay_rate(times, np.array(S_q_vals))
        r_Ec = effective_decay_rate(times, np.array(E_c_vals))
        r_Eq = effective_decay_rate(times, np.array(E_q_vals))

        if np.isfinite(r_Sc):
            rates_Sc.append(r_Sc)
        if np.isfinite(r_Sq):
            rates_Sq.append(r_Sq)
        if np.isfinite(r_Ec):
            rates_Ec.append(r_Ec)
        if np.isfinite(r_Eq):
            rates_Eq.append(r_Eq)

    # Aggregate per model
    def stats(v):
        if not v:
            return (np.nan, np.nan)
        arr = np.array(v)
        return float(np.nanmean(arr)), float(np.nanmax(np.abs(arr - lambda_F)))

    mean_Sc, dev_Sc = stats(rates_Sc)
    mean_Sq, dev_Sq = stats(rates_Sq)
    mean_Ec, dev_Ec = stats(rates_Ec)
    mean_Eq, dev_Eq = stats(rates_Eq)

    return {
        "index": model_index,
        "stat_resid_Q": float(stat_resid_Q),
        "stat_resid_K": stat_resid_K,
        "lambda_F": float(lambda_F),
        "mean_rate_Sc": mean_Sc,
        "dev_rate_Sc": dev_Sc,
        "mean_rate_Sq": mean_Sq,
        "dev_rate_Sq": dev_Sq,
        "mean_rate_Ec": mean_Ec,
        "dev_rate_Ec": dev_Ec,
        "mean_rate_Eq": mean_Eq,
        "dev_rate_Eq": dev_Eq,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Non reversible GKLS Fisher decay clock checks."
    )
    parser.add_argument("--N", type=int, default=3, help="Hilbert/Markov dimension")
    parser.add_argument("--n_models", type=int, default=10, help="Number of random models in the ensemble")
    parser.add_argument("--n_traj", type=int, default=10, help="Number of trajectories per model")
    parser.add_argument("--t_final", type=float, default=5.0, help="Final time for trajectories")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of time steps per trajectory")
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

    print("gkls_nonrev_decay_clock.py")
    print("--------------------------")
    print(f"Dimension N                    = {N}")
    print(f"Ensemble size n_models         = {n_models}")
    print(f"Trajectories per model         = {n_traj}")
    print(f"Final time t_final             = {t_final}")
    print(f"Time steps per trajectory      = {n_steps}")
    print(f"Rate scale                     = {rate_scale}")
    print(f"Base random seed               = {seed}")
    print(f"Parallel workers n_jobs        = {n_jobs}")
    print()

    start_time = time.time()

    worker_args = [(m, N, n_traj, t_final, n_steps, rate_scale, seed) for m in range(n_models)]

    if n_jobs == 1:
        results = [run_single_model(*wa) for wa in worker_args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_model, worker_args)

    elapsed = time.time() - start_time

    # Summaries
    max_stat_resid_Q = max(r["stat_resid_Q"] for r in results)
    max_stat_resid_K = max(r["stat_resid_K"] for r in results)
    lambdas = [r["lambda_F"] for r in results]

    print("Summary over ensemble:")
    print(f"  Max stationarity residual Q   = {max_stat_resid_Q:.3e}")
    print(f"  Max stationarity residual GKLS= {max_stat_resid_K:.3e}")
    print(f"  Min Fisher gap lambda_F       = {min(lambdas):.3e}")
    print(f"  Max Fisher gap lambda_F       = {max(lambdas):.3e}")
    print()

    def print_rate(label_mean, label_dev, key_mean, key_dev):
        means = [r[key_mean] for r in results]
        devs = [r[key_dev] for r in results]
        means = np.array(means, dtype=float)
        devs = np.array(devs, dtype=float)
        print(f"  {label_mean} mean over models = {np.nanmean(means):.3e}")
        print(f"  {label_dev} max |rate - lambda_F| = {np.nanmax(devs):.3e}")

    print_rate("S_c rate", "S_c rate", "mean_rate_Sc", "dev_rate_Sc")
    print_rate("S_q rate", "S_q rate", "mean_rate_Sq", "dev_rate_Sq")
    print_rate("E_c rate", "E_c rate", "mean_rate_Ec", "dev_rate_Ec")
    print_rate("E_q rate", "E_q rate", "mean_rate_Eq", "dev_rate_Eq")

    print()
    print(f"Elapsed wall clock time         = {elapsed:.2f} s")


if __name__ == "__main__":
    main()
