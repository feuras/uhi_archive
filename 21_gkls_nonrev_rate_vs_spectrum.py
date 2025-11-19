#!/usr/bin/env python3
"""
21_gkls_nonrev_rate_vs_spectrum.py

Non reversible Markov chains and GKLS lifts:
compare effective decay rates of entropy and Fisher energy with

  - lambda_Q   = spectral gap of Q (real part based)
  - lambda_sym = spectral gap of the symmetric part Q_sym in the pi-metric
  - lambda_F   = Fisher Laplacian gap from the symmetric Dirichlet operator L

We reuse the GKLS diagonal lift and BKM metric as in scripts 19 and 20.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng
from numpy.linalg import eig, eigh
from scipy.linalg import expm


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


def fisher_laplacian(pi, w):
    pi = np.asarray(pi, dtype=float)
    w = np.asarray(w, dtype=float)
    N = len(pi)
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

    # Build Q, pi, w
    Q = make_random_nonrev_Q(N, rng, rate_scale=rate_scale)
    pi, stat_resid_Q = stationary_distribution(Q)
    w = build_w_from_Q(Q)

    # GKLS
    H = np.zeros((N, N), dtype=complex)
    K = lindblad_superoperator_diagonal(H, w)
    rho_ss = np.diag(pi)
    stat_resid_K = float(np.linalg.norm(K @ vec(rho_ss)))

    # BKM and G
    C = bkm_weights(pi)
    M_diag = C.reshape(-1, order="F")
    M_inv = 1.0 / M_diag
    Kdag = K.conj().T
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G = 0.5 * (K + Ksharp)

    # Fisher Laplacian and gap
    L = fisher_laplacian(pi, w)
    evals_L, _ = eigh(L)
    evals_sorted = np.sort(evals_L)
    lambda_F = evals_sorted[1] if len(evals_sorted) > 1 else 0.0

    # Spectral gap of Q
    evals_Q = eig(Q)[0]
    # Exclude eigenvalue near 0
    idx_nonzero = np.where(np.abs(evals_Q) > 1e-8)[0]
    if len(idx_nonzero) > 0:
        lambda_Q = -np.max(evals_Q[idx_nonzero].real)
    else:
        lambda_Q = 0.0

    # Symmetric part of Q in pi metric: Q_sym = 0.5 (Q + Pi^{-1} Q^T Pi)
    Pi = np.diag(pi)
    Q_sharp = np.linalg.inv(Pi) @ Q.T @ Pi
    Q_sym = 0.5 * (Q + Q_sharp)
    evals_Qsym = eig(Q_sym)[0]
    idx_nonzero_sym = np.where(np.abs(evals_Qsym) > 1e-8)[0]
    if len(idx_nonzero_sym) > 0:
        lambda_sym = -np.max(evals_Qsym[idx_nonzero_sym].real)
    else:
        lambda_sym = 0.0

    # Time stepping
    times = np.linspace(0.0, t_final, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0
    expQ_dt = expm(Q * dt) if dt > 0.0 else np.eye(N)
    expK_dt = expm(K * dt) if dt > 0.0 else np.eye(N * N, dtype=complex)

    rates_Sc = []
    rates_Sq = []
    rates_Ec = []
    rates_Eq = []

    for _ in range(n_traj):
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

    def stat_pair(v):
        if not v:
            return (np.nan, np.nan)
        arr = np.array(v)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    mean_Sc, std_Sc = stat_pair(rates_Sc)
    mean_Sq, std_Sq = stat_pair(rates_Sq)
    mean_Ec, std_Ec = stat_pair(rates_Ec)
    mean_Eq, std_Eq = stat_pair(rates_Eq)

    return {
        "index": model_index,
        "stat_resid_Q": float(stat_resid_Q),
        "stat_resid_K": float(stat_resid_K),
        "lambda_Q": float(lambda_Q),
        "lambda_sym": float(lambda_sym),
        "lambda_F": float(lambda_F),
        "mean_Sc": mean_Sc,
        "std_Sc": std_Sc,
        "mean_Sq": mean_Sq,
        "std_Sq": std_Sq,
        "mean_Ec": mean_Ec,
        "std_Ec": std_Ec,
        "mean_Eq": mean_Eq,
        "std_Eq": std_Eq,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Non reversible GKLS: rates vs spectra."
    )
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--n_models", type=int, default=10)
    parser.add_argument("--n_traj", type=int, default=10)
    parser.add_argument("--t_final", type=float, default=5.0)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--rate_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n_jobs", type=int, default=0)
    args = parser.parse_args()

    if args.n_jobs > 0:
        n_jobs = args.n_jobs
    else:
        cpu_count = mp.cpu_count()
        n_jobs = min(20, cpu_count)

    print("gkls_nonrev_rate_vs_spectrum.py")
    print("--------------------------------")
    print(f"Dimension N                    = {args.N}")
    print(f"Ensemble size n_models         = {args.n_models}")
    print(f"Trajectories per model         = {args.n_traj}")
    print(f"Final time t_final             = {args.t_final}")
    print(f"Time steps per trajectory      = {args.n_steps}")
    print(f"Rate scale                     = {args.rate_scale}")
    print(f"Base random seed               = {args.seed}")
    print(f"Parallel workers n_jobs        = {n_jobs}")
    print()

    start_time = time.time()
    worker_args = [
        (m, args.N, args.n_traj, args.t_final, args.n_steps, args.rate_scale, args.seed)
        for m in range(args.n_models)
    ]

    if n_jobs == 1:
        results = [run_single_model(*wa) for wa in worker_args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_model, worker_args)

    elapsed = time.time() - start_time

    lambda_Qs = np.array([r["lambda_Q"] for r in results])
    lambda_syms = np.array([r["lambda_sym"] for r in results])
    lambda_Fs = np.array([r["lambda_F"] for r in results])

    print("Spectral gaps over models:")
    print(f"  lambda_Q   mean              = {np.nanmean(lambda_Qs):.3e}")
    print(f"  lambda_Q   min, max          = {np.nanmin(lambda_Qs):.3e}, {np.nanmax(lambda_Qs):.3e}")
    print(f"  lambda_sym mean              = {np.nanmean(lambda_syms):.3e}")
    print(f"  lambda_sym min, max          = {np.nanmin(lambda_syms):.3e}, {np.nanmax(lambda_syms):.3e}")
    print(f"  lambda_F   mean              = {np.nanmean(lambda_Fs):.3e}")
    print(f"  lambda_F   min, max          = {np.nanmin(lambda_Fs):.3e}, {np.nanmax(lambda_Fs):.3e}")
    print()

    def summarise(label, key_mean, key_std):
        vals = np.array([r[key_mean] for r in results])
        stds = np.array([r[key_std] for r in results])
        print(f"  {label} mean rate            = {np.nanmean(vals):.3e}")
        print(f"  {label} per model std (avg)  = {np.nanmean(stds):.3e}")
        print(f"  {label} rate / lambda_Q mean = {np.nanmean(vals / lambda_Qs):.3e}")
        print(f"  {label} rate / lambda_sym mean = {np.nanmean(vals / lambda_syms):.3e}")
        print(f"  {label} rate / lambda_F mean = {np.nanmean(vals / lambda_Fs):.3e}")
        print()

    summarise("S_c", "mean_Sc", "std_Sc")
    summarise("S_q", "mean_Sq", "std_Sq")
    summarise("E_c", "mean_Ec", "std_Ec")
    summarise("E_q", "mean_Eq", "std_Eq")

    print(f"Elapsed wall clock time         = {elapsed:.2f} s")


if __name__ == "__main__":
    main()
