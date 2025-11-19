#!/usr/bin/env python3
"""
15_qutrit_markov_vs_FP_universal_gap_checks.py

Qutrit Markov vs continuum-like FP Markov: universal gap driven Fisher decay.

This script closes a loop between:
  - the classical 3-state reversible Markov generator Q_markov that arises as the
    population block of a thermal qutrit GKLS model (see scripts 11–14), and
  - a high resolution 1D reversible Markov chain on a ring, interpreted as a
    discrete Fokker–Planck generator.

Steps:

  1. Reconstruct the 3-state reversible generator Q_markov and its stationary
     Gibbs distribution pi_q from thermal data (beta, energies, downward jump
     rates). Check detailed balance and compute:

       - Markov spectral gap lambda_Q_q (pi-weighted gap of Q_markov),
       - Fisher Dirichlet operator G_q = Q_markov diag(pi_q),
       - Fisher curvature gap lambda_G_q (smallest positive eigenvalue of
         -G_q).

  2. Build a high resolution reversible nearest neighbour Markov chain Q_fp on
     a periodic ring, with uniform stationary distribution pi_fp. This is a
     discrete Laplacian-type Fokker–Planck generator. Compute its spectral gap
     lambda_Q_fp.

  3. Rescale Q_fp by a scalar factor so that the rescaled chain Q_fp_scaled has
     spectral gap matching the qutrit value:

         lambda_Q_fp_scaled = lambda_Q_q.

     Recompute diagnostics for Q_fp_scaled and its Fisher operator
     G_fp = Q_fp_scaled diag(pi_fp).

  4. For both Q_markov and Q_fp_scaled, perform time evolution experiments for
     N_INIT random initial densities. For each chain:

       - Evolve p(t) exactly in the reversible representation using the
         symmetric generator S = B^{-1} Q B with B = diag(sqrt(pi)).
       - Compute the Fisher Dirichlet quadratic

             F(t) = - δp(t)^T G δp(t)  ≥ 0

         with δp(t) = p(t) - pi, using G = Q diag(pi).
       - Fit an exponential envelope F(t) ≈ const * exp(-r t) on the late time
         window and extract empirical decay rates r_est.

  5. Compare the empirical decay rates with the Markov gap prediction
     r_pred = 2 lambda_Q_q for both chains.

Multithreading:
  The time evolution over initial conditions is parallelised over up to 20
  worker threads (or fewer if the machine exposes less cores).

Expected behaviour:
  Both the finite qutrit Markov chain and the high resolution FP-like chain,
  once gap matched, should exhibit Fisher Dirichlet decay rates clustered
  around 2 lambda_Q_q, despite their very different microscopic structure.
"""

import os
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_qutrit_markov(beta: float = 1.0):
    """
    Reconstruct the 3-state reversible Markov generator Q_markov and stationary
    distribution pi_q corresponding to the thermal qutrit GKLS population
    dynamics used in scripts 11–14.

    Energies are E0 = 0, E1 = 1, E2 = 2. Downward jump rates (from higher to
    lower energy) are:

        g01_down: 1 -> 0
        g12_down: 2 -> 1
        g02_down: 2 -> 0

    Upward jump rates follow detailed balance at inverse temperature beta.

    The generator Q_markov is in the convention that columns j sum to zero and
    Q_{ij} (i != j) is the rate from state j to state i.
    """
    E = np.array([0.0, 1.0, 2.0], dtype=float)
    pi_q = np.exp(-beta * E)
    pi_q /= pi_q.sum()

    # Downward jump rates
    g01_down = 0.8  # 1 -> 0
    g12_down = 0.5  # 2 -> 1
    g02_down = 0.3  # 2 -> 0

    # Upward rates from detailed balance: pi_i q_{j,i} = pi_j q_{i,j}
    g10_up = g01_down * math.exp(-beta * (E[1] - E[0]))  # 0 -> 1
    g21_up = g12_down * math.exp(-beta * (E[2] - E[1]))  # 1 -> 2
    g20_up = g02_down * math.exp(-beta * (E[2] - E[0]))  # 0 -> 2

    Q = np.zeros((3, 3), dtype=float)

    # Off-diagonal entries: rates from j to i
    # Pair 0 <-> 1
    Q[0, 1] = g01_down  # 1 -> 0
    Q[1, 0] = g10_up    # 0 -> 1

    # Pair 1 <-> 2
    Q[1, 2] = g12_down  # 2 -> 1
    Q[2, 1] = g21_up    # 1 -> 2

    # Pair 0 <-> 2
    Q[0, 2] = g02_down  # 2 -> 0
    Q[2, 0] = g20_up    # 0 -> 2

    # Diagonals: columns sum to zero
    for j in range(3):
        col_sum = np.sum(Q[:, j])
        Q[j, j] = -col_sum

    return Q, pi_q


def markov_symmetrised_generator(Q: np.ndarray, pi: np.ndarray):
    """
    Build the symmetrised generator S in the pi-weighted L^2 space.

      B = diag(sqrt(pi)),  B_inv = diag(1/sqrt(pi))
      L_sym = B^{-1} Q B

    For a reversible chain L_sym is symmetric; we symmetrise explicitly to
    suppress numerical asymmetries.
    """
    sqrt_pi = np.sqrt(pi)
    B = np.diag(sqrt_pi)
    B_inv = np.diag(1.0 / sqrt_pi)
    L_sym = B_inv @ Q @ B
    S = 0.5 * (L_sym + L_sym.T)
    return S, B, B_inv


def markov_gap(Q: np.ndarray, pi: np.ndarray, tol_zero: float = 1e-10):
    """
    Compute the pi-weighted spectral gap of a reversible Markov generator Q.

    Returns:
      lambda_gap: smallest positive value of -Re(lambda) over non-zero modes.
      eigvals: eigenvalues of the symmetrised generator S.
      eigvecs: corresponding eigenvectors (columns).
      B, B_inv: sqrt(pi) diagonal matrices used in the symmetrisation.
    """
    S, B, B_inv = markov_symmetrised_generator(Q, pi)
    eigvals, eigvecs = np.linalg.eigh(S)
    neg = eigvals[eigvals < -tol_zero]
    if neg.size == 0:
        lambda_gap = 0.0
    else:
        lambda_gap = -neg.max()
    return lambda_gap, eigvals, eigvecs, B, B_inv


def fisher_dirichlet_operator(Q: np.ndarray, pi: np.ndarray):
    """
    Fisher Dirichlet operator G = Q diag(pi), whose quadratic form is negative:
      δp^T G δp <= 0.
    The positive Dirichlet energy is -δp^T G δp.
    """
    return Q @ np.diag(pi)


def fisher_curvature_gap(G: np.ndarray, tol_zero: float = 1e-8):
    """
    Compute the smallest positive eigenvalue of -G (the Fisher curvature gap)
    for a symmetric G representation.
    """
    G_metric = -G
    G_metric = 0.5 * (G_metric + G_metric.T)
    eigvals = np.linalg.eigvalsh(G_metric)
    pos = eigvals[eigvals > tol_zero]
    if pos.size == 0:
        return 0.0, eigvals
    return float(pos.min()), eigvals


def build_fp_markov_chain(Nx: int = 256, L: float = 2 * math.pi):
    """
    Build a high resolution nearest neighbour reversible Markov chain on a ring
    that mimics an overdamped Fokker–Planck operator with uniform stationary
    density.

    States are arranged on a periodic lattice of length L with spacing dx. The
    generator is:

      Q_fp[i+1, i] = Q_fp[i-1, i] = base_rate
      Q_fp[i, i]   = -2 * base_rate

    with base_rate = D / dx^2 and D = 1. This has uniform stationary measure
    pi_fp = (1/Nx, ..., 1/Nx).
    """
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = L / Nx
    D_base = 1.0
    base_rate = D_base / dx**2

    Q_fp = np.zeros((Nx, Nx), dtype=float)
    for j in range(Nx):
        ip = (j + 1) % Nx
        im = (j - 1) % Nx
        Q_fp[ip, j] += base_rate
        Q_fp[im, j] += base_rate
        Q_fp[j, j] -= 2.0 * base_rate

    pi_fp = np.full(Nx, 1.0 / Nx, dtype=float)
    return x, dx, Q_fp, pi_fp


def simulate_F_decay_for_chain(
    Q: np.ndarray,
    pi: np.ndarray,
    G: np.ndarray,
    T_max: float,
    N_T: int,
    N_init: int,
    t_min_frac: float,
    rng_seed: int,
    n_workers: int,
):
    """
    For a given reversible Markov chain (Q, pi, G), simulate the decay of the
    Fisher Dirichlet quadratic

        F(t) = - δp(t)^T G δp(t)  ≥ 0,

    for N_init random initial conditions over a time grid of length N_T up to
    T_max. Uses the eigen decomposition of the symmetrised generator S to
    evolve exactly, and parallelises over initial conditions.

    Returns:
      t_grid: time grid (N_T,)
      F_all: array of shape (N_init, N_T)
      rates: array of estimated decay rates r_est, obtained by least squares
             fit of log F(t) on the late time window t >= t_min_frac * T_max.
    """
    dim = Q.shape[0]
    S, B, B_inv = markov_symmetrised_generator(Q, pi)
    eigvals, eigvecs = np.linalg.eigh(S)

    t_grid = np.linspace(0.0, T_max, N_T)
    exp_ev_t = np.exp(eigvals[:, None] * t_grid[None, :])

    rng = np.random.default_rng(rng_seed)

    def compute_F_for_init(seed_offset: int):
        local_rng = np.random.default_rng(rng_seed + 1000 * seed_offset)
        p0 = local_rng.random(dim)
        p0 /= p0.sum()

        dp0 = p0 - pi
        z0 = B_inv @ dp0
        y0 = eigvecs.T @ z0

        Y = y0[:, None] * exp_ev_t
        Z = eigvecs @ Y
        DP = B @ Z

        P = DP + pi[:, None]
        P = np.clip(P, 1e-15, None)
        P /= P.sum(axis=0, keepdims=True)
        DP = P - pi[:, None]

        # Correct: Dirichlet energy is minus the G quadratic
        F_t = -np.einsum("it,ij,jt->t", DP, G, DP)

        t_min = t_grid[0] + t_min_frac * (t_grid[-1] - t_grid[0])
        mask = t_grid >= t_min
        t_fit = t_grid[mask]
        F_fit = F_t[mask]
        F_fit = np.maximum(F_fit, 1e-30)
        y = np.log(F_fit)

        A = np.vstack([np.ones_like(t_fit), -t_fit]).T  # y ≈ a - r t
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r_est = coef[1]
        return F_t, float(r_est)

    F_all = np.zeros((N_init, N_T), dtype=float)
    rates = np.zeros(N_init, dtype=float)

    n_workers_eff = max(1, min(n_workers, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=n_workers_eff) as ex:
        futures = {ex.submit(compute_F_for_init, k): k for k in range(N_init)}
        for fut in as_completed(futures):
            k = futures[fut]
            F_k, r_k = fut.result()
            F_all[k, :] = F_k
            rates[k] = r_k

    return t_grid, F_all, rates


def main():
    print("Qutrit Markov vs FP-like Markov: universal gap driven Fisher decay")
    print("-----------------------------------------------------------------")

    # Qutrit Markov chain
    Q_q, pi_q = build_qutrit_markov(beta=1.0)
    print("Qutrit Markov generator Q_q:")
    print(Q_q)
    print("pi_q (thermal probabilities) =", pi_q)

    col_sums = Q_q.sum(axis=0)
    resid_pi = np.linalg.norm(Q_q @ pi_q)
    print("Q_q column sums (should be ~0):", col_sums)
    print("||Q_q pi_q||_2 ≈ %.3e (should be near 0)" % resid_pi)

    lambda_Q_q, eigvals_q, eigvecs_q, B_q, B_inv_q = markov_gap(Q_q, pi_q)
    G_q = fisher_dirichlet_operator(Q_q, pi_q)
    G_q_sym_resid = np.linalg.norm(G_q - G_q.T) / max(np.linalg.norm(G_q), 1e-16)
    lambda_G_q, eigvals_G_q = fisher_curvature_gap(G_q)

    print()
    print("Qutrit Markov spectral and Fisher diagnostics:")
    print("  Eigenvalues of symmetrised generator S_q:", eigvals_q)
    print("  Markov spectral gap lambda_Q_q ≈ %.6e" % lambda_Q_q)
    print("  G_q symmetry residual ||G_q - G_q^T||/||G_q|| ≈ %.3e" % G_q_sym_resid)
    print("  Eigenvalues of -G_q (Fisher metric):", -eigvals_G_q)
    print("  Fisher curvature gap lambda_G_q ≈ %.6e" % lambda_G_q)

    # FP-like chain
    Nx = 256
    x, dx, Q_fp, pi_fp = build_fp_markov_chain(Nx=Nx)
    col_sums_fp = Q_fp.sum(axis=0)
    resid_pi_fp = np.linalg.norm(Q_fp @ pi_fp)
    print()
    print("FP-like Markov generator Q_fp on ring (Nx = %d):" % Nx)
    print("  Column sums (should be ~0): min ≈ %.3e, max ≈ %.3e"
          % (col_sums_fp.min(), col_sums_fp.max()))
    print("  ||Q_fp pi_fp||_2 ≈ %.3e (should be near 0)" % resid_pi_fp)

    lambda_Q_fp, eigvals_fp, eigvecs_fp, B_fp, B_inv_fp = markov_gap(Q_fp, pi_fp)
    print("  Eigenvalues of symmetrised S_fp (first few):", eigvals_fp[:5])
    print("  Raw FP Markov gap lambda_Q_fp ≈ %.6e" % lambda_Q_fp)

    # Gap matching
    scale = lambda_Q_q / lambda_Q_fp if lambda_Q_fp > 0.0 else 1.0
    Q_fp_scaled = scale * Q_fp
    lambda_Q_fp_scaled, eigvals_fp_scaled, _, _, _ = markov_gap(Q_fp_scaled, pi_fp)

    print()
    print("Gap matching for FP-like chain:")
    print("  Scaling factor applied to Q_fp: %.6e" % scale)
    print("  lambda_Q_q (qutrit gap)       ≈ %.6e" % lambda_Q_q)
    print("  lambda_Q_fp_scaled (FP gap)   ≈ %.6e" % lambda_Q_fp_scaled)
    print("  Relative difference           ≈ %.3e"
          % (abs(lambda_Q_fp_scaled - lambda_Q_q) / max(lambda_Q_q, 1e-16)))

    # Fisher operator for FP-like chain
    G_fp = fisher_dirichlet_operator(Q_fp_scaled, pi_fp)
    G_fp_sym_resid = np.linalg.norm(G_fp - G_fp.T) / max(np.linalg.norm(G_fp), 1e-16)
    lambda_G_fp, eigvals_G_fp = fisher_curvature_gap(G_fp)
    print()
    print("FP-like Fisher Dirichlet diagnostics:")
    print("  G_fp symmetry residual ||G_fp - G_fp^T||/||G_fp|| ≈ %.3e" % G_fp_sym_resid)
    print("  Eigenvalues of -G_fp (first few):", (-eigvals_G_fp)[:5])
    print("  Fisher curvature gap lambda_G_fp ≈ %.6e" % lambda_G_fp)

    print()
    print("Gap hierarchy comparison (qutrit vs FP-like):")
    print("  Qutrit:  lambda_G_q ≈ %.6e, lambda_Q_q ≈ %.6e"
          % (lambda_G_q, lambda_Q_q))
    print("  FP-like: lambda_G_fp ≈ %.6e, lambda_Q_fp_scaled ≈ %.6e"
          % (lambda_G_fp, lambda_Q_fp_scaled))

    # Time evolution parameters
    N_INIT = 32
    N_T = 200
    T_max = 10.0  # same 0–10 window you just used
    t_min_frac = 0.5
    rng_seed = 314159
    n_workers = min(20, os.cpu_count() or 1)

    print()
    print("Time evolution parameters:")
    print("  N_INIT = %d, N_T = %d, T_max ≈ %.3e, t_min_frac = %.2f"
          % (N_INIT, N_T, T_max, t_min_frac))
    print("  Detected CPU cores: %d, worker threads used: %d"
          % (os.cpu_count() or 1, n_workers))

    # Qutrit chain
    print()
    print("Simulating Fisher Dirichlet decay for qutrit Markov chain...")
    t_grid_q, F_all_q, rates_q = simulate_F_decay_for_chain(
        Q_q, pi_q, G_q,
        T_max=T_max,
        N_T=N_T,
        N_init=N_INIT,
        t_min_frac=t_min_frac,
        rng_seed=rng_seed,
        n_workers=n_workers,
    )
    r_pred = 2.0 * lambda_Q_q
    rel_err_q = np.abs(rates_q - r_pred) / max(r_pred, 1e-16)
    print("  Predicted asymptotic decay rate r_pred = 2 lambda_Q_q ≈ %.6e" % r_pred)
    print("  Number of rates =", rates_q.size)
    print("  r_q mean ≈ %.6e, min ≈ %.6e, max ≈ %.6e"
          % (rates_q.mean(), rates_q.min(), rates_q.max()))
    print("  Relative error vs r_pred: mean ≈ %.3e, min ≈ %.3e, max ≈ %.3e"
          % (rel_err_q.mean(), rel_err_q.min(), rel_err_q.max()))

    # FP-like chain
    print()
    print("Simulating Fisher Dirichlet decay for FP-like Markov chain...")
    t_grid_fp, F_all_fp, rates_fp = simulate_F_decay_for_chain(
        Q_fp_scaled, pi_fp, G_fp,
        T_max=T_max,
        N_T=N_T,
        N_init=N_INIT,
        t_min_frac=t_min_frac,
        rng_seed=rng_seed + 8675309,
        n_workers=n_workers,
    )
    rel_err_fp = np.abs(rates_fp - r_pred) / max(r_pred, 1e-16)
    print("  Predicted asymptotic decay rate r_pred = 2 lambda_Q_q ≈ %.6e" % r_pred)
    print("  Number of rates =", rates_fp.size)
    print("  r_fp mean ≈ %.6e, min ≈ %.6e, max ≈ %.6e"
          % (rates_fp.mean(), rates_fp.min(), rates_fp.max()))
    print("  Relative error vs r_pred: mean ≈ %.3e, min ≈ %.3e, max ≈ %.3e"
          % (rel_err_fp.mean(), rel_err_fp.min(), rel_err_fp.max()))

    # Pass/fail
    tol_gap = 1e-6
    tol_rate_rel = 0.2

    gap_match_ok = abs(lambda_Q_fp_scaled - lambda_Q_q) <= tol_gap
    rate_q_ok = rel_err_q.mean() <= tol_rate_rel
    rate_fp_ok = rel_err_fp.mean() <= tol_rate_rel

    print()
    print("Summary of universal gap driven decay checks:")
    print("  Gap match lambda_Q_fp_scaled ≈ lambda_Q_q?            %s (tol = %.1e)"
          % ("True" if gap_match_ok else "False", tol_gap))
    print("  Qutrit Fisher decay rates ≈ 2 lambda_Q_q on average?   %s (tol_rel = %.1e)"
          % ("True" if rate_q_ok else "False", tol_rate_rel))
    print("  FP-like Fisher decay rates ≈ 2 lambda_Q_q on average?  %s (tol_rel = %.1e)"
          % ("True" if rate_fp_ok else "False", tol_rate_rel))

    all_ok = gap_match_ok and rate_q_ok and rate_fp_ok
    print()
    if all_ok:
        print("Qutrit vs FP-like universal gap driven Fisher decay CHECK: PASS")
        print("  A 3-state qutrit-derived Markov chain and a high resolution")
        print("  FP-like chain, once gap matched, exhibit essentially the same")
        print("  Fisher Dirichlet decay clock set by 2 lambda_Q_q, despite")
        print("  very different microscopic dynamics.")
    else:
        print("Qutrit vs FP-like universal gap driven Fisher decay CHECK: FAIL")
        print("  At least one of the conditions (gap match or decay rate")
        print("  clustering) did not meet the specified tolerances. Inspect")
        print("  diagnostics above for details.")


if __name__ == "__main__":
    main()
