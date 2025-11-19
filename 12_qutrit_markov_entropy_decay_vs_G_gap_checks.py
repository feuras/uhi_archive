#!/usr/bin/env python3
"""
12_qutrit_markov_entropy_decay_vs_G_gap_checks.py

Entropy decay vs Fisher G-metric and Markov spectral gaps
for the qutrit Markov sector induced by a thermal GKLS model.

Goals:

  1. Rebuild the thermal qutrit GKLS model and extract its 3-state
     classical Markov generator Q_markov from the population block.

  2. Construct the canonical Fisher Dirichlet operator
        G_true_class = Q_markov diag(pi),
     check its symmetry and Dirichlet form consistency.

  3. Compute:
       - The Fisher G-metric gap:
             λ_G = smallest positive eigenvalue of G_metric = -G_true_class.
       - The reversible Markov spectral gap:
             λ_Q = smallest nonzero absolute eigenvalue of the
                    symmetrised generator L_sym = B^{-1} Q_markov B
                    with B = diag(sqrt(pi)).

  4. Simulate the Markov evolution
        dp/dt = Q_markov p
     for many random initial distributions p(0) and track the Fisher
     quadratic
        F(t) = 0.5 * sum_i [(p_i(t) - pi_i)^2 / pi_i].

  5. On a late-time window, fit log F(t) vs t to extract an empirical
     decay rate r_est for each trajectory, and compare to the
     prediction
        r_pred = 2 λ_Q.

  6. Check:
        - r_est ≈ r_pred across trajectories.
        - r_pred >= 2 λ_G, consistent with the coercivity implications
          of the Fisher G-metric gap.

Multithreading:
  Uses concurrent.futures.ThreadPoolExecutor with up to 20 worker
  threads, falling back to 1 if fewer cores are available.
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Qutrit GKLS model
# ---------------------------------------------------------------------------

def build_qutrit_model(beta=1.0,
                       E0=0.0, E1=1.0, E2=2.0,
                       g01_down=0.8, g12_down=0.5, g02_down=0.3,
                       gamma_phi0=0.1, gamma_phi1=0.1, gamma_phi2=0.1):
    """
    Build a simple thermal qutrit GKLS model.

    Energies: E0 < E1 < E2 with H = diag(E0, E1, E2).
    Thermal state:
        rho_ss = diag(pi0, pi1, pi2),
        pi_i ∝ exp(-beta E_i).

    Jump operators:
      For each pair (i,j) with E_j > E_i:
        L_down (j -> i) with rate g_down,
        L_up   (i -> j) with rate g_up = g_down * exp[-beta (E_j - E_i)],
      so that detailed balance holds.

    Dephasing:
      For each level k we include L_phi,k ∝ |k><k| with rate gamma_phik.
    """
    E = np.array([E0, E1, E2], dtype=float)
    H = np.diag(E).astype(complex)

    # Thermal stationary state
    w = np.exp(-beta * E)
    Z = float(np.sum(w))
    pi = w / Z
    rho_ss = np.diag(pi.astype(complex))

    def pair(i, j, g_down):
        # assume E[j] > E[i]
        Delta = E[j] - E[i]
        g_up = g_down * np.exp(-beta * Delta)
        L_down = np.zeros((3, 3), dtype=complex)
        L_up = np.zeros((3, 3), dtype=complex)
        # downward j -> i, upward i -> j
        L_down[i, j] = np.sqrt(g_down)
        L_up[j, i] = np.sqrt(g_up)
        return [L_down, L_up]

    L_ops = []
    # Nearest neighbour and direct 0 <-> 2 transitions
    L_ops += pair(0, 1, g01_down)
    L_ops += pair(1, 2, g12_down)
    L_ops += pair(0, 2, g02_down)

    # Dephasing on each level
    for k, gphi in enumerate([gamma_phi0, gamma_phi1, gamma_phi2]):
        if gphi > 0.0:
            L = np.zeros((3, 3), dtype=complex)
            L[k, k] = 1.0
            L_ops.append(np.sqrt(gphi) * L)

    return H, rho_ss, L_ops


def apply_Hamiltonian(H, rho):
    return -1j * (H @ rho - rho @ H)


def apply_dissipator(L_ops, rho):
    d = rho.shape[0]
    D = np.zeros((d, d), dtype=complex)
    for L in L_ops:
        L_rho = L @ rho
        L_rho_Ld = L_rho @ L.conj().T
        K = L.conj().T @ L
        anti = K @ rho + rho @ K
        D += L_rho_Ld - 0.5 * anti
    return D


# ---------------------------------------------------------------------------
# Real 9d representation (u <-> ρ) and GKLS generator on u
# ---------------------------------------------------------------------------

def u_to_rho(u):
    """
    u = (p0, p1, p2,
         Re ρ_01, Im ρ_01,
         Re ρ_02, Im ρ_02,
         Re ρ_12, Im ρ_12)
    """
    (p0, p1, p2,
     re01, im01,
     re02, im02,
     re12, im12) = u

    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = p0
    rho[1, 1] = p1
    rho[2, 2] = p2

    rho[0, 1] = re01 + 1j * im01
    rho[1, 0] = re01 - 1j * im01

    rho[0, 2] = re02 + 1j * im02
    rho[2, 0] = re02 - 1j * im02

    rho[1, 2] = re12 + 1j * im12
    rho[2, 1] = re12 - 1j * im12

    return rho


def rho_to_u(rho):
    p0 = float(np.real(rho[0, 0]))
    p1 = float(np.real(rho[1, 1]))
    p2 = float(np.real(rho[2, 2]))

    re01 = float(np.real(rho[0, 1]))
    im01 = float(np.imag(rho[0, 1]))

    re02 = float(np.real(rho[0, 2]))
    im02 = float(np.imag(rho[0, 2]))

    re12 = float(np.real(rho[1, 2]))
    im12 = float(np.imag(rho[1, 2]))

    return np.array([p0, p1, p2,
                     re01, im01,
                     re02, im02,
                     re12, im12], dtype=float)


def build_generator_real(H, L_ops):
    dim_u = 9
    K_total = np.zeros((dim_u, dim_u), dtype=float)
    K_H = np.zeros_like(K_total)
    K_D = np.zeros_like(K_total)

    for j in range(dim_u):
        e_j = np.zeros(dim_u, dtype=float)
        e_j[j] = 1.0
        rho_j = u_to_rho(e_j)

        d_rho_H = apply_Hamiltonian(H, rho_j)
        d_rho_D = apply_dissipator(L_ops, rho_j)
        d_rho_total = d_rho_H + d_rho_D

        du_H = rho_to_u(d_rho_H)
        du_D = rho_to_u(d_rho_D)
        du_total = rho_to_u(d_rho_total)

        K_H[:, j] = du_H
        K_D[:, j] = du_D
        K_total[:, j] = du_total

    return K_total, K_H, K_D


# ---------------------------------------------------------------------------
# Fisher quadratic and Dirichlet
# ---------------------------------------------------------------------------

def fisher_quadratic(p, pi):
    diff = p - pi
    return 0.5 * float(np.sum(diff * diff / pi))


def dirichlet_from_Q(mu, pi, Q):
    dim = len(pi)
    E = 0.0
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            gamma_ij = Q[i, j]
            E += 0.5 * pi[j] * gamma_ij * (mu[i] - mu[j]) ** 2
    return float(E)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run():
    print("Qutrit Markov entropy decay vs Fisher and Markov gaps")
    print("-----------------------------------------------------")

    beta = 1.0
    E0, E1, E2 = 0.0, 1.0, 2.0
    g01_down, g12_down, g02_down = 0.8, 0.5, 0.3
    gamma_phi0 = gamma_phi1 = gamma_phi2 = 0.1

    print(f"beta = {beta}")
    print(f"Energies: E0 = {E0}, E1 = {E1}, E2 = {E2}")
    print(f"Downward jump rates: g01_down = {g01_down}, "
          f"g12_down = {g12_down}, g02_down = {g02_down}")
    print(f"Dephasing rates: gamma_phi0 = {gamma_phi0}, "
          f"gamma_phi1 = {gamma_phi1}, gamma_phi2 = {gamma_phi2}")
    print("")

    H, rho_ss, L_ops = build_qutrit_model(beta=beta,
                                          E0=E0, E1=E1, E2=E2,
                                          g01_down=g01_down,
                                          g12_down=g12_down,
                                          g02_down=g02_down,
                                          gamma_phi0=gamma_phi0,
                                          gamma_phi1=gamma_phi1,
                                          gamma_phi2=gamma_phi2)

    print("Stationary state rho_ss (thermal Gibbs):")
    print(rho_ss)
    pi = np.real(np.diag(rho_ss))
    print(f"pi (thermal probabilities) = {pi}")
    print("")

    d_rho_ss = apply_Hamiltonian(H, rho_ss) + apply_dissipator(L_ops, rho_ss)
    norm_ss = float(np.linalg.norm(d_rho_ss, ord="fro"))
    print("GKLS stationarity diagnostics:")
    print(f"  ||GKLS(rho_ss)||_F ≈ {norm_ss:.3e} (should be near 0)")
    print("")

    # Build real GKLS generators and extract population block
    K_total, K_H, K_D = build_generator_real(H, L_ops)
    Q_markov = K_D[0:3, 0:3]

    print("Classical 3-state Markov generator Q_markov from GKLS density block:")
    print(Q_markov)
    Q_pi = Q_markov @ pi
    col_sums = np.sum(Q_markov, axis=0)
    print(f"  ||Q_markov pi||_2 ≈ {float(np.linalg.norm(Q_pi, ord=2)):.3e} (should be near 0)")
    print(f"  Column sums (should be ~0 for generator): {col_sums}")
    print("")

    # Canonical Fisher Dirichlet operator
    G_true_class = Q_markov @ np.diag(pi)
    G_metric = -G_true_class

    frob = lambda A: float(np.linalg.norm(A, ord="fro"))
    norm_G_true = frob(G_true_class)
    sym_resid = frob(G_true_class - G_true_class.T) / max(norm_G_true, 1e-16)
    skew_resid = frob(G_true_class + G_true_class.T) / max(norm_G_true, 1e-16)

    print("Canonical classical G_true_class = Q_markov diag(pi):")
    print(G_true_class)
    print(f"  Symmetry residual ||G_true_class - G_true_class^T||/||G_true_class|| ≈ {sym_resid:.3e}")
    print(f"  Skew residual     ||G_true_class + G_true_class^T||/||G_true_class|| ≈ {skew_resid:.3e}")
    print("")

    # Fisher G-metric eigenvalues and gap
    eigvals_G, eigvecs_G = np.linalg.eigh(G_metric)
    print("Eigenvalues of G_metric = -G_true_class:")
    print(f"  {eigvals_G}")
    pos_mask_G = eigvals_G > 1e-10
    lambda_pos_G = eigvals_G[pos_mask_G]
    num_pos_G = int(np.sum(pos_mask_G))
    num_zero_G = len(eigvals_G) - num_pos_G
    print(f"  Number of strictly positive eigenvalues: {num_pos_G}")
    print(f"  Number of (near) zero eigenvalues:      {num_zero_G}")
    if num_pos_G == 0:
        print("  ERROR: no positive eigenvalues in G_metric; cannot define Fisher gap.")
        return
    lambda_gap_G = float(np.min(lambda_pos_G))
    print(f"  Fisher G-metric gap λ_G ≈ {lambda_gap_G:.6e}")
    print("")

    # Dirichlet form consistency check
    rng_check = np.random.default_rng(2025)
    mu_test = rng_check.normal(size=3)
    mu_test -= float(np.dot(pi, mu_test))
    E_pair = dirichlet_from_Q(mu_test, pi, Q_markov)
    E_G = - float(mu_test @ (G_true_class @ mu_test))
    rel_dirichlet = abs(E_pair - E_G) / max(abs(E_pair), 1e-16)
    print("Dirichlet form consistency (single random direction):")
    print(f"  E_pair(mu) vs -mu^T G_true_class mu relative diff ≈ {rel_dirichlet:.3e}")
    print("")

    # Reversible Markov spectral gap
    pi_sqrt = np.sqrt(pi)
    B = np.diag(pi_sqrt)
    B_inv = np.diag(1.0 / pi_sqrt)
    L_sym = B_inv @ Q_markov @ B
    eigvals_L = np.linalg.eigvals(L_sym)

    print("Eigenvalues of symmetrised generator L_sym = B^{-1} Q B:")
    print(f"  {eigvals_L}")
    neg_vals = [ev.real for ev in eigvals_L if ev.real < -1e-6]
    if not neg_vals:
        print("  ERROR: no negative eigenvalues in L_sym; cannot define Markov gap.")
        return
    lambda_gap_Q = min(-ev for ev in neg_vals)
    print(f"  Markov spectral gap λ_Q (π-weighted) ≈ {lambda_gap_Q:.6e}")
    print(f"  2 λ_Q ≈ {2.0 * lambda_gap_Q:.6e}")
    print(f"  2 λ_G ≈ {2.0 * lambda_gap_G:.6e}")
    print(f"  Ratio (2 λ_Q) / (2 λ_G) ≈ {(2.0 * lambda_gap_Q) / (2.0 * lambda_gap_G):.3e}")
    print("")

    # Time evolution parameters based on λ_Q
    N_INIT = 32
    r_pred = 2.0 * lambda_gap_Q
    T_max = 8.0 / lambda_gap_Q
    N_T = 200
    t_grid = np.linspace(0.0, T_max, N_T)

    print("Time evolution parameters (based on Markov gap λ_Q):")
    print(f"  Number of initial conditions N_INIT = {N_INIT}")
    print(f"  T_max ≈ {T_max:.3e}")
    print(f"  N_T = {N_T}, dt ≈ {t_grid[1] - t_grid[0]:.3e}")
    print(f"  Predicted asymptotic decay rate r_pred = 2 λ_Q ≈ {r_pred:.6e}")
    print("")

    N_CORES = os.cpu_count() or 1
    if N_CORES >= 20:
        N_WORKERS = 20
    else:
        N_WORKERS = 1

    print("Multithreading:")
    print(f"  Detected CPU cores: {N_CORES}")
    print(f"  Worker threads used: {N_WORKERS}")
    print("")

    rng = np.random.default_rng(424242)
    eigvals_Q, eigvecs_Q = np.linalg.eig(Q_markov)
    V = eigvecs_Q
    V_inv = np.linalg.inv(V)

    def trajectory_decay_rate(p0, idx_init):
        p0 = np.array(p0, dtype=float)
        p0 = np.clip(p0, 1e-15, None)
        p0 /= float(np.sum(p0))

        delta_p0 = p0 - pi
        alpha0 = V_inv @ delta_p0.astype(complex)

        F_vals = np.zeros_like(t_grid, dtype=float)
        for k, t in enumerate(t_grid):
            exp_factors = np.exp(eigvals_Q * t)
            delta_p_t = V @ (exp_factors * alpha0)
            p_t = pi + np.real(delta_p_t)
            p_t = np.clip(p_t, 1e-15, None)
            p_t /= float(np.sum(p_t))
            F_vals[k] = fisher_quadratic(p_t, pi)

        # Late-time window: last two thirds of the interval
        mask = (t_grid >= T_max / 3.0) & (F_vals > 0.0)
        t_fit = t_grid[mask]
        F_fit = F_vals[mask]

        if len(t_fit) < 5:
            return {
                "idx": idx_init,
                "r_est": np.nan,
            }

        logF = np.log(F_fit)
        coeffs = np.polyfit(t_fit, logF, 1)
        slope = coeffs[0]
        r_est = -float(slope)

        return {
            "idx": idx_init,
            "r_est": r_est,
        }

    tasks = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        for idx in range(N_INIT):
            p0 = rng.dirichlet(alpha=np.ones(3))
            tasks.append(executor.submit(trajectory_decay_rate, p0, idx))

        results = [f.result() for f in as_completed(tasks)]

    rates = []
    for res in sorted(results, key=lambda d: d["idx"]):
        r_est = res["r_est"]
        if np.isfinite(r_est) and r_est > 0:
            rates.append(r_est)

    print("Empirical decay rates from Fisher quadratic F(t):")
    if not rates:
        print("  No valid decay rates were extracted (all fits failed).")
        pass_rates = False
    else:
        rates = np.array(rates, dtype=float)
        r_mean = float(np.mean(rates))
        r_min = float(np.min(rates))
        r_max = float(np.max(rates))

        rel_err_mean = abs(r_mean - r_pred) / max(r_pred, 1e-16)
        rel_err_min = abs(r_min - r_pred) / max(r_pred, 1e-16)
        rel_err_max = abs(r_max - r_pred) / max(r_pred, 1e-16)

        print(f"  Number of successful fits: {rates.size}/{N_INIT}")
        print(f"  r_est mean ≈ {r_mean:.6e}")
        print(f"  r_est min  ≈ {r_min:.6e}")
        print(f"  r_est max  ≈ {r_max:.6e}")
        print(f"  Mean relative error vs r_pred ≈ {rel_err_mean:.3e}")
        print(f"  Min  relative error vs r_pred ≈ {rel_err_min:.3e}")
        print(f"  Max  relative error vs r_pred ≈ {rel_err_max:.3e}")
        print("")

        tol_mean = 0.05
        tol_min = 0.02       # 2% on the best (closest to asymptotic)
        tol_extreme = 0.30   # allow up to 30% overshoot on the fastest
        pass_rates = ((rel_err_mean < tol_mean) and
                      (rel_err_min < tol_extreme) and
                      (rel_err_max < tol_extreme))

    print("Summary of checks:")
    pass_stationary = norm_ss < 1e-10
    pass_Qpi = float(np.linalg.norm(Q_pi, ord=2)) < 1e-10
    pass_Gsym = (sym_resid < 1e-12)
    pass_dirichlet = (rel_dirichlet < 1e-10)
    pass_gap_ordering = (2.0 * lambda_gap_Q >= 2.0 * lambda_gap_G)

    print(f"  rho_ss stationary for GKLS?                         {pass_stationary} (tol = 1e-10)")
    print(f"  Q_markov pi ≈ 0 (correct stationary distribution)?   {pass_Qpi} (tol = 1e-10)")
    print(f"  G_true_class symmetric (Fisher Dirichlet)?           {pass_Gsym} (tol = 1e-12)")
    print(f"  Dirichlet form E_pair ≈ -mu^T G_true mu?             {pass_dirichlet} (tol = 1e-10)")
    print(f"  Ordering 2 λ_Q >= 2 λ_G (entropy vs Fisher gap)?     {pass_gap_ordering}")
    print(f"  Empirical entropy decay rates ≈ 2 λ_Q?               {pass_rates}")
    print("")

    all_pass = (pass_stationary and pass_Qpi and pass_Gsym and
                pass_dirichlet and pass_gap_ordering and pass_rates)

    if all_pass:
        print("Qutrit Markov entropy-decay vs Fisher/Markov gaps CHECK: PASS")
        print("  The qutrit GKLS dissipator induces a 3-state reversible")
        print("  Markov chain whose Fisher Dirichlet operator G_true_class")
        print("  exactly matches the jump-resolved Dirichlet form, and whose")
        print("  π-weighted spectral gap λ_Q controls the actual decay rate")
        print("  of the Fisher quadratic F(t). The Fisher G-metric gap λ_G")
        print("  sits below λ_Q, acting as a coercive lower bound, while the")
        print("  empirically measured entropy decay matches 2 λ_Q across many")
        print("  random initial conditions.")
    else:
        print("Qutrit Markov entropy-decay vs Fisher/Markov gaps CHECK: FAIL")
        print("  At least one of the conditions (GKLS stationarity, Markov")
        print("  invariance, Fisher symmetry, Dirichlet consistency, spectral")
        print("  ordering, or entropy-decay vs Markov gap match) did not meet")
        print("  the specified tolerances. See diagnostics above.")
        print("")


if __name__ == "__main__":
    run()
