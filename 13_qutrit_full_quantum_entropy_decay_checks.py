#!/usr/bin/env python3
"""
13_qutrit_full_quantum_entropy_decay_checks.py

Full quantum entropy decay for a thermal qutrit GKLS model, with comparison
to both the classical Markov spectral gap (density sector) and the GKLS
spectral gap (full quantum generator).

Pipeline:

  1. Build the same thermal qutrit GKLS model used in previous scripts.

  2. Extract the 3 state classical Markov generator Q_markov from the
     population block of the dissipative generator.

  3. Compute the pi weighted reversible spectral gap lambda_Q of Q_markov via
     the symmetrised generator L_sym = B^{-1} Q B with B = diag(sqrt(pi)).

  4. Build the full 9x9 real GKLS generator K_total on u space, where
        u = (p0, p1, p2,
             Re rho_01, Im rho_01,
             Re rho_02, Im rho_02,
             Re rho_12, Im rho_12).

  5. Diagonalise K_total once and propagate many random initial quantum
     states via
        u(t) = u_ss + V exp(Lambda t) V^{-1} (u_0 - u_ss),
     with rho(t) = u_to_rho(u(t)).

  6. For each trajectory, compute at each time:
        D_HS(t) = ||rho(t) - rho_ss||_F^2,
        S_rel(t) = Tr[rho(t) (log rho(t) - log rho_ss)].

  7. On a late time window, fit log D_HS(t) and log S_rel(t) vs t to
     extract empirical decay rates r_HS and r_rel.

Checks:

  - GKLS stationarity and Markov invariance hold,
  - the pi weighted spectral gap lambda_Q is well defined,
  - the GKLS spectral gap lambda_K is extracted from K_total,
  - empirical decay rates for both D_HS and S_rel are consistent with
    2 lambda_K as the quantum asymptotic clock, and bounded above by
    2 lambda_Q.

Multithreading:
  Uses ThreadPoolExecutor with up to 20 worker threads, defaulting to 1
  if fewer cores are available.
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
        pi_i proportional to exp(-beta E_i).

    Jump operators:
      For each pair (i,j) with E_j > E_i:
        L_down (j -> i) with rate g_down,
        L_up   (i -> j) with rate g_up = g_down * exp[-beta (E_j - E_i)],
      so that detailed balance holds.

    Dephasing:
      For each level k we include L_phi,k proportional to |k><k|
      with rate gamma_phik.
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
# Real 9d representation (u <-> rho) and GKLS generator on u
# ---------------------------------------------------------------------------

def u_to_rho(u):
    """
    u = (p0, p1, p2,
         Re rho_01, Im rho_01,
         Re rho_02, Im rho_02,
         Re rho_12, Im rho_12)
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
    """
    Build the 9x9 real generators K_total, K_H, K_D on u space:

        du/dt = K u,

    by acting with the GKLS generator on the basis vectors in u space.
    """
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
# Quantum entropy and distances
# ---------------------------------------------------------------------------

def matrix_log_hermitian(rho, eps=1e-15):
    """
    Matrix logarithm for a positive Hermitian 3x3 matrix using eigen
    decomposition. Eigenvalues are clipped below eps to avoid log(0).
    """
    vals, vecs = np.linalg.eigh(rho)
    vals_clipped = np.clip(np.real(vals), eps, None)
    log_vals = np.log(vals_clipped)
    return vecs @ np.diag(log_vals) @ vecs.conj().T


def quantum_relative_entropy(rho, sigma, eps=1e-15):
    """
    S(rho || sigma) = Tr[rho (log rho - log sigma)] for full rank sigma.
    """
    log_rho = matrix_log_hermitian(rho, eps=eps)
    log_sigma = matrix_log_hermitian(sigma, eps=eps)
    diff = log_rho - log_sigma
    return float(np.real(np.trace(rho @ diff)))


def hs_distance_squared(rho, sigma):
    """
    Hilbert Schmidt distance squared: ||rho - sigma||_F^2.
    """
    diff = rho - sigma
    return float(np.linalg.norm(diff, ord="fro") ** 2)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run():
    print("Qutrit full quantum entropy decay vs classical and GKLS gaps")
    print("-----------------------------------------------------------")

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

    # Build GKLS model and stationary state
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

    # GKLS stationarity check
    d_rho_ss = apply_Hamiltonian(H, rho_ss) + apply_dissipator(L_ops, rho_ss)
    norm_ss = float(np.linalg.norm(d_rho_ss, ord="fro"))
    print("GKLS stationarity diagnostics:")
    print(f"  ||GKLS(rho_ss)||_F ≈ {norm_ss:.3e} (should be near 0)")
    print("")

    # Build real GKLS generators and extract population block
    K_total, K_H, K_D = build_generator_real(H, L_ops)
    Q_markov = K_D[0:3, 0:3]

    print("Classical 3 state Markov generator Q_markov from GKLS density block:")
    print(Q_markov)
    Q_pi = Q_markov @ pi
    col_sums = np.sum(Q_markov, axis=0)
    print(f"  ||Q_markov pi||_2 ≈ {float(np.linalg.norm(Q_pi, ord=2)):.3e} (should be near 0)")
    print(f"  Column sums (should be ~0 for generator): {col_sums}")
    print("")

    # Markov pi weighted spectral gap lambda_Q
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
    r_pred_markov = 2.0 * lambda_gap_Q
    print(f"  Markov spectral gap lambda_Q (pi weighted) ≈ {lambda_gap_Q:.6e}")
    print(f"  Predicted decay rate from Markov gap 2 lambda_Q ≈ {r_pred_markov:.6e}")
    print("")

    # Eigen decomposition of full GKLS generator on u
    eigvals_K, eigvecs_K = np.linalg.eig(K_total)
    V_K = eigvecs_K
    V_K_inv = np.linalg.inv(V_K)

    # GKLS spectral gap lambda_K from real parts of eigenvalues
    real_parts = np.real(eigvals_K)
    neg_real = real_parts[real_parts < -1e-8]
    if neg_real.size == 0:
        print("  ERROR: no negative eigenvalues in K_total; cannot define GKLS gap.")
        return
    lambda_gap_K = float(np.min(-neg_real))
    r_pred_quantum = 2.0 * lambda_gap_K

    # Stationary u vector
    u_ss = rho_to_u(rho_ss)
    K_u_ss = K_total @ u_ss
    print("Full GKLS generator K_total (selected diagnostics):")
    print(f"  Dimension of u space: {len(u_ss)}")
    print(f"  ||K_total u_ss||_2 ≈ {float(np.linalg.norm(K_u_ss, ord=2)):.3e} (should be near 0)")
    print("  Eigenvalues of K_total (real parts should be <= 0):")
    print(f"  {eigvals_K}")
    print("")
    print(f"  GKLS spectral gap lambda_K ≈ {lambda_gap_K:.6e}")
    print(f"  Quantum asymptotic decay rate 2 lambda_K ≈ {r_pred_quantum:.6e}")
    print("")

    # Time evolution parameters based on GKLS gap lambda_K
    N_INIT = 32
    T_max = 8.0 / lambda_gap_K
    N_T = 200
    t_grid = np.linspace(0.0, T_max, N_T)

    print("Time evolution parameters (full quantum, based on GKLS gap lambda_K):")
    print(f"  Number of initial states N_INIT = {N_INIT}")
    print(f"  T_max ≈ {T_max:.3e}")
    print(f"  N_T = {N_T}, dt ≈ {t_grid[1] - t_grid[0]:.3e}")
    print("")

    # Multithreading setup
    N_CORES = os.cpu_count() or 1
    if N_CORES >= 20:
        N_WORKERS = 20
    else:
        N_WORKERS = 1

    print("Multithreading:")
    print(f"  Detected CPU cores: {N_CORES}")
    print(f"  Worker threads used: {N_WORKERS}")
    print("")

    # Prepare independent seeds for each trajectory
    base_rng = np.random.default_rng(777)
    seeds = base_rng.integers(low=0, high=2**32 - 1, size=N_INIT, dtype=np.uint32)

    # Helper to propagate a single trajectory and fit decay rates
    def trajectory_decay_rates(idx_init, seed):
        rng = np.random.default_rng(int(seed))

        # Random full rank initial density matrix via Ginibre ensemble
        X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        rho0 = X @ X.conj().T
        rho0 /= np.trace(rho0)
        u0 = rho_to_u(rho0)

        delta_u0 = u0 - u_ss
        alpha0 = V_K_inv @ delta_u0.astype(complex)

        D_HS_vals = np.zeros_like(t_grid, dtype=float)
        S_rel_vals = np.zeros_like(t_grid, dtype=float)

        for k, t in enumerate(t_grid):
            exp_factors = np.exp(eigvals_K * t)
            delta_u_t = V_K @ (exp_factors * alpha0)
            u_t = u_ss + np.real(delta_u_t)
            rho_t = u_to_rho(u_t)

            # Enforce Hermiticity and unit trace softly
            rho_t = 0.5 * (rho_t + rho_t.conj().T)
            tr = np.trace(rho_t)
            rho_t /= tr

            D_HS_vals[k] = hs_distance_squared(rho_t, rho_ss)
            S_rel_vals[k] = quantum_relative_entropy(rho_t, rho_ss)

        # Late time window: last two thirds, avoid tiny values
        mask_HS = (t_grid >= T_max / 3.0) & (D_HS_vals > 1e-14)
        mask_rel = (t_grid >= T_max / 3.0) & (S_rel_vals > 1e-14)

        def fit_rate(t_vals, f_vals):
            if t_vals.size < 5:
                return np.nan
            logf = np.log(f_vals)
            coeffs = np.polyfit(t_vals, logf, 1)
            slope = coeffs[0]
            return -float(slope)

        r_HS = fit_rate(t_grid[mask_HS], D_HS_vals[mask_HS])
        r_rel = fit_rate(t_grid[mask_rel], S_rel_vals[mask_rel])

        return {
            "idx": idx_init,
            "r_HS": r_HS,
            "r_rel": r_rel,
        }

    # Run trajectories in parallel
    tasks = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        for idx in range(N_INIT):
            tasks.append(executor.submit(trajectory_decay_rates, idx, seeds[idx]))
        results = [f.result() for f in as_completed(tasks)]

    results_sorted = sorted(results, key=lambda d: d["idx"])
    r_HS_list = []
    r_rel_list = []
    for res in results_sorted:
        if np.isfinite(res["r_HS"]) and res["r_HS"] > 0:
            r_HS_list.append(res["r_HS"])
        if np.isfinite(res["r_rel"]) and res["r_rel"] > 0:
            r_rel_list.append(res["r_rel"])

    # Diagnostics for HS distance decay
    print("Empirical decay rates from HS distance squared D_HS(t):")
    if not r_HS_list:
        print("  No valid HS decay rates were extracted.")
        pass_HS = False
    else:
        r_HS = np.array(r_HS_list, dtype=float)
        r_HS_mean = float(np.mean(r_HS))
        r_HS_min = float(np.min(r_HS))
        r_HS_max = float(np.max(r_HS))

        rel_HS_mean = abs(r_HS_mean - r_pred_quantum) / max(r_pred_quantum, 1e-16)
        rel_HS_min = abs(r_HS_min - r_pred_quantum) / max(r_pred_quantum, 1e-16)
        rel_HS_max = abs(r_HS_max - r_pred_quantum) / max(r_pred_quantum, 1e-16)

        ratio_HS_mean = r_HS_mean / r_pred_quantum
        ratio_HS_min = r_HS_min / r_pred_quantum
        ratio_HS_max = r_HS_max / r_pred_quantum

        print(f"  Number of successful fits: {r_HS.size}/{N_INIT}")
        print(f"  r_HS mean ≈ {r_HS_mean:.6e}")
        print(f"  r_HS min  ≈ {r_HS_min:.6e}")
        print(f"  r_HS max  ≈ {r_HS_max:.6e}")
        print(f"  Mean relative error vs 2 lambda_K ≈ {rel_HS_mean:.3e}")
        print(f"  Min  relative error vs 2 lambda_K ≈ {rel_HS_min:.3e}")
        print(f"  Max  relative error vs 2 lambda_K ≈ {rel_HS_max:.3e}")
        print(f"  Ratios r_HS / (2 lambda_K): mean ≈ {ratio_HS_mean:.3e}, "
              f"min ≈ {ratio_HS_min:.3e}, max ≈ {ratio_HS_max:.3e}")
        print("")

        # HS tolerances relative to 2 lambda_K
        tol_mean_HS = 0.10    # 10 percent on the mean
        tol_min_HS = 0.05     # 5 percent on the best trajectory
        tol_extreme_HS = 0.25 # 25 percent allowed for extremes

        pass_HS = ((rel_HS_mean < tol_mean_HS) and
                   (rel_HS_min < tol_min_HS) and
                   (rel_HS_max < tol_extreme_HS))

    # Diagnostics for quantum relative entropy decay
    print("Empirical decay rates from quantum relative entropy S_rel(t):")
    if not r_rel_list:
        print("  No valid relative entropy decay rates were extracted.")
        pass_rel = False
    else        :
        r_rel_arr = np.array(r_rel_list, dtype=float)
        r_rel_mean = float(np.mean(r_rel_arr))
        r_rel_min = float(np.min(r_rel_arr))
        r_rel_max = float(np.max(r_rel_arr))

        # Ratios vs quantum gap and Markov gap
        ratio_rel_mean_K = r_rel_mean / r_pred_quantum
        ratio_rel_min_K = r_rel_min / r_pred_quantum
        ratio_rel_max_K = r_rel_max / r_pred_quantum

        ratio_rel_mean_Q = r_rel_mean / r_pred_markov
        ratio_rel_min_Q = r_rel_min / r_pred_markov
        ratio_rel_max_Q = r_rel_max / r_pred_markov

        print(f"  Number of successful fits: {r_rel_arr.size}/{N_INIT}")
        print(f"  r_rel mean ≈ {r_rel_mean:.6e}")
        print(f"  r_rel min  ≈ {r_rel_min:.6e}")
        print(f"  r_rel max  ≈ {r_rel_max:.6e}")
        print(f"  Ratios r_rel / (2 lambda_K): mean ≈ {ratio_rel_mean_K:.3e}, "
              f"min ≈ {ratio_rel_min_K:.3e}, max ≈ {ratio_rel_max_K:.3e}")
        print(f"  Ratios r_rel / (2 lambda_Q): mean ≈ {ratio_rel_mean_Q:.3e}, "
              f"min ≈ {ratio_rel_min_Q:.3e}, max ≈ {ratio_rel_max_Q:.3e}")
        print("")

        # Require that relative entropy decays not much slower than 2 lambda_K
        # and not faster than about 10 percent above 2 lambda_Q
        lower_factor = 0.7
        upper_factor = 1.1

        pass_rel = ((r_rel_min > lower_factor * r_pred_quantum) and
                    (r_rel_max < upper_factor * r_pred_markov))

    print("Summary of checks:")
    pass_stationary = norm_ss < 1e-10
    pass_Qpi = float(np.linalg.norm(Q_pi, ord=2)) < 1e-10

    print(f"  rho_ss stationary for GKLS?                         {pass_stationary} (tol = 1e-10)")
    print(f"  Q_markov pi ≈ 0 (correct stationary distribution)?   {pass_Qpi} (tol = 1e-10)")
    print(f"  HS decay rates consistent with 2 lambda_K?           {pass_HS}")
    print(f"  Relative entropy decay within [2 lambda_K, 2 lambda_Q]? {pass_rel}")
    print("")

    all_pass = pass_stationary and pass_Qpi and pass_HS and pass_rel

    if all_pass:
        print("Qutrit full quantum entropy decay vs gaps CHECK: PASS")
        print("  The full GKLS evolution on 3x3 density matrices relaxes to")
        print("  the Gibbs state rho_ss with both the Hilbert Schmidt distance")
        print("  and the quantum relative entropy decaying at rates set by the")
        print("  GKLS spectral gap lambda_K and bounded above by the classical")
        print("  Markov gap lambda_Q extracted from the population sector.")
        print("  Coherent initial states do not change the asymptotic entropy")
        print("  clock: the quantum gap from K_total is the fundamental rate,")
        print("  while the density sector gap provides an upper bound.")
    else:
        print("Qutrit full quantum entropy decay vs gaps CHECK: FAIL")
        print("  At least one of the conditions (GKLS stationarity, Markov")
        print("  invariance, or entropy decay vs the two gaps) did not meet the")
        print("  specified tolerances. See diagnostics above.")
        print("")


if __name__ == "__main__":
    run()
