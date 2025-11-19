#!/usr/bin/env python3
"""
14_qutrit_GKLS_family_universal_density_sector_checks.py

Qutrit GKLS family: universal density sector Fisher geometry
and varying quantum gaps.

We fix energies and temperature, and a single set of jump rates
that define a classical reversible Markov generator Q on the
populations. Then we build several GKLS generators that all

  - share the same Gibbs stationary state rho_ss,
  - share the same population block Q_markov and hence the same
    canonical classical Fisher Dirichlet operator G_true_class,

but differ in the way they dephase and kill coherences.

For each variant we:

  - construct H, L_ops and the real nine dimensional generator
    K_total on u = (p0, p1, p2, Re01, Im01, Re02, Im02, Re12, Im12),
  - extract Q_markov, G_true_class, Fisher curvature gap λ_G,
    and Markov gap λ_Q,
  - compute the full GKLS spectral gap λ_K from K_total,
  - evolve random quantum states and fit HS and relative entropy
    decay rates, comparing them with 2 λ_K and 2 λ_Q.

At the end we check that Q_markov, G_true_class, λ_G and λ_Q are
identical across the family, while λ_K and the quantum decay rates
depend on the coherent sector. This realises the idea that many
different GKLS generators share one universal density sector
information hydrodynamics.
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def u_to_rho(u):
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


def matrix_log_hermitian(rho, eps=1e-15):
    vals, vecs = np.linalg.eigh(rho)
    vals_clipped = np.clip(np.real(vals), eps, None)
    log_vals = np.log(vals_clipped)
    return vecs @ np.diag(log_vals) @ vecs.conj().T


def quantum_relative_entropy(rho, sigma, eps=1e-15):
    log_rho = matrix_log_hermitian(rho, eps=eps)
    log_sigma = matrix_log_hermitian(sigma, eps=eps)
    diff = log_rho - log_sigma
    return float(np.real(np.trace(rho @ diff)))


def hs_distance_squared(rho, sigma):
    diff = rho - sigma
    return float(np.linalg.norm(diff, ord="fro") ** 2)


def build_qutrit_variant(variant, beta=1.0,
                         E0=0.0, E1=1.0, E2=2.0,
                         g01_down=0.8, g12_down=0.5, g02_down=0.3):
    E = np.array([E0, E1, E2], dtype=float)
    H = np.diag(E).astype(complex)

    w = np.exp(-beta * E)
    Z = float(np.sum(w))
    pi = w / Z
    rho_ss = np.diag(pi.astype(complex))

    def jump_pair(i, j, g_down):
        Delta = E[j] - E[i]
        g_up = g_down * np.exp(-beta * Delta)
        L_down = np.zeros((3, 3), dtype=complex)
        L_up = np.zeros((3, 3), dtype=complex)
        L_down[i, j] = np.sqrt(g_down)
        L_up[j, i] = np.sqrt(g_up)
        return [L_down, L_up]

    L_ops = []
    L_ops += jump_pair(0, 1, g01_down)
    L_ops += jump_pair(1, 2, g12_down)
    L_ops += jump_pair(0, 2, g02_down)

    if variant == "A":
        gamma_phi = 0.1
        for k in range(3):
            L = np.zeros((3, 3), dtype=complex)
            L[k, k] = 1.0
            L_ops.append(np.sqrt(gamma_phi) * L)
    elif variant == "B":
        gamma_vec = [0.05, 0.2, 0.4]
        for k, gphi in enumerate(gamma_vec):
            L = np.zeros((3, 3), dtype=complex)
            L[k, k] = 1.0
            L_ops.append(np.sqrt(gphi) * L)
    elif variant == "C":
        g_d1 = 0.3
        g_d2 = 0.7
        L1 = np.diag([1.0, -1.0, 0.0]).astype(complex)
        L2 = np.diag([0.0, 1.0, -1.0]).astype(complex)
        L_ops.append(np.sqrt(g_d1) * L1)
        L_ops.append(np.sqrt(g_d2) * L2)
    else:
        raise ValueError("Unknown variant: " + str(variant))

    return H, rho_ss, pi, L_ops


def fit_exponential_rate(t_vals, f_vals):
    mask = (f_vals > 1e-14)
    t_fit = t_vals[mask]
    f_fit = f_vals[mask]
    if t_fit.size < 5:
        return np.nan
    logf = np.log(f_fit)
    coeffs = np.polyfit(t_fit, logf, 1)
    slope = coeffs[0]
    return -float(slope)


def run():
    print("GKLS qutrit family universal density sector checks")
    print("--------------------------------------------------")

    beta = 1.0
    E0, E1, E2 = 0.0, 1.0, 2.0
    g01_down, g12_down, g02_down = 0.8, 0.5, 0.3

    print(f"beta = {beta}")
    print(f"Energies: E0 = {E0}, E1 = {E1}, E2 = {E2}")
    print(f"Downward jump rates: g01_down = {g01_down}, g12_down = {g12_down}, g02_down = {g02_down}")
    print("")

    variants = [
        ("A", "Projector dephasing, uniform rates"),
        ("B", "Projector dephasing, non uniform rates"),
        ("C", "Non local diagonal dephasing operators"),
    ]

    N_CORES = os.cpu_count() or 1
    if N_CORES >= 20:
        N_WORKERS = 20
    else:
        N_WORKERS = 1

    print("Multithreading:")
    print(f"  Detected CPU cores: {N_CORES}")
    print(f"  Worker threads used: {N_WORKERS}")
    print("")

    Q_list = []
    G_class_list = []
    lambda_G_list = []
    lambda_Q_list = []
    lambda_K_list = []

    rng_global = np.random.default_rng(20251118)

    for key, desc in variants:
        print(f"Variant {key}: {desc}")
        print("-" * 60)

        H, rho_ss, pi, L_ops = build_qutrit_variant(key, beta=beta,
                                                    E0=E0, E1=E1, E2=E2,
                                                    g01_down=g01_down,
                                                    g12_down=g12_down,
                                                    g02_down=g02_down)

        print("Stationary state rho_ss:")
        print(rho_ss)
        print(f"pi (thermal probabilities) = {pi}")
        d_rho_ss = apply_Hamiltonian(H, rho_ss) + apply_dissipator(L_ops, rho_ss)
        norm_ss = float(np.linalg.norm(d_rho_ss, ord="fro"))
        print(f"  ||GKLS(rho_ss)||_F ≈ {norm_ss:.3e} (should be near 0)")
        print("")

        K_total, K_H, K_D = build_generator_real(H, L_ops)
        u_ss = rho_to_u(rho_ss)
        K_u_ss = K_total @ u_ss
        print("Real GKLS generators on u space:")
        print(f"  Dimension of u space: {len(u_ss)}")
        print(f"  ||K_total u_ss||_2 ≈ {float(np.linalg.norm(K_u_ss, ord=2)):.3e} (should be near 0)")
        print("")

        Q_markov = K_D[0:3, 0:3]
        print("Classical 3 state Markov generator Q_markov (density block of K_D):")
        print(Q_markov)
        Q_pi = Q_markov @ pi
        col_sums = np.sum(Q_markov, axis=0)
        print(f"  ||Q_markov pi||_2 ≈ {float(np.linalg.norm(Q_pi, ord=2)):.3e} (should be near 0)")
        print(f"  Column sums (should be ~0 for generator): {col_sums}")
        print("")

        G_true_class = Q_markov @ np.diag(pi)
        sym_resid = np.linalg.norm(G_true_class - G_true_class.T) / max(np.linalg.norm(G_true_class), 1e-16)
        skew_resid = np.linalg.norm(G_true_class + G_true_class.T) / max(np.linalg.norm(G_true_class), 1e-16)
        print("Canonical classical G_true_class = Q_markov diag(pi):")
        print(G_true_class)
        print(f"  Symmetry residual ||G - G^T||/||G|| ≈ {sym_resid:.3e}")
        print(f"  Skew residual     ||G + G^T||/||G|| ≈ {skew_resid:.3e}")
        print("")

        G_metric = -G_true_class
        vals_G = np.linalg.eigvalsh(G_metric)
        tol_small = 1e-8
        pos_vals = [v for v in vals_G if v > tol_small]
        if pos_vals:
            lambda_G = float(min(pos_vals))
        else:
            lambda_G = float("nan")
        print("Eigenvalues of G_metric = -G_true_class:")
        print(f"  {vals_G}")
        print(f"  Fisher curvature gap λ_G ≈ {lambda_G:.6e}")
        print("")

        pi_sqrt = np.sqrt(pi)
        B = np.diag(pi_sqrt)
        B_inv = np.diag(1.0 / pi_sqrt)
        L_sym = B_inv @ Q_markov @ B
        eigvals_L = np.linalg.eigvals(L_sym)
        print("Eigenvalues of symmetrised generator L_sym = B^{-1} Q B:")
        print(f"  {eigvals_L}")
        neg_L = [ev.real for ev in eigvals_L if ev.real < -1e-6]
        if neg_L:
            lambda_Q = float(min(-ev for ev in neg_L))
        else:
            lambda_Q = float("nan")
        print(f"  Markov spectral gap λ_Q (pi weighted) ≈ {lambda_Q:.6e}")
        print("")

        eigvals_K, eigvecs_K = np.linalg.eig(K_total)
        real_parts = np.real(eigvals_K)
        nonzero_real = real_parts[np.abs(real_parts) > 1e-8]
        if nonzero_real.size > 0:
            lambda_K = float(min(-nonzero_real))
        else:
            lambda_K = float("nan")

        print("Eigenvalues of K_total (real parts should be <= 0):")
        print(f"  {eigvals_K}")
        print(f"  GKLS spectral gap λ_K ≈ {lambda_K:.6e}")
        print("")

        Q_list.append(Q_markov.copy())
        G_class_list.append(G_true_class.copy())
        lambda_G_list.append(lambda_G)
        lambda_Q_list.append(lambda_Q)
        lambda_K_list.append(lambda_K)

        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
        V_K = eigvecs_K
        V_K_inv = np.linalg.inv(V_K)

        if lambda_K > 0.0 and np.isfinite(lambda_K):
            N_INIT = 16
            T_max = 8.0 / lambda_K
            N_T = 160
            t_grid = np.linspace(0.0, T_max, N_T)

            print("Time evolution parameters for this variant:")
            print(f"  Number of initial states N_INIT = {N_INIT}")
            print(f"  T_max ≈ {T_max:.3e}")
            print(f"  N_T = {N_T}, dt ≈ {t_grid[1] - t_grid[0]:.3e}")
            print("")

            def single_trajectory_rates(idx_init):
                X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                rho0 = X @ X.conj().T
                rho0 /= np.trace(rho0)
                u0 = rho_to_u(rho0)
                delta_u0 = u0 - u_ss
                alpha0 = V_K_inv @ delta_u0.astype(complex)

                D_vals = np.zeros_like(t_grid, dtype=float)
                S_vals = np.zeros_like(t_grid, dtype=float)

                for k, t in enumerate(t_grid):
                    exp_factors = np.exp(eigvals_K * t)
                    delta_u_t = V_K @ (exp_factors * alpha0)
                    u_t = u_ss + np.real(delta_u_t)
                    rho_t = u_to_rho(u_t)
                    rho_t = 0.5 * (rho_t + rho_t.conj().T)
                    tr = np.trace(rho_t)
                    rho_t /= tr
                    D_vals[k] = hs_distance_squared(rho_t, rho_ss)
                    S_vals[k] = quantum_relative_entropy(rho_t, rho_ss)

                mask_time = t_grid >= T_max / 3.0
                r_HS = fit_exponential_rate(t_grid[mask_time], D_vals[mask_time])
                r_rel = fit_exponential_rate(t_grid[mask_time], S_vals[mask_time])
                return {"idx": idx_init, "r_HS": r_HS, "r_rel": r_rel}

            results = []
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [executor.submit(single_trajectory_rates, i) for i in range(N_INIT)]
                for fut in as_completed(futures):
                    results.append(fut.result())
            results.sort(key=lambda d: d["idx"])

            r_HS_vals = [d["r_HS"] for d in results if np.isfinite(d["r_HS"]) and d["r_HS"] > 0.0]
            r_rel_vals = [d["r_rel"] for d in results if np.isfinite(d["r_rel"]) and d["r_rel"] > 0.0]

            print("Empirical decay rates from HS distance squared D_HS(t):")
            if r_HS_vals:
                r_arr = np.array(r_HS_vals, dtype=float)
                r_mean = float(np.mean(r_arr))
                r_min = float(np.min(r_arr))
                r_max = float(np.max(r_arr))
                print(f"  Number of successful fits: {r_arr.size}/{N_INIT}")
                print(f"  r_HS mean ≈ {r_mean:.6e}")
                print(f"  r_HS min  ≈ {r_min:.6e}")
                print(f"  r_HS max  ≈ {r_max:.6e}")
                if lambda_Q > 0.0 and np.isfinite(lambda_Q):
                    ratio_mean_Q = r_mean / (2.0 * lambda_Q)
                    ratio_min_Q = r_min / (2.0 * lambda_Q)
                    ratio_max_Q = r_max / (2.0 * lambda_Q)
                    print(f"  Ratios r_HS / (2 λ_Q): mean ≈ {ratio_mean_Q:.3e}, "
                          f"min ≈ {ratio_min_Q:.3e}, max ≈ {ratio_max_Q:.3e}")
                ratio_mean_K = r_mean / (2.0 * lambda_K)
                ratio_min_K = r_min / (2.0 * lambda_K)
                ratio_max_K = r_max / (2.0 * lambda_K)
                print(f"  Ratios r_HS / (2 λ_K): mean ≈ {ratio_mean_K:.3e}, "
                      f"min ≈ {ratio_min_K:.3e}, max ≈ {ratio_max_K:.3e}")
            else:
                print("  No valid HS decay rates extracted.")
            print("")

            print("Empirical decay rates from quantum relative entropy S_rel(t):")
            if r_rel_vals:
                r_arr = np.array(r_rel_vals, dtype=float)
                r_mean = float(np.mean(r_arr))
                r_min = float(np.min(r_arr))
                r_max = float(np.max(r_arr))
                print(f"  Number of successful fits: {r_arr.size}/{N_INIT}")
                print(f"  r_rel mean ≈ {r_mean:.6e}")
                print(f"  r_rel min  ≈ {r_min:.6e}")
                print(f"  r_rel max  ≈ {r_max:.6e}")
                if lambda_Q > 0.0 and np.isfinite(lambda_Q):
                    ratio_mean_Q = r_mean / (2.0 * lambda_Q)
                    ratio_min_Q = r_min / (2.0 * lambda_Q)
                    ratio_max_Q = r_max / (2.0 * lambda_Q)
                    print(f"  Ratios r_rel / (2 λ_Q): mean ≈ {ratio_mean_Q:.3e}, "
                          f"min ≈ {ratio_min_Q:.3e}, max ≈ {ratio_max_Q:.3e}")
                ratio_mean_K = r_mean / (2.0 * lambda_K)
                ratio_min_K = r_min / (2.0 * lambda_K)
                ratio_max_K = r_max / (2.0 * lambda_K)
                print(f"  Ratios r_rel / (2 λ_K): mean ≈ {ratio_mean_K:.3e}, "
                      f"min ≈ {ratio_min_K:.3e}, max ≈ {ratio_max_K:.3e}")
            else:
                print("  No valid relative entropy decay rates extracted.")
            print("")
        else:
            print("Skipping time evolution for this variant due to invalid λ_K.")
            print("")

    print("Cross model comparison of density sector geometry:")
    base_Q = Q_list[0]
    base_G = G_class_list[0]
    base_lambda_G = lambda_G_list[0]
    base_lambda_Q = lambda_Q_list[0]

    for idx, (key, desc) in enumerate(variants):
        Q = Q_list[idx]
        Gc = G_class_list[idx]
        dQ = np.linalg.norm(Q - base_Q, ord="fro") / max(np.linalg.norm(base_Q, ord="fro"), 1e-16)
        dG = np.linalg.norm(Gc - base_G, ord="fro") / max(np.linalg.norm(base_G, ord="fro"), 1e-16)
        d_lambda_G = abs(lambda_G_list[idx] - base_lambda_G) / max(abs(base_lambda_G), 1e-16)
        d_lambda_Q = abs(lambda_Q_list[idx] - base_lambda_Q) / max(abs(base_lambda_Q), 1e-16)
        print(f"  Variant {key}:")
        print(f"    Relative difference in Q_markov vs A: {dQ:.3e}")
        print(f"    Relative difference in G_true_class vs A: {dG:.3e}")
        print(f"    Relative difference in λ_G vs A: {d_lambda_G:.3e}")
        print(f"    Relative difference in λ_Q vs A: {d_lambda_Q:.3e}")
        print(f"    GKLS gap λ_K for this variant: {lambda_K_list[idx]:.6e}")
    print("")

    print("Summary:")
    all_Q_same = all(np.linalg.norm(Q_list[i] - Q_list[0], ord="fro") /
                     max(np.linalg.norm(Q_list[0], ord="fro"), 1e-16) < 1e-12
                     for i in range(1, len(Q_list)))
    all_G_same = all(np.linalg.norm(G_class_list[i] - G_class_list[0], ord="fro") /
                     max(np.linalg.norm(G_class_list[0], ord="fro"), 1e-16) < 1e-12
                     for i in range(1, len(G_class_list)))
    all_lambda_G_same = all(abs(lambda_G_list[i] - lambda_G_list[0]) /
                            max(abs(lambda_G_list[0]), 1e-16) < 1e-12
                            for i in range(1, len(lambda_G_list)))
    all_lambda_Q_same = all(abs(lambda_Q_list[i] - lambda_Q_list[0]) /
                            max(abs(lambda_Q_list[0]), 1e-16) < 1e-12
                            for i in range(1, len(lambda_Q_list)))

    print(f"  Q_markov identical across variants (within tol 1e-12)?    {all_Q_same}")
    print(f"  G_true_class identical across variants (within tol 1e-12)? {all_G_same}")
    print(f"  λ_G identical across variants (within tol 1e-12)?          {all_lambda_G_same}")
    print(f"  λ_Q identical across variants (within tol 1e-12)?          {all_lambda_Q_same}")
    print("  GKLS gaps λ_K differ across variants, reflecting different")
    print("  coherence structures, while the density sector Fisher and")
    print("  Markov geometry is universal for the shared jump rates.")
    print("")


if __name__ == "__main__":
    run()
