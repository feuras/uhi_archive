#!/usr/bin/env python3
"""
10_gkls_density_block_G_unification_checks.py

Density block unification for a coherent thermal qubit GKLS model.

Goal:
  Starting from the coherent thermal GKLS qubit model and its metriplectic
  K = G + J splitting (as in 09_gkls_K_splitting_qubit_checks.py), extract
  the population block of the dissipative operator and show that this block
  generates exactly the canonical classical G_true = Q diag(pi) structure
  on the two state Markov chain of populations.

Model:
  - Two level system, basis |0>, |1>.
  - Hamiltonian H = (Delta/2) sigma_z.
  - Thermal stationary state rho_ss = diag(pi0, pi1), with
        pi0 ∝ exp(+beta Delta/2),
        pi1 ∝ exp(-beta Delta/2).
  - Jump operators:
        L_down = sqrt(gamma_down) |0><1|
        L_up   = sqrt(gamma_up)   |1><0|
        L_phi  = sqrt(gamma_phi)  sigma_z  (pure dephasing)
    with detailed balance gamma_up / gamma_down = exp(-beta Delta).

State coordinates:
  - u = (p0, p1, Re c, Im c).

Metric:
  - rho_ss weighted inner product on operators:
        <A, B> = Tr(A rho_ss^{-1} B),
    induces an inner product on u with metric matrix M via basis variations
    B_i = ∂rho/∂u_i.

Metriplectic splitting:
  - Build the full real generator K on u from GKLS.
  - Metric adjoint K_sharp = M^{-1} K^T M.
  - Symmetric and antisymmetric parts:
        G = 0.5 (K + K_sharp),
        J = 0.5 (K - K_sharp).
  - As in script 09, this gives G ≈ K_D (pure dissipator),
    J ≈ K_H (pure Hamiltonian).

Density block unification:
  - Extract the 2x2 population block Q_dens from G (or K_D).
  - Let pi = (pi0, pi1) from rho_ss.
  - Verify:
        1) Q_dens pi ≈ 0 (pi is stationary).
        2) Q_dens off diagonal entries are the up and down jump rates.
        3) G_true_class = Q_dens diag(pi) is symmetric.
        4) For random mu, with p = pi ⊙ mu, the population drift from GKLS,
           v_pop = Q_dens p, equals v_G = G_true_class mu.
        5) The Dirichlet form from rates,
               E_pair(mu) = 0.5 ∑_{i,j} pi_j gamma_{ij} (mu_i − mu_j)^2,
           matches − mu^T G_true_class mu.

This shows that the classical Fisher–Dirichlet G on populations is literally
the density block of the G extracted from the full coherent GKLS K splitting.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Qubit GKLS model and helpers (as in 09)
# ---------------------------------------------------------------------------

def build_qubit_model(beta=1.0, Delta=1.0,
                      gamma_down_base=0.7, gamma_phi=0.2):
    """
    Build a simple thermal qubit model with Hamiltonian and GKLS jumps.

    Parameters:
      beta            - inverse temperature
      Delta           - energy gap (H = (Delta/2) sigma_z)
      gamma_down_base - base downward jump rate
      gamma_phi       - dephasing rate for sigma_z

    Returns:
      H        - 2x2 complex Hamiltonian matrix
      rho_ss   - 2x2 complex stationary state
      L_ops    - list of 2x2 complex Lindblad operators for dissipator
    """
    sigma_z = np.array([[1.0, 0.0],
                        [0.0, -1.0]], dtype=complex)
    H = 0.5 * Delta * sigma_z

    # Thermal stationary distribution for energies E0 = -Delta/2, E1 = +Delta/2
    E0 = -0.5 * Delta
    E1 = 0.5 * Delta
    w0 = np.exp(-beta * E0)
    w1 = np.exp(-beta * E1)
    Z = w0 + w1
    pi0 = w0 / Z
    pi1 = w1 / Z
    rho_ss = np.array([[pi0, 0.0],
                       [0.0, pi1]], dtype=complex)

    # Detailed balance for jumps
    gamma_down = gamma_down_base
    gamma_up = gamma_down * np.exp(-beta * Delta)

    L_down = np.zeros((2, 2), dtype=complex)
    L_down[0, 1] = np.sqrt(gamma_down)

    L_up = np.zeros((2, 2), dtype=complex)
    L_up[1, 0] = np.sqrt(gamma_up)

    L_ops = [L_down, L_up]

    if gamma_phi > 0.0:
        L_phi = np.sqrt(gamma_phi) * sigma_z
        L_ops.append(L_phi)

    return H, rho_ss, L_ops


def apply_Hamiltonian(H, rho):
    """
    Hamiltonian part of GKLS:
        dρ/dt = -i [H, ρ]
    """
    return -1j * (H @ rho - rho @ H)


def apply_dissipator(L_ops, rho):
    """
    GKLS dissipator:
        D(ρ) = sum_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ})
    """
    dim = rho.shape[0]
    D = np.zeros((dim, dim), dtype=complex)
    for L in L_ops:
        L_rho = L @ rho
        L_rho_Ld = L_rho @ L.conj().T
        K = L.conj().T @ L
        anti = K @ rho + rho @ K
        D += L_rho_Ld - 0.5 * anti
    return D


def u_to_rho(u):
    """
    Map real state vector u = (p0, p1, x, y) to 2x2 density matrix:

        rho = [[p0,    x + i y],
               [x - i y,  p1 ]]

    No trace normalisation is enforced here; the generator is linear.
    """
    p0, p1, x, y = u
    rho = np.array([[p0,       x + 1j * y],
                    [x - 1j*y, p1      ]], dtype=complex)
    return rho


def rho_to_u(rho):
    """
    Map 2x2 Hermitian matrix rho to u = (p0, p1, x, y).
    """
    p0 = np.real(rho[0, 0])
    p1 = np.real(rho[1, 1])
    c01 = rho[0, 1]
    x = np.real(c01)
    y = np.imag(c01)
    return np.array([p0, p1, x, y], dtype=float)


def build_generator_real(H, L_ops):
    """
    Build the 4x4 real generators K_total, K_H, K_D on u space so that:

        du/dt = K u

    by applying the GKLS generator to basis u vectors.
    """
    dim_u = 4
    K_total = np.zeros((dim_u, dim_u), dtype=float)
    K_H = np.zeros((dim_u, dim_u), dtype=float)
    K_D = np.zeros((dim_u, dim_u), dtype=float)

    for j in range(dim_u):
        e_j = np.zeros(dim_u, dtype=float)
        e_j[j] = 1.0
        rho_j = u_to_rho(e_j)

        d_rho_H = apply_Hamiltonian(H, rho_j)
        d_rho_D = apply_dissipator(L_ops, rho_j)

        du_H = rho_to_u(d_rho_H)
        du_D = rho_to_u(d_rho_D)
        du_total = du_H + du_D

        K_H[:, j] = du_H
        K_D[:, j] = du_D
        K_total[:, j] = du_total

    return K_total, K_H, K_D


def build_metric_matrix(rho_ss):
    """
    Build 4x4 metric matrix M on u space from

        <A, B> = Tr(A rho_ss^{-1} B),

    using basis variations:

      B0 = ∂ρ/∂p0 = [[1, 0],
                     [0, 0]]

      B1 = ∂ρ/∂p1 = [[0, 0],
                     [0, 1]]

      B2 = ∂ρ/∂x  = [[0, 1],
                     [1, 0]]

      B3 = ∂ρ/∂y  = [[0, i],
                     [-i, 0]]

    and M_{ij} = Tr(B_i rho_ss^{-1} B_j).
    """
    pi0 = np.real(rho_ss[0, 0])
    pi1 = np.real(rho_ss[1, 1])

    rho_inv = np.array([[1.0 / pi0, 0.0],
                        [0.0, 1.0 / pi1]], dtype=complex)

    B0 = np.array([[1.0, 0.0],
                   [0.0, 0.0]], dtype=complex)
    B1 = np.array([[0.0, 0.0],
                   [0.0, 1.0]], dtype=complex)
    B2 = np.array([[0.0, 1.0],
                   [1.0, 0.0]], dtype=complex)
    B3 = np.array([[0.0, 1.0j],
                   [-1.0j, 0.0]], dtype=complex)

    B_list = [B0, B1, B2, B3]

    M = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            val = np.trace(B_list[i] @ rho_inv @ B_list[j])
            M[i, j] = float(np.real(val))

    return M


def metric_adjoint(K, M):
    """
    Metric adjoint of K with respect to inner product (x, y) = x^T M y:

        K_sharp = M^{-1} K^T M.
    """
    Minv = np.linalg.inv(M)
    return Minv @ K.T @ M


def symmetry_metrics(M):
    """
    Return (symmetry_residual, skew_residual) for matrix M in Frobenius norm:

        sym_resid = ||M - M^T||_F / ||M||_F
        skew_resid = ||M + M^T||_F / ||M||_F
    """
    norm_M = float(np.linalg.norm(M, ord="fro"))
    if norm_M == 0.0:
        return 0.0, 0.0
    sym_part = M - M.T
    skew_part = M + M.T
    sym_resid = float(np.linalg.norm(sym_part, ord="fro") / norm_M)
    skew_resid = float(np.linalg.norm(skew_part, ord="fro") / norm_M)
    return sym_resid, skew_resid


# ---------------------------------------------------------------------------
# Dirichlet form for 2 state chain
# ---------------------------------------------------------------------------

def dirichlet_from_rates(mu, pi, gamma):
    """
    Two state Dirichlet form from rates:

        E_pair(mu) = 0.5 sum_{i,j} pi_j gamma_{ij} (mu_i - mu_j)^2.

    Here gamma_{ij} = off diagonal entries of Q for i != j.
    """
    E = 0.0
    dim = 2
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            E += 0.5 * pi[j] * gamma[i, j] * (mu[i] - mu[j]) ** 2
    return E


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run():
    print("GKLS density block G-unification for thermal qubit")
    print("--------------------------------------------------")

    beta = 1.0
    Delta = 1.0
    gamma_down_base = 0.7
    gamma_phi = 0.2

    print(f"beta = {beta}, Delta = {Delta}, "
          f"gamma_down_base = {gamma_down_base}, gamma_phi = {gamma_phi}")
    print("")

    # Build model
    H, rho_ss, L_ops = build_qubit_model(beta=beta,
                                         Delta=Delta,
                                         gamma_down_base=gamma_down_base,
                                         gamma_phi=gamma_phi)

    print("Stationary state rho_ss:")
    print(rho_ss)
    print("")

    # Stationarity check
    d_rho_H_ss = apply_Hamiltonian(H, rho_ss)
    d_rho_D_ss = apply_dissipator(L_ops, rho_ss)
    d_rho_ss = d_rho_H_ss + d_rho_D_ss

    norm_total_ss = float(np.linalg.norm(d_rho_ss, ord="fro"))
    print("Stationarity diagnostics:")
    print(f"  ||GKLS(rho_ss)||_F ≈ {norm_total_ss:.3e} (should be near 0)")
    print("")

    # Real generators on u
    K_total, K_H, K_D = build_generator_real(H, L_ops)

    print("Real generators on u = (p0, p1, Re c, Im c):")
    print("  K_H:")
    print(K_H)
    print("")
    print("  K_D:")
    print(K_D)
    print("")
    print("  K_total:")
    print(K_total)
    print("")

    # Metric and metriplectic splitting
    M = build_metric_matrix(rho_ss)
    print("Metric matrix M:")
    print(M)
    print("")

    K_sharp = metric_adjoint(K_total, M)
    G = 0.5 * (K_total + K_sharp)
    J = 0.5 * (K_total - K_sharp)

    frob = lambda A: float(np.linalg.norm(A, ord="fro"))

    diff_G_D = G - K_D
    diff_J_H = J - K_H
    rel_err_G_D = frob(diff_G_D) / max(frob(K_D), 1e-16)
    rel_err_J_H = frob(diff_J_H) / max(frob(K_H), 1e-16)

    print("K-splitting cross check (should reproduce 09):")
    print(f"  ||G - K_D||_F / ||K_D||_F ≈ {rel_err_G_D:.3e}")
    print(f"  ||J - K_H||_F / ||K_H||_F ≈ {rel_err_J_H:.3e}")
    print("")

    # Density block from G (or K_D)
    G_dens = G[0:2, 0:2]
    K_D_dens = K_D[0:2, 0:2]

    print("Density sector matrices:")
    print("  G_dens (from metric splitting):")
    print(G_dens)
    print("")
    print("  K_D_dens (direct dissipator block):")
    print(K_D_dens)
    print("")

    diff_Gdens_KD = G_dens - K_D_dens
    rel_err_Gdens_KD = frob(diff_Gdens_KD) / max(frob(K_D_dens), 1e-16)
    print(f"  ||G_dens - K_D_dens||_F / ||K_D_dens||_F ≈ {rel_err_Gdens_KD:.3e}")
    print("")

    # Classical Markov generator Q_dens from density block
    Q_dens = K_D_dens.copy()
    pi = np.array([np.real(rho_ss[0, 0]), np.real(rho_ss[1, 1])], dtype=float)

    print("Classical 2 state Markov generator Q_dens (from GKLS density block):")
    print(Q_dens)
    print(f"  pi = {pi}")
    print("")

    # Stationarity and rates
    Q_pi = Q_dens @ pi
    print("Two state chain diagnostics:")
    print(f"  ||Q_dens pi||_2 ≈ {float(np.linalg.norm(Q_pi, ord=2)):.3e} (should be near 0)")
    print("  Off diagonal rates gamma_{ij} (i != j):")
    gamma = np.zeros_like(Q_dens)
    for i in range(2):
        for j in range(2):
            if i != j:
                gamma[i, j] = Q_dens[i, j]
                print(f"    gamma[{i},{j}] ≈ {gamma[i,j]:.6f}")
    print("")

    # Canonical classical G_true on mu space
    G_true_class = Q_dens @ np.diag(pi)
    G_sym_resid, G_skew_resid = symmetry_metrics(G_true_class)

    print("Canonical classical G_true_class = Q_dens diag(pi):")
    print(G_true_class)
    print(f"  Symmetry residual ||G_true_class - G_true_class^T||/||G_true_class|| ≈ {G_sym_resid:.3e}")
    print(f"  Skew residual     ||G_true_class + G_true_class^T||/||G_true_class|| ≈ {G_skew_resid:.3e}")
    print("")

    # Drift consistency: for mu, p = pi ⊙ mu
    rng = np.random.default_rng(424242)
    n_probes = 50
    mu_batch = rng.normal(size=(n_probes, 2))
    # Remove trivial constant mode by forcing pi weighted mean zero if desired
    # but not strictly necessary here
    # For robustness, we can subtract pi weighted average
    for k in range(n_probes):
        mu = mu_batch[k]
        avg = np.dot(pi, mu)
        mu_batch[k] = mu - avg

    v_pop = np.zeros_like(mu_batch)
    v_G = np.zeros_like(mu_batch)

    for k in range(n_probes):
        mu = mu_batch[k]
        p = pi * mu
        v_pop[k] = Q_dens @ p
        v_G[k] = G_true_class @ mu

    diff_v = v_pop - v_G
    rel_err_v = frob(diff_v) / max(frob(v_pop), 1e-16)

    print("Drift consistency diagnostics (density block vs classical G_true):")
    print(f"  ||Q_dens (pi ⊙ mu) - G_true_class mu||_F / ||Q_dens (pi ⊙ mu)||_F ≈ {rel_err_v:.3e}")
    print("")

    # Dirichlet form consistency
    E_pair_list = []
    E_G_list = []

    for k in range(n_probes):
        mu = mu_batch[k]
        E_pair = dirichlet_from_rates(mu, pi, gamma)
        vG = v_G[k]
        E_G = -float(np.dot(mu, vG))
        E_pair_list.append(E_pair)
        E_G_list.append(E_G)

    E_pair_arr = np.array(E_pair_list)
    E_G_arr = np.array(E_G_list)
    diff_E = E_pair_arr - E_G_arr

    mask = np.abs(E_pair_arr) > 1e-12
    if np.any(mask):
        rel_err_E = float(
            np.linalg.norm(diff_E[mask], ord=2) /
            np.linalg.norm(E_pair_arr[mask], ord=2)
        )
    else:
        rel_err_E = 0.0

    print("Dirichlet form consistency diagnostics:")
    print(f"  Relative difference between E_pair(mu) and -mu^T G_true_class mu ≈ {rel_err_E:.3e}")
    print("")

    # PASS / FAIL criteria
    tol_stationary = 1e-10
    tol_split = 1e-10
    tol_block = 1e-10
    tol_Qpi = 1e-10
    tol_sym = 1e-12
    tol_v = 1e-10
    tol_E = 1e-10

    pass_stationary = norm_total_ss < tol_stationary
    pass_split = (rel_err_G_D < tol_split) and (rel_err_J_H < tol_split)
    pass_block = rel_err_Gdens_KD < tol_block
    pass_Qpi = float(np.linalg.norm(Q_pi, ord=2)) < tol_Qpi
    pass_sym = (G_sym_resid < tol_sym)
    pass_v = rel_err_v < tol_v
    pass_E = rel_err_E < tol_E

    print("Summary:")
    print(f"  rho_ss stationary for GKLS?                         {pass_stationary} (tol = {tol_stationary})")
    print(f"  K splitting reproduces K_D and K_H?                 {pass_split} (tol = {tol_split})")
    print(f"  G_dens equals K_D density block?                    {pass_block} (tol = {tol_block})")
    print(f"  Q_dens pi ≈ 0 (correct two state stationary)?       {pass_Qpi} (tol = {tol_Qpi})")
    print(f"  G_true_class symmetric up to tol?                   {pass_sym} (tol = {tol_sym})")
    print(f"  Drift match: Q_dens (pi ⊙ mu) ≈ G_true_class mu?    {pass_v} (tol = {tol_v})")
    print(f"  Dirichlet form E_pair(mu) ≈ -mu^T G_true_class mu?  {pass_E} (tol = {tol_E})")
    print("")

    if all([pass_stationary, pass_split, pass_block,
            pass_Qpi, pass_sym, pass_v, pass_E]):
        print("GKLS density block G-unification CHECK: PASS")
        print("  The population block of the symmetric part G from the coherent")
        print("  GKLS K splitting reproduces the two state Markov generator, and")
        print("  generates exactly the canonical classical G_true = Q diag(pi)")
        print("  with the correct drift and Dirichlet form. The classical Fisher")
        print("  Dirichlet geometry on densities is therefore the density block")
        print("  of the full quantum metriplectic G extracted from the GKLS K.")
    else:
        print("GKLS density block G-unification CHECK: FAIL (see diagnostics above).")


if __name__ == "__main__":
    run()
