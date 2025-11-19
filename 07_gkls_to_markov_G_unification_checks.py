#!/usr/bin/env python3
"""
07_gkls_to_markov_G_unification_checks.py

Canonical G alignment between a finite dimensional GKLS generator and its
induced classical Markov chain, in the detailed balance thermal case.

Model:
  - Finite Hilbert space of dimension dim.
  - Thermal weights pi_therm from a diagonal Hamiltonian H with energies E_i.
  - Jump operators L_{i<-j} = sqrt(gamma_{ij}) |i><j| implementing jumps j -> i.
  - Rates gamma_{ij} are constructed from symmetric base couplings k_{ij} and
    thermal weights pi_therm so that detailed balance holds exactly:
        pi_j * gamma_{ij} = pi_i * gamma_{ji}.

From this GKLS generator we:

  1) Build the classical Markov generator Q_markov from the same rates gamma.

  2) Recover a numerical classical generator Q_gkls by applying the GKLS
     dissipator to basis diagonal density matrices |j><j| and reading off
     the induced dp/dt on the diagonal.

  3) Verify that Q_gkls and Q_markov coincide to high precision, and that
     both share the same stationary distribution pi_therm.

  4) Construct the canonical irreversible operator on the classical density
     manifold:
         G_true = Q_markov @ diag(pi_therm),
     and check that:
       - G_true is symmetric (Dirichlet form is a quadratic form).
       - For small perturbations q = pi ⊙ mu, the drift v satisfies
             v = Q_markov q = G_true mu
         up to numerical precision.
       - The Dirichlet form from rates,
             E_pair(μ) = 0.5 ∑_{i,j} π_j γ_{ij} (μ_i - μ_j)^2,
         matches - μᵀ G_true μ.

The script reports detailed diagnostics and a final PASS / FAIL summary
for the canonical G alignment.

No external dependencies beyond numpy are required.
"""

import math
import numpy as np


def build_thermal_pi(dim, beta=1.0):
    """
    Build a nontrivial thermal distribution pi_i ∝ exp(-beta * E_i).

    We choose a simple increasing energy ladder E_i.
    """
    E = np.linspace(0.0, 1.5, dim)
    weights = np.exp(-beta * E)
    pi = weights / weights.sum()
    return E, pi


def build_symmetric_base_rates(dim, rng, base_scale=1.0, jitter=0.5):
    """
    Build symmetric base couplings k_ij for i != j.

    We use a sparse nearest neighbour ring for clarity. The symmetry
    k_ij = k_ji is enforced exactly.
    """
    k = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        j_plus = (i + 1) % dim
        j_minus = (i - 1) % dim
        for j in (j_plus, j_minus):
            if k[i, j] == 0.0 and k[j, i] == 0.0:
                noise = 1.0 + jitter * rng.normal()
                val = base_scale * abs(noise)
                k[i, j] = val
                k[j, i] = val
    return k


def build_gamma_from_pi_and_k(pi, k):
    """
    Build jump rates gamma_{ij} from symmetric k_ij and stationary weights pi_i,
    enforcing detailed balance pi_j gamma_{ij} = pi_i gamma_{ji}.

    We set
        gamma_{ij} = k_ij * sqrt(pi_i / pi_j)
    for i != j, and gamma_{ii} = 0.
    """
    dim = pi.size
    gamma = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            if k[i, j] != 0.0:
                gamma[i, j] = k[i, j] * math.sqrt(pi[i] / pi[j])
    return gamma


def build_Q_from_gamma(gamma):
    """
    Build the classical Markov generator Q from jump rates gamma_{ij} (j -> i).

    For k != j: Q_{kj} = gamma_{kj}.
    For k == j: Q_{kk} = - sum_{i != k} gamma_{ik}.

    We evolve column densities p via dp/dt = Q p, so columns of Q sum to 0.
    """
    dim = gamma.shape[0]
    Q = np.zeros((dim, dim), dtype=float)
    for k in range(dim):
        for j in range(dim):
            if j != k:
                Q[k, j] = gamma[k, j]
        Q[k, k] = -float(np.sum(gamma[:, k])) + gamma[k, k]
    return Q


def build_Lindblad_ops(gamma):
    """
    Build Lindblad jump operators L_{i<-j} = sqrt(gamma_{ij}) |i><j| for all
    i != j with gamma_{ij} > 0.

    Returns a list of complex matrices of shape (dim, dim).
    """
    dim = gamma.shape[0]
    L_ops = []
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            rate = gamma[i, j]
            if rate > 0.0:
                L = np.zeros((dim, dim), dtype=complex)
                L[i, j] = math.sqrt(rate)
                L_ops.append(L)
    return L_ops


def dissipator_rho(rho, L_ops):
    """
    Compute the GKLS dissipator D(ρ) for jump operators L_ops and H = 0:

        D(ρ) = sum_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ}).

    rho is a (dim, dim) complex matrix.
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


def build_Q_from_GKLS_diag(L_ops):
    """
    Recover a classical generator Q_gkls by applying the GKLS dissipator
    to basis diagonal density matrices |j><j|.

    For each j, let ρ_j = |j><j|, compute D(ρ_j), and set the k-th diagonal
    element (k, k) of D(ρ_j) to (Q_gkls)_{kj}.
    """
    dim = L_ops[0].shape[0]
    Q = np.zeros((dim, dim), dtype=float)
    for j in range(dim):
        rho_j = np.zeros((dim, dim), dtype=complex)
        rho_j[j, j] = 1.0
        D_rho_j = dissipator_rho(rho_j, L_ops)
        diag_D = np.real(np.diag(D_rho_j))
        Q[:, j] = diag_D
    return Q


def compute_stationary_from_Q(Q):
    """
    Compute a stationary distribution pi_stationary for Q by finding a
    right eigenvector of Q with eigenvalue closest to zero and normalising it.

    This is consistent with dp/dt = Q p, where p is a column density.
    """
    evals, evecs = np.linalg.eig(Q)
    idx = np.argmin(np.abs(evals))
    v = np.real(evecs[:, idx])
    # Normalise to sum 1 and enforce nonnegativity by flipping sign if needed
    v = v / np.sum(v)
    if np.any(v < 0):
        v = -v
        v = v / np.sum(v)
    return v


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


def run():
    print("GKLS to Markov G-unification")
    print("-----------------------------")

    dim = 4
    beta = 1.0
    rng_seed = 13579
    rng = np.random.default_rng(rng_seed)

    print(f"dim = {dim}, beta = {beta}, rng_seed = {rng_seed}")
    print("")

    # Build thermal weights and symmetric base couplings
    E, pi_therm = build_thermal_pi(dim, beta=beta)
    k = build_symmetric_base_rates(dim, rng, base_scale=1.0, jitter=0.5)
    gamma = build_gamma_from_pi_and_k(pi_therm, k)

    print("Thermal data:")
    print(f"  Energies E_i: {E}")
    print(f"  Thermal pi_therm: {pi_therm}")
    print("")
    print("Base symmetric couplings k_ij (nonzero entries):")
    for i in range(dim):
        for j in range(dim):
            if i < j and k[i, j] != 0.0:
                print(f"  k[{i},{j}] = k[{j},{i}] ≈ {k[i,j]:.3e}")
    print("")

    # Build classical Markov generator from gamma
    Q_markov = build_Q_from_gamma(gamma)

    # Build GKLS jump operators and recover Q_gkls from diagonal dynamics
    L_ops = build_Lindblad_ops(gamma)
    Q_gkls = build_Q_from_GKLS_diag(L_ops)

    # Compare Q_markov and Q_gkls
    frob = lambda M: float(np.linalg.norm(M, ord="fro"))
    diff_Q = Q_markov - Q_gkls
    norm_Q = frob(Q_markov)
    rel_err_Q = frob(diff_Q) / norm_Q

    print("Classical generator diagnostics:")
    print(f"  ||Q_markov - Q_gkls||_F / ||Q_markov||_F ≈ {rel_err_Q:.3e}")
    print("  Q_markov:")
    print(Q_markov)
    print("  Q_gkls:")
    print(Q_gkls)
    print("")

    # Stationary distributions
    Q_pi = Q_markov @ pi_therm
    pi_from_Q = compute_stationary_from_Q(Q_markov)
    diff_pi = pi_from_Q - pi_therm
    rel_err_pi = float(np.linalg.norm(diff_pi, ord=2) / np.linalg.norm(pi_therm, ord=2))

    print("Stationary state diagnostics:")
    print(f"  ||Q_markov pi_therm||_2 ≈ {float(np.linalg.norm(Q_pi, ord=2)):.3e} (should be near 0)")
    print(f"  pi_from_Q (right eigenvector of Q near 0): {pi_from_Q}")
    print(f"  Relative difference pi_from_Q vs pi_therm ≈ {rel_err_pi:.3e}")
    print("")

    # Check that rho_ss = diag(pi_therm) is stationary for GKLS
    rho_ss = np.diag(pi_therm.astype(complex))
    D_rho_ss = dissipator_rho(rho_ss, L_ops)
    norm_D_ss = float(np.linalg.norm(D_rho_ss, ord="fro"))
    diag_D_ss = np.real(np.diag(D_rho_ss))

    print("GKLS stationary state diagnostics:")
    print(f"  ||D(rho_ss)||_F ≈ {norm_D_ss:.3e} (should be near 0)")
    print(f"  diag(D(rho_ss)) ≈ {diag_D_ss}")
    print("")

    # Canonical irreversible operator on classical density manifold
    G_true = Q_markov @ np.diag(pi_therm)
    G_sym_resid, G_skew_resid = symmetry_metrics(G_true)

    print("Canonical G_true = Q_markov diag(pi_therm) diagnostics:")
    print(f"  Symmetry residual ||G_true - G_true^T||/||G_true|| ≈ {G_sym_resid:.3e}")
    print(f"  Skew residual     ||G_true + G_true^T||/||G_true|| ≈ {G_skew_resid:.3e}")
    print("")

    # Drift consistency on random perturbations
    n_probes = 40
    mu_batch = rng.normal(size=(n_probes, dim))
    mu_batch = mu_batch - mu_batch.mean(axis=1, keepdims=True)

    v_Q = np.zeros_like(mu_batch)
    v_G = np.zeros_like(mu_batch)
    for k_probe in range(n_probes):
        mu = mu_batch[k_probe]
        q = pi_therm * mu
        v_Q[k_probe] = Q_markov @ q
        v_G[k_probe] = G_true @ mu

    err_v = v_Q - v_G
    rel_err_v = frob(err_v) / frob(v_Q)

    print("Canonical drift consistency diagnostics (random probes):")
    print(f"  ||G_true mu - Q (pi ⊙ mu)||_F / ||Q (pi ⊙ mu)||_F ≈ {rel_err_v:.3e}")
    print("")

    # Dirichlet form consistency:
    # E_pair(mu) = 0.5 * sum_{i,j} pi_j gamma_{ij} (mu_i - mu_j)^2
    # should match - mu^T G_true mu.
    sigma_pair_list = []
    sigma_G_list = []
    for k_probe in range(n_probes):
        mu = mu_batch[k_probe]
        # Pair-sum Dirichlet form from rates
        sigma_pair = 0.0
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                sigma_pair += 0.5 * pi_therm[j] * gamma[i, j] * (mu[i] - mu[j]) ** 2
        # Quadratic from G_true
        vG = v_G[k_probe]
        sigma_G = -float(np.dot(mu, vG))
        sigma_pair_list.append(sigma_pair)
        sigma_G_list.append(sigma_G)

    sigma_pair_arr = np.array(sigma_pair_list)
    sigma_G_arr = np.array(sigma_G_list)
    diff_sigma = sigma_pair_arr - sigma_G_arr

    # Restrict to nontrivial cases
    mask = np.abs(sigma_pair_arr) > 1e-12
    if np.any(mask):
        rel_err_sigma = float(
            np.linalg.norm(diff_sigma[mask], ord=2)
            / np.linalg.norm(sigma_pair_arr[mask], ord=2)
        )
    else:
        rel_err_sigma = 0.0

    print("Dirichlet form consistency diagnostics:")
    print(f"  Relative difference between E_pair(mu) and -mu^T G_true mu ≈ {rel_err_sigma:.3e}")
    print("")

    # PASS / FAIL criteria
    tol_Q = 1e-10
    tol_Q_pi = 1e-10
    tol_pi = 1e-10
    tol_rho_ss = 1e-10
    tol_sym_G = 1e-12
    tol_v = 1e-10
    tol_sigma = 1e-10

    pass_Q = rel_err_Q < tol_Q
    pass_Q_pi = float(np.linalg.norm(Q_pi, ord=2)) < tol_Q_pi
    pass_pi = rel_err_pi < tol_pi
    pass_rho_ss = norm_D_ss < tol_rho_ss
    pass_G_sym = G_sym_resid < tol_sym_G
    pass_v = rel_err_v < tol_v
    pass_sigma = rel_err_sigma < tol_sigma

    all_pass = (
        pass_Q
        and pass_Q_pi
        and pass_pi
        and pass_rho_ss
        and pass_G_sym
        and pass_v
        and pass_sigma
    )

    print("Summary:")
    print(f"  Q_markov ≈ Q_gkls?                               {pass_Q} (tol = {tol_Q})")
    print(f"  Q_markov pi_therm ≈ 0?                          {pass_Q_pi} (tol = {tol_Q_pi})")
    print(f"  pi_from_Q ≈ pi_therm?                           {pass_pi} (tol = {tol_pi})")
    print(f"  GKLS D(rho_ss) ≈ 0?                             {pass_rho_ss} (tol = {tol_rho_ss})")
    print(f"  G_true symmetric up to tol?                     {pass_G_sym} (tol = {tol_sym_G})")
    print(f"  G_true mu ≈ Q (pi ⊙ mu) on random probes?       {pass_v} (tol = {tol_v})")
    print(f"  E_pair(mu) ≈ -mu^T G_true mu (Dirichlet form)?  {pass_sigma} (tol = {tol_sigma})")
    print("")

    if all_pass:
        print("GKLS to Markov G-unification CHECK: PASS")
        print("  The GKLS jump model induces a classical Markov generator Q that")
        print("  shares the same stationary distribution as the thermal state, and")
        print("  the canonical G_true = Q diag(pi) is symmetric and reproduces both")
        print("  the drift and the Dirichlet form from the jump rates, as expected")
        print("  in the Fisher-metriplectic framework.")
    else:
        print("GKLS to Markov G-unification CHECK: FAIL (see diagnostics above).")


if __name__ == "__main__":
    run()
