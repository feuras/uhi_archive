#!/usr/bin/env python3
"""
26_gkls_coherent_dressing_fp_chain.py

Coherent dressing of the FP -> Markov -> GKLS realisation.

We start from the reversible nearest neighbour Markov generator Q and the
diagonal jump GKLS generator K_diss constructed from it, as in
25_fp_to_markov_to_gkls_realisation.py.

Then:

  1) We construct a tight binding Hamiltonian H on the same lattice, with
     hopping between nearest neighbours on the ring.

  2) We build the Hamiltonian superoperator

         K_H = -i (I ⊗ H - H^T ⊗ I),

     so that d rho / dt = -i [H, rho] corresponds to d/dt vec(rho) = K_H vec(rho).

  3) We define the total GKLS generator

         K_tot = K_diss + K_H.

     This has the same stationary state rho_ss = diag(pi) as K_diss, since
     [H, rho_ss] = 0 when rho_ss is diagonal in the computational basis.

  4) We check that

       - K_tot leaves rho_ss stationary,
       - the effective density generator Q_eff extracted from K_tot matches Q,
       - the density sector Dirichlet form built from the BKM metric and the
         symmetric part G_tot of K_tot matches both the classical Fisher
         Dirichlet and the density sector Dirichlet of K_diss.

This shows explicitly that many different coherent GKLS models, differing by
their Hamiltonian part, share the same irreversible Fisher hydrodynamics on
densities: G is fixed, J changes.
"""

import numpy as np


# ----------------------------------------------------------------------
# Utility functions (adapted from script 25)
# ----------------------------------------------------------------------

def vec(mat):
    """Column-stack a matrix."""
    return np.asarray(mat, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    """Inverse of vec for N x N matrices."""
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def build_potential(x, V0=0.0, alpha=1.0, beta=0.5):
    """
    Example potential on [0, L):

        V(x) = V0 + alpha * cos(x) + beta * cos(2x)
    """
    return V0 + alpha * np.cos(x) + beta * np.cos(2.0 * x)


def build_reversible_markov_from_potential(x, V, D=1.0):
    """
    Build a reversible nearest neighbour Markov generator Q on a periodic
    lattice from a potential V(x), with column sums zero and stationary
    vector pi.

    We use a detailed balance scheme:

      - pi_i ∝ exp(-V_i),
      - symmetric conductance c = D / dx^2,
      - rates

            W_{i->i+1} = c / pi_i,
            W_{i+1->i} = c / pi_{i+1},

        so that pi_i W_{i->i+1} = pi_{i+1} W_{i+1->i}.

      - Markov generator with column-sum-zero convention:

            Q_{ij} = W_{j->i} for i != j,
            Q_{ii} = -sum_k W_{i->k}.
    """
    N = len(x)
    L = x[-1] + (x[1] - x[0])
    dx = L / N

    V = np.asarray(V, dtype=float)
    pi = np.exp(-V)
    pi /= pi.sum()

    c = D / (dx * dx)
    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        ip1 = (i + 1) % N
        W[i, ip1] = c / pi[i]
        W[ip1, i] = c / pi[ip1]

    Q = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                Q[i, j] = W[j, i]
        Q[i, i] = -np.sum(W[i, :])

    return Q, pi


def build_gkls_superoperator_from_Q(Q):
    """
    Construct a diagonal jump GKLS superoperator K_diss on an N dimensional
    Hilbert space whose density sector generator is exactly Q.

    For each off diagonal entry Q_{ij} >= 0 with i != j we introduce

        L_ij = sqrt(Q_ij) |i><j|,

    and form

        K_diss = sum_{i!=j} [ L_ij ⊗ L_ij^*
                              - 0.5 (I ⊗ (L_ij^dag L_ij)^T
                                     + (L_ij^dag L_ij) ⊗ I) ].
    """
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    dim = N * N
    I_N = np.eye(N, dtype=complex)
    K = np.zeros((dim, dim), dtype=complex)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            gamma_ij = Q[i, j]
            if gamma_ij <= 0.0:
                continue
            L_ij = np.zeros((N, N), dtype=complex)
            L_ij[i, j] = np.sqrt(gamma_ij)
            LdL = L_ij.conj().T @ L_ij

            jump = np.kron(L_ij.conj(), L_ij)
            left = np.kron(I_N, LdL)
            right = np.kron(LdL.T, I_N)

            K += jump - 0.5 * (left + right)

    return K


def build_tight_binding_H(N, J_hop=1.0):
    """
    Tight binding Hamiltonian on a ring of N sites:

        H = -J_hop * sum_i ( |i><i+1| + |i+1><i| ),

    with periodic wrap.
    """
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        ip1 = (i + 1) % N
        H[i, ip1] = -J_hop
        H[ip1, i] = -J_hop
    return H


def build_H_superoperator(H):
    """
    Hamiltonian superoperator for d rho / dt = -i [H, rho]:

        K_H = -i (I ⊗ H - H^T ⊗ I).
    """
    N = H.shape[0]
    I_N = np.eye(N, dtype=complex)
    K_H = -1j * (np.kron(I_N, H) - np.kron(H.T, I_N))
    return K_H


def bkm_weights(pi):
    """
    BKM weights c_ij at rho_ss = diag(pi) in its eigenbasis.

    For commuting families:

      c_ii = 1 / pi_i,
      c_ij = (log pi_i - log pi_j) / (pi_i - pi_j) if pi_i != pi_j.
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


def build_Ksharp_and_G(K, M_diag):
    """
    Given K and diagonal metric M_diag in the vec basis, build metric adjoint
    K^sharp and symmetric part G = (K + K^sharp) / 2.

    For diagonal M:

        K^sharp = M^{-1} K^dag M.
    """
    M_diag = np.asarray(M_diag, dtype=float)
    M_inv = 1.0 / M_diag
    Kdag = K.conj().T
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G = 0.5 * (K + Ksharp)
    return Ksharp, G


def extract_Q_eff_from_K(K, N):
    """
    Extract effective density generator Q_eff from K in the computational
    basis, assuming rho_ss is diagonal in this basis.

    For each diagonal basis matrix E_jj we compute K(E_jj) and read off the
    induced drift of the diagonal entries.
    """
    Q_eff = np.zeros((N, N), dtype=float)
    for j in range(N):
        e = np.zeros(N, dtype=float)
        e[j] = 1.0
        rho = np.diag(e)
        v = vec(rho)
        dv = K @ v
        drho = unvec(dv, N)
        dp = np.diag(drho).real
        Q_eff[:, j] = dp
    return Q_eff


def classical_fisher_dirichlet(delta_p, pi, Q):
    """
    Classical Fisher Dirichlet for dp/dt = Q p with stationary pi, using
    column-sum-zero convention:

        w_ij = Q_{j i},
        E = 0.5 sum_{i,j} pi_i w_ij (phi_j - phi_i)^2,
        phi_i = delta_p_i / pi_i.
    """
    delta_p = np.asarray(delta_p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    Q = np.asarray(Q, dtype=float)
    N = len(pi)

    w = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                w[i, j] = Q[j, i]

    phi = delta_p / pi
    E = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            E += 0.5 * pi[i] * w[i, j] * (phi[j] - phi[i]) ** 2
    return float(E)


def gkls_density_dirichlet(delta_p, G, M_diag, N):
    """
    GKLS density sector Dirichlet from G and BKM metric M_diag at
    rho_ss = diag(pi):

        E_GKLS = - <delta u, M G delta u>,

    where delta u = vec(diag(delta_p)).
    """
    delta_p = np.asarray(delta_p, dtype=float)
    delta_rho = np.diag(delta_p)
    delta_u = vec(delta_rho)
    MG_du = M_diag * (G @ delta_u)
    E = -np.vdot(delta_u, MG_du).real
    return float(E)


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main():
    # Domain and diffusion as in script 25
    L = 2.0 * np.pi
    N_x = 40
    D = 1.0

    x = np.linspace(0.0, L, N_x, endpoint=False)
    V = build_potential(x, V0=0.0, alpha=1.0, beta=0.5)

    print("26_gkls_coherent_dressing_fp_chain.py")
    print("-------------------------------------")
    print(f"Domain length L               = {L:.6f}")
    print(f"Number of lattice points N_x  = {N_x}")
    print(f"Diffusion coefficient D       = {D:.3f}")
    print()

    # Build reversible Markov generator Q and stationary pi
    Q, pi = build_reversible_markov_from_potential(x, V, D=D)
    col_sums = Q.sum(axis=0)
    Qpi_resid = np.linalg.norm(Q @ pi)
    print(f"Max column sum residual Q     = {np.max(np.abs(col_sums)):.3e}")
    print(f"Stationarity residual ||Q pi||= {Qpi_resid:.3e}")
    print(f"pi: min, max                  = {pi.min():.3e}, {pi.max():.3e}")
    print()

    # GKLS dissipative lift
    K_diss = build_gkls_superoperator_from_Q(Q)
    N = N_x

    # Stationary state rho_ss = diag(pi)
    rho_ss = np.diag(pi)
    v_ss = vec(rho_ss)
    stat_resid_diss = np.linalg.norm(K_diss @ v_ss)
    print(f"||K_diss vec(rho_ss)||        = {stat_resid_diss:.3e}")

    # Hamiltonian dressing
    J_hop = 1.0
    H = build_tight_binding_H(N, J_hop=J_hop)
    K_H = build_H_superoperator(H)
    K_tot = K_diss + K_H

    stat_resid_tot = np.linalg.norm(K_tot @ v_ss)
    print(f"||K_tot vec(rho_ss)||         = {stat_resid_tot:.3e}")
    print()

    # Effective density generators
    Q_eff_diss = extract_Q_eff_from_K(K_diss, N)
    Q_eff_tot = extract_Q_eff_from_K(K_tot, N)

    diff_Q_diss = np.max(np.abs(Q_eff_diss - Q))
    diff_Q_tot = np.max(np.abs(Q_eff_tot - Q))
    diff_Q_between = np.max(np.abs(Q_eff_tot - Q_eff_diss))

    print(f"Max |Q_eff_diss - Q|          = {diff_Q_diss:.3e}")
    print(f"Max |Q_eff_tot  - Q|          = {diff_Q_tot:.3e}")
    print(f"Max |Q_eff_tot - Q_eff_diss|  = {diff_Q_between:.3e}")
    print()

    # BKM metric and symmetric parts G_diss, G_tot
    C = bkm_weights(pi)
    M_diag = C.reshape(-1, order="F")

    _, G_diss = build_Ksharp_and_G(K_diss, M_diag)
    _, G_tot = build_Ksharp_and_G(K_tot, M_diag)

    # Dirichlet comparisons
    rng = np.random.default_rng(12345)
    n_tests = 50

    max_abs_cl_vs_diss = 0.0
    max_rel_cl_vs_diss = 0.0

    max_abs_cl_vs_tot = 0.0
    max_rel_cl_vs_tot = 0.0

    max_abs_diss_vs_tot = 0.0
    max_rel_diss_vs_tot = 0.0

    for _ in range(n_tests):
        delta = rng.normal(size=N)
        delta -= delta.mean()
        delta *= 0.1 / np.max(np.abs(delta))
        delta_p = delta

        E_cl = classical_fisher_dirichlet(delta_p, pi, Q)
        E_diss = gkls_density_dirichlet(delta_p, G_diss, M_diag, N)
        E_tot = gkls_density_dirichlet(delta_p, G_tot, M_diag, N)

        max_abs_cl_vs_diss = max(max_abs_cl_vs_diss, abs(E_diss - E_cl))
        max_abs_cl_vs_tot = max(max_abs_cl_vs_tot, abs(E_tot - E_cl))
        max_abs_diss_vs_tot = max(max_abs_diss_vs_tot, abs(E_tot - E_diss))

        if abs(E_cl) > 1e-14:
            max_rel_cl_vs_diss = max(
                max_rel_cl_vs_diss, abs(E_diss - E_cl) / abs(E_cl)
            )
            max_rel_cl_vs_tot = max(
                max_rel_cl_vs_tot, abs(E_tot - E_cl) / abs(E_cl)
            )
        if abs(E_diss) > 1e-14:
            max_rel_diss_vs_tot = max(
                max_rel_diss_vs_tot, abs(E_tot - E_diss) / abs(E_diss)
            )

    print("Dirichlet comparisons over random density perturbations:")
    print(f"  max |E_diss - E_cl|         = {max_abs_cl_vs_diss:.3e}")
    print(f"  max rel(E_diss, E_cl)       = {max_rel_cl_vs_diss:.3e}")
    print(f"  max |E_tot  - E_cl|         = {max_abs_cl_vs_tot:.3e}")
    print(f"  max rel(E_tot, E_cl)        = {max_rel_cl_vs_tot:.3e}")
    print(f"  max |E_tot  - E_diss|       = {max_abs_diss_vs_tot:.3e}")
    print(f"  max rel(E_tot, E_diss)      = {max_rel_diss_vs_tot:.3e}")


if __name__ == "__main__":
    main()
