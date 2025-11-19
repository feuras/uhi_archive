#!/usr/bin/env python3
"""
25_fp_to_markov_to_gkls_realisation.py

Constructive FP -> Markov -> GKLS realisation of a Fisher-metriplectic flow.

We:
  1) Choose a 1D periodic domain [0, L) and a potential V(x), giving a target
     stationary density pi(x) ∝ exp(-V(x)).

  2) Discretise [0, L) on a periodic lattice with N_x points and build a
     reversible nearest neighbour Markov generator Q with column sums zero.
     The continuum target is an overdamped FP equation in the Fisher class.

  3) Lift Q to a diagonal jump GKLS generator K on an N_x dimensional Hilbert
     space, with Lindblad rates γ_ij = Q_ij for i ≠ j, so that diagonal
     states evolve exactly as dp/dt = Q p.

  4) Take rho_ss = diag(pi) with pi_i ∝ exp(-V_i), build the BKM metric at
     rho_ss, construct the metric adjoint K^sharp and G = (K + K^sharp)/2.

  5) Extract an effective density generator Q_eff from K and compare:
       - Q_eff versus Q entrywise,
       - the density-sector GKLS Dirichlet form versus the classical Fisher
         Dirichlet built from (pi, Q).

This should demonstrate that a discrete FP-like Fisher flow can be realised
as the density sector of a GKLS semigroup constructed from its Markov generator.
"""

import numpy as np
from numpy.linalg import eigh


# ----------------------------------------------------------------------
# Utility functions
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

    which gives a nontrivial stationary density pi ∝ exp(-V).
    """
    return V0 + alpha * np.cos(x) + beta * np.cos(2.0 * x)


def build_reversible_markov_from_potential(x, V, D=1.0):
    """
    Build a reversible nearest neighbour Markov generator Q on a periodic
    lattice from a potential V(x), with column sums zero.

    We use a standard detailed-balance scheme with nearest neighbour jumps:

      - First construct the discrete stationary density
            pi_i ∝ exp(-V_i).

      - Then choose symmetric "conductances" c_{i,i+1} = D/dx^2 and define
            w_{i→i+1} = c_{i,i+1} / pi_i,
            w_{i+1→i} = c_{i,i+1} / pi_{i+1}.

        This ensures detailed balance:
            pi_i w_{i→i+1} = pi_{i+1} w_{i+1→i} = c_{i,i+1}.

      - Finally build Q with the column-sum-zero convention:
            Q_{ij} = W_{j→i} for i ≠ j,
            Q_{ii} = -sum_k W_{i→k},

        so that dp/dt = Q p, p a column vector.
    """
    N = len(x)
    L = x[-1] + (x[1] - x[0])
    dx = L / N

    V = np.asarray(V, dtype=float)
    # Target stationary density from the potential
    pi = np.exp(-V)
    pi /= pi.sum()

    # Symmetric conductance
    c = D / (dx * dx)

    # Rates W_{i->j}
    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        ip1 = (i + 1) % N
        # pair (i, ip1)
        W[i, ip1] = c / pi[i]
        W[ip1, i] = c / pi[ip1]

    # Build Q with column-sum-zero convention
    Q = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                # inflow to i from j
                Q[i, j] = W[j, i]
        # diagonal element: outflow from i
        Q[i, i] = -np.sum(W[i, :])

    return Q, pi


def build_gkls_superoperator_from_Q(Q):
    """
    Construct a diagonal jump GKLS superoperator K on an N dimensional
    Hilbert space whose density sector generator is exactly Q.

    We work in the computational basis |i>, and for each off diagonal entry
    Q_{ij} ≥ 0 with i ≠ j we introduce a Lindblad operator

        L_ij = sqrt(Q_ij) |i><j|.

    The GKLS generator on vec(rho) is then

        K = sum_{i≠j} (L_ij ⊗ L_ij^*)
            - 0.5 * (I ⊗ (L_ij^dag L_ij)^T + (L_ij^dag L_ij) ⊗ I),

    with no Hamiltonian part. For diagonal rho, this yields dp/dt = Q p.
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
            # Markov off-diagonals should be ≥ 0; guard against tiny negatives
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


def bkm_weights(pi):
    """
    BKM weights c_ij at rho_ss = diag(pi) in its eigenbasis.

    For a commuting family, these reduce to

      c_ii = 1/pi_i,
      c_ij = (log pi_i - log pi_j)/(pi_i - pi_j) if pi_i != pi_j.
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
    Given K and diagonal metric M_diag in the vec basis, build the metric
    adjoint K^sharp and symmetric part G = (K + K^sharp)/2.

    The metric inner product is <u, v>_M = u^* (M v), with M diagonal.
    The adjoint satisfies <u, K v>_M = <K^sharp u, v>_M, which gives

        K^sharp = M^{-1} K^dag M,

    when M is diagonal.
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

    We act K on the basis diagonal matrices E_jj and read off the induced
    drift of the diagonal entries.

    For each column j, dp/dt = Q_eff p has

        dp_i/dt = (Q_eff)_{ij}

    where dp_i is the diagonal of K(E_jj).
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
    Classical Fisher Dirichlet for dp/dt = Q p with stationary pi.

    For a Markov chain with generator Q (column sums zero), the rate from
    i → j is w_ij = Q_{j i}. We use

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
    rho_ss = diag(pi) in the computational basis.

        E_GKLS = - <delta u, M G delta u>,

    with delta u = vec(diag(delta_p)).
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
    # Continuum and lattice parameters
    L = 2.0 * np.pi
    N_x = 40
    D = 1.0

    x = np.linspace(0.0, L, N_x, endpoint=False)
    V = build_potential(x, V0=0.0, alpha=1.0, beta=0.5)

    print("25_fp_to_markov_to_gkls_realisation.py")
    print("--------------------------------------")
    print(f"Domain length L               = {L:.6f}")
    print(f"Number of lattice points N_x  = {N_x}")
    print(f"Diffusion coefficient D       = {D:.3f}")
    print()

    # Build reversible Markov generator Q and target stationary pi
    Q, pi = build_reversible_markov_from_potential(x, V, D=D)

    # Check column sums and stationarity residual
    col_sums = Q.sum(axis=0)
    print(f"Max column sum residual Q     = {np.max(np.abs(col_sums)):.3e}")

    Qpi_resid = np.linalg.norm(Q @ pi)
    print(f"Stationarity residual ||Q pi||= {Qpi_resid:.3e}")
    print(f"Stationary pi: min, max       = {pi.min():.3e}, {pi.max():.3e}")
    print(f"pi sum                        = {pi.sum():.6f}")
    print()

    # GKLS lift from Q
    K = build_gkls_superoperator_from_Q(Q)
    N = N_x

    # Stationary state rho_ss = diag(pi), check stationarity
    rho_ss = np.diag(pi)
    v_ss = vec(rho_ss)
    stat_resid = np.linalg.norm(K @ v_ss)
    print(f"Stationarity residual ||K vec(rho_ss)|| = {stat_resid:.3e}")

    # Extract effective density generator Q_eff from GKLS
    Q_eff = extract_Q_eff_from_K(K, N)
    diff_Q = np.max(np.abs(Q_eff - Q))
    print(f"Max |Q_eff - Q|               = {diff_Q:.3e}")
    print()

    # Build BKM metric and symmetric part G
    C = bkm_weights(pi)
    M_diag = C.reshape(-1, order="F")
    _, G = build_Ksharp_and_G(K, M_diag)

    # Dirichlet comparisons over random density perturbations
    rng = np.random.default_rng(12345)
    n_tests = 50

    max_abs_err = 0.0
    max_rel_err = 0.0
    min_Ec = float("inf")
    min_Eg = float("inf")
    max_Ec = 0.0
    max_Eg = 0.0

    for _ in range(n_tests):
        delta = rng.normal(size=N)
        delta -= delta.mean()
        # small amplitude
        delta *= 0.1 / np.max(np.abs(delta))
        delta_p = delta

        Ec = classical_fisher_dirichlet(delta_p, pi, Q)
        Eg = gkls_density_dirichlet(delta_p, G, M_diag, N)

        max_abs_err = max(max_abs_err, abs(Eg - Ec))
        if abs(Ec) > 1e-14:
            rel = abs(Eg - Ec) / abs(Ec)
            max_rel_err = max(max_rel_err, rel)

        min_Ec = min(min_Ec, Ec)
        min_Eg = min(min_Eg, Eg)
        max_Ec = max(max_Ec, Ec)
        max_Eg = max(max_Eg, Eg)

    print("Dirichlet comparison over random density perturbations:")
    print(f"  min E_classical             = {min_Ec:.3e}")
    print(f"  min E_GKLS                  = {min_Eg:.3e}")
    print(f"  max E_classical             = {max_Ec:.3e}")
    print(f"  max E_GKLS                  = {max_Eg:.3e}")
    print(f"  max |E_GKLS - E_classical|  = {max_abs_err:.3e}")
    print(f"  max relative error          = {max_rel_err:.3e}")


if __name__ == "__main__":
    main()
