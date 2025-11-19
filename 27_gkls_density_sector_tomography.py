#!/usr/bin/env python3
"""
27_gkls_density_sector_tomography.py

Density-sector tomography for a diagonal-jump GKLS generator.

Goal:
  Show that for a finite GKLS model built from a reversible Markov generator Q,
  the effective density-sector generator Q_eff can be reconstructed purely from
  density responses, and that the Fisher Dirichlet built from the reconstructed
  Q_rec matches both the exact classical Dirichlet and the GKLS Dirichlet from
  the BKM + G construction.

Steps:

  1) Construct a random 3-state reversible Markov generator Q with a strictly
     positive stationary distribution pi, using symmetric "conductances" S_ij.

  2) Lift Q to a diagonal-jump GKLS generator K_diss on a 3-dimensional Hilbert
     space via Lindblad operators L_ij = sqrt(Q_ij) |i><j|.

  3) Confirm that rho_ss = diag(pi) is stationary for K_diss and extract the
     exact effective density generator Q_eff_exact from K_diss by acting on the
     diagonal basis matrices E_jj.

  4) Perform "tomography" of Q_eff using only diagonal states:
       - Sample a collection of random probability vectors p^(k) (full support).
       - For each p^(k), build rho^(k) = diag(p^(k)) and compute
             dp^(k)/dt = diag( K_diss rho^(k) ).
       - Stack these into matrices D and P such that D ≈ Q_eff P and solve for
         Q_rec by least squares.

     This mimics an operational reconstruction of the density generator from
     observed density response data.

  5) Build the BKM metric at rho_ss, compute K^sharp and G = (K + K^sharp)/2,
     and use G to evaluate the GKLS density-sector Dirichlet
         E_GKLS(delta p) = -<delta u, M G delta u>,
     where delta u = vec(diag(delta p)).

  6) Compare:
       - Q_rec vs the original Q and vs Q_eff_exact,
       - the classical Dirichlet built from Q_rec vs that built from Q and vs
         the GKLS Dirichlet from G.

The expectation is:
  * Q_rec ≈ Q ≈ Q_eff_exact to machine precision,
  * the Fisher Dirichlet built from Q_rec matches both the classical and GKLS
    Dirichlet forms on densities up to numerical noise.
"""

import numpy as np


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def vec(mat):
    """Column-stack a matrix."""
    return np.asarray(mat, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    """Inverse of vec for N x N matrices."""
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def build_random_reversible_markov(N=3, rate_scale=1.0, seed=12345):
    """
    Build a random reversible Markov generator Q on N states with strictly
    positive stationary distribution pi.

    Construction:

      - Sample random positive pi_i and normalise to sum 1.
      - Sample a symmetric "conductance" matrix S_ij >= 0 for i < j and set
            S_ji = S_ij, S_ii = 0.
      - Define off-diagonal rates using detailed balance:
            pi_j Q_ij = S_ij  =>  Q_ij = S_ij / pi_j  for i != j,
            Q_jj = -sum_{i!=j} Q_ij.

    This ensures detailed balance:
        pi_j Q_ij = S_ij = pi_i Q_ji,
    and column-sum-zero convention for dp/dt = Q p.
    """
    rng = np.random.default_rng(seed)
    pi_raw = rng.random(N)
    pi = pi_raw / pi_raw.sum()

    S = rng.random((N, N)) * rate_scale
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)

    Q = np.zeros((N, N), dtype=float)
    for j in range(N):
        for i in range(N):
            if i != j:
                Q[i, j] = S[i, j] / pi[j]
        Q[j, j] = -np.sum(Q[:, j]) + Q[j, j]  # ensure exact column sums

    return Q, pi


def build_gkls_superoperator_from_Q(Q):
    """
    Construct a diagonal jump GKLS superoperator K_diss on an N dimensional
    Hilbert space whose density sector generator is exactly Q.

    For each off-diagonal entry Q_ij >= 0 with i != j we introduce

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


def extract_Q_eff_from_K(K, N):
    """
    Extract effective density generator Q_eff from K in the computational
    basis, assuming rho_ss is diagonal in this basis.

    For each diagonal basis matrix E_jj we compute K(E_jj) and read off the
    induced drift of the diagonal entries. This yields the columns of Q_eff.
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
    K^sharp and symmetric part G = (K + K^sharp)/2.

    For diagonal M:

        K^sharp = M^{-1} K^dag M.
    """
    M_diag = np.asarray(M_diag, dtype=float)
    M_inv = 1.0 / M_diag
    Kdag = K.conj().T
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G = 0.5 * (K + Ksharp)
    return Ksharp, G


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
    GKLS density-sector Dirichlet from G and BKM metric M_diag at
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
    N = 3
    rate_scale = 1.0
    rng = np.random.default_rng(12345)

    print("27_gkls_density_sector_tomography.py")
    print("------------------------------------")
    print(f"Hilbert / Markov dimension N   = {N}")
    print(f"Rate scale                     = {rate_scale:.3f}")
    print()

    # 1) Random reversible Markov generator and stationary pi
    Q, pi = build_random_reversible_markov(N=N, rate_scale=rate_scale, seed=12345)
    col_sums = Q.sum(axis=0)
    Qpi_resid = np.linalg.norm(Q @ pi)
    print("Random reversible Q:")
    print(Q)
    print(f"Max column sum residual Q      = {np.max(np.abs(col_sums)):.3e}")
    print(f"Stationarity residual ||Q pi|| = {Qpi_resid:.3e}")
    print(f"pi: min, max                   = {pi.min():.3e}, {pi.max():.3e}")
    print()

    # 2) GKLS dissipative lift
    K_diss = build_gkls_superoperator_from_Q(Q)

    rho_ss = np.diag(pi)
    v_ss = vec(rho_ss)
    stat_resid_K = np.linalg.norm(K_diss @ v_ss)
    print(f"||K_diss vec(rho_ss)||         = {stat_resid_K:.3e}")
    print()

    # 3) Exact density-sector generator from K_diss
    Q_eff_exact = extract_Q_eff_from_K(K_diss, N)
    diff_Q_eff = np.max(np.abs(Q_eff_exact - Q))
    col_sums_eff = Q_eff_exact.sum(axis=0)
    Qeff_pi_resid = np.linalg.norm(Q_eff_exact @ pi)
    print("Exact density-sector generator Q_eff from GKLS:")
    print(Q_eff_exact)
    print(f"Max |Q_eff_exact - Q|          = {diff_Q_eff:.3e}")
    print(f"Max column sum residual Q_eff  = {np.max(np.abs(col_sums_eff)):.3e}")
    print(f"Stationarity residual ||Q_eff pi|| = {Qeff_pi_resid:.3e}")
    print()

    # 4) Tomography of Q_eff from density responses
    n_probes = 12
    P = np.zeros((N, n_probes), dtype=float)
    D = np.zeros((N, n_probes), dtype=float)

    for k in range(n_probes):
        # random probability vector with full support
        w = rng.random(N)
        p = w / w.sum()
        rho = np.diag(p)
        v = vec(rho)
        dv = K_diss @ v
        drho = unvec(dv, N)
        dp = np.diag(drho).real

        P[:, k] = p
        D[:, k] = dp

    # Solve D ≈ Q_rec P in least squares sense: D P^T ≈ Q_rec P P^T
    PPt = P @ P.T
    DPt = D @ P.T
    # Regularised inverse if needed
    try:
        PPt_inv = np.linalg.inv(PPt)
    except np.linalg.LinAlgError:
        PPt_inv = np.linalg.pinv(PPt)
    Q_rec = DPt @ PPt_inv

    # Enforce column-sum-zero constraint softly by projecting each column
    for j in range(N):
        col_sum = np.sum(Q_rec[:, j])
        Q_rec[:, j] -= col_sum / N

    col_sums_rec = Q_rec.sum(axis=0)
    Qrec_pi_resid = np.linalg.norm(Q_rec @ pi)
    diff_Q_rec = np.max(np.abs(Q_rec - Q))
    diff_Q_rec_eff = np.max(np.abs(Q_rec - Q_eff_exact))

    print("Tomographically reconstructed Q_rec from density responses:")
    print(Q_rec)
    print(f"Max |Q_rec - Q|                = {diff_Q_rec:.3e}")
    print(f"Max |Q_rec - Q_eff_exact|      = {diff_Q_rec_eff:.3e}")
    print(f"Max column sum residual Q_rec  = {np.max(np.abs(col_sums_rec)):.3e}")
    print(f"Stationarity residual ||Q_rec pi|| = {Qrec_pi_resid:.3e}")
    print()

    # 5) BKM metric and G from GKLS
    C = bkm_weights(pi)
    M_diag = C.reshape(-1, order="F")
    _, G = build_Ksharp_and_G(K_diss, M_diag)

    # 6) Dirichlet comparisons using Q, Q_rec and G
    n_tests = 50
    max_abs_Q_vs_G = 0.0
    max_rel_Q_vs_G = 0.0

    max_abs_Qrec_vs_Q = 0.0
    max_rel_Qrec_vs_Q = 0.0

    max_abs_Qrec_vs_G = 0.0
    max_rel_Qrec_vs_G = 0.0

    for _ in range(n_tests):
        delta = rng.normal(size=N)
        delta -= delta.mean()
        delta *= 0.1 / np.max(np.abs(delta))
        delta_p = delta

        E_cl_Q = classical_fisher_dirichlet(delta_p, pi, Q)
        E_cl_Qrec = classical_fisher_dirichlet(delta_p, pi, Q_rec)
        E_gkls = gkls_density_dirichlet(delta_p, G, M_diag, N)

        max_abs_Q_vs_G = max(max_abs_Q_vs_G, abs(E_gkls - E_cl_Q))
        max_abs_Qrec_vs_Q = max(max_abs_Qrec_vs_Q, abs(E_cl_Qrec - E_cl_Q))
        max_abs_Qrec_vs_G = max(max_abs_Qrec_vs_G, abs(E_gkls - E_cl_Qrec))

        if abs(E_cl_Q) > 1e-14:
            max_rel_Q_vs_G = max(max_rel_Q_vs_G, abs(E_gkls - E_cl_Q) / abs(E_cl_Q))
            max_rel_Qrec_vs_Q = max(max_rel_Qrec_vs_Q, abs(E_cl_Qrec - E_cl_Q) / abs(E_cl_Q))
        if abs(E_gkls) > 1e-14:
            max_rel_Qrec_vs_G = max(max_rel_Qrec_vs_G, abs(E_gkls - E_cl_Qrec) / abs(E_gkls))

    print("Dirichlet comparisons over random density perturbations:")
    print(f"  max |E_GKLS - E_cl(Q)|      = {max_abs_Q_vs_G:.3e}")
    print(f"  max rel(E_GKLS, E_cl(Q))    = {max_rel_Q_vs_G:.3e}")
    print(f"  max |E_cl(Q_rec) - E_cl(Q)| = {max_abs_Qrec_vs_Q:.3e}")
    print(f"  max rel(E_cl(Q_rec), E_cl(Q)) = {max_rel_Qrec_vs_Q:.3e}")
    print(f"  max |E_GKLS - E_cl(Q_rec)|  = {max_abs_Qrec_vs_G:.3e}")
    print(f"  max rel(E_GKLS, E_cl(Q_rec)) = {max_rel_Qrec_vs_G:.3e}")


if __name__ == "__main__":
    main()
