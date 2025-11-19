#!/usr/bin/env python3
"""
22_gkls_nondiagonal_coherent_density_checks.py

Single explicit non diagonal GKLS model with genuine coherences:

  - 2 level system (qubit)
  - Hamiltonian H = 0.5 (Omega sigma_x + Delta sigma_z)
  - Dissipators: amplitude damping L1 = sqrt(gamma) sigma_minus,
                 dephasing     L2 = sqrt(gamma_phi) sigma_z

This GKLS has a unique stationary state rho_ss which generically has
non zero coherences in the computational basis.

We then:

  1. Compute rho_ss from the GKLS generator K.
  2. Diagonalise rho_ss = U diag(pi) U^\dagger, with pi > 0.
  3. Transform K into the rho_ss eigenbasis: K' = T^{-1} K T with
        T = U* \otimes U   (column stacking vec convention).
  4. On the subspace of diagonal matrices in this eigenbasis, extract the
     induced density generator Q_eff by acting K' on the diagonal basis
     elements E_11 and E_22 and taking the diagonal of the result.
  5. Verify:
        - column sums of Q_eff are ~ 0,
        - Q_eff pi ≈ 0,
        - detailed balance holds (2 state chain is always reversible).
  6. Construct the BKM metric at diag(pi), build K'^sharp, G', and compare
     the GKLS density sector Dirichlet form

         E_GKLS(delta p) = - <delta u, M G' delta u>

     with the classical Fisher Dirichlet built from (pi, Q_eff)

         E_classical(delta p) = 0.5 sum_{i != j} pi_i w_ij
                                (delta p_j/pi_j - delta p_i/pi_i)^2

     for many random mass conserving perturbations delta p.

This tests that even for a genuinely coherent, non diagonal GKLS model,
the density sector Fisher Dirichlet structure in the rho_ss eigenbasis is
exactly classical and reversible.
"""

import numpy as np
from numpy.linalg import eig, eigh
from scipy.linalg import expm  # not strictly needed but useful if you extend


def pauli_matrices():
    sigma_x = np.array([[0.0, 1.0],
                        [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j],
                        [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0],
                        [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def ladder_operators():
    # Basis |0> = [1, 0]^T (ground), |1> = [0, 1]^T (excited)
    sigma_minus = np.array([[0.0, 0.0],
                            [1.0, 0.0]], dtype=complex)
    sigma_plus = np.array([[0.0, 1.0],
                           [0.0, 0.0]], dtype=complex)
    return sigma_minus, sigma_plus


def build_gkls_superoperator(H, L_list):
    """
    Build GKLS superoperator K acting on vec(rho) with column stacking.

      d/dt vec(rho) = K vec(rho)

    H is 2x2 Hermitian.
    L_list is list of 2x2 Lindblad operators.
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    dim = N * N

    K = np.zeros((dim, dim), dtype=complex)

    # Hamiltonian part: -i [H, rho]
    I_N = np.eye(N, dtype=complex)
    K_H = -1j * (np.kron(I_N, H) - np.kron(H.T, I_N))
    K += K_H

    # Dissipative part: sum_a L_a rho L_a^\dagger - 0.5 {L_a^\dagger L_a, rho}
    for L in L_list:
        L = np.asarray(L, dtype=complex)
        Ld = L.conj().T
        LdL = Ld @ L
        term_jump = np.kron(L.conj(), L)
        term_left = np.kron(I_N, LdL)
        term_right = np.kron(LdL.T, I_N)
        K += term_jump - 0.5 * (term_left + term_right)

    return K


def vec(rho):
    """Vectorise rho in column stacking convention."""
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    """Unvectorise v back to N x N matrix in column stacking convention."""
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def find_stationary_state(K, N, tol=1e-12):
    """
    Find stationary density matrix rho_ss from GKLS generator K:

        K vec(rho_ss) = 0

    We pick the eigenvector with eigenvalue closest to zero, then
    hermitise and normalise to trace 1, and ensure positivity by
    clipping small negative eigenvalues.
    """
    evals, evecs = eig(K)
    idx = np.argmin(np.abs(evals))
    v_ss = evecs[:, idx]
    rho_ss = unvec(v_ss, N)
    # Hermitise
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    # Diagonalise to enforce positivity
    vals, vecs = eigh(rho_ss)
    vals_clipped = np.clip(vals.real, 0.0, None)
    if vals_clipped.sum() < tol:
        raise RuntimeError("Stationary state eigenvalues too small after clipping")
    vals_clipped = vals_clipped / vals_clipped.sum()
    rho_ss_pos = vecs @ np.diag(vals_clipped) @ vecs.conj().T
    # Final hermitisation
    rho_ss_pos = 0.5 * (rho_ss_pos + rho_ss_pos.conj().T)
    return rho_ss_pos


def bkm_weights(pi):
    """
    BKM (Kubo Mori) weights c_ij for Hessian of S(rho || diag(pi))
    at rho = diag(pi), in the eigenbasis of rho_ss.

    For 2 levels:

      c_ii = 1 / pi_i
      c_ij = (log pi_i - log pi_j) / (pi_i - pi_j)  for i != j
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


def classical_dirichlet(delta_p, pi, Q_eff):
    """
    Classical Fisher Dirichlet energy from an effective 2 state generator Q_eff.

    We reconstruct rates w[i,j] = Q_eff[j,i] and use

      E = 0.5 sum_{i != j} pi_i w_ij (phi_j - phi_i)^2

    with phi_i = delta_p_i / pi_i.
    """
    delta_p = np.asarray(delta_p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    Q_eff = np.asarray(Q_eff, dtype=float)
    N = len(pi)
    assert N == 2
    w = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                w[i, j] = Q_eff[j, i]

    phi = delta_p / pi
    E = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            E += 0.5 * pi[i] * w[i, j] * (phi[j] - phi[i]) ** 2
    return float(E)


def gkls_dirichlet_density(delta_p, G, M_diag, N=2):
    """
    GKLS density sector Dirichlet energy in rho_ss eigenbasis:

        E_GKLS = - <delta u, M G delta u>

    where delta u = vec(diag(delta_p)) in that basis, M is diagonal in the
    E_ij basis with entries M_diag, and G is the symmetric part of K^sharp.

    We assume N = 2, so vec(diag(delta_p)) has entries:
      [delta_p0, 0, 0, delta_p1] in column stacking.
    """
    delta_p = np.asarray(delta_p, dtype=float)
    delta_rho = np.diag(delta_p)
    delta_u = vec(delta_rho)  # length 4
    MG_du = M_diag * (G @ delta_u)
    E = -np.vdot(delta_u, MG_du).real
    return float(E)


def main():
    # Parameters for the driven qubit
    Omega = 1.0       # Rabi frequency
    Delta = 0.7       # detuning
    gamma = 1.0       # amplitude damping rate
    gamma_phi = 0.4   # dephasing rate

    N = 2

    print("22_gkls_nondiagonal_coherent_density_checks.py")
    print("------------------------------------------------")
    print(f"Omega        = {Omega:.3f}")
    print(f"Delta        = {Delta:.3f}")
    print(f"gamma        = {gamma:.3f}")
    print(f"gamma_phi    = {gamma_phi:.3f}")
    print()

    # Build H and Lindblad operators in computational basis
    sigma_x, sigma_y, sigma_z = pauli_matrices()
    sigma_minus, sigma_plus = ladder_operators()

    H = 0.5 * (Omega * sigma_x + Delta * sigma_z)
    L1 = np.sqrt(gamma) * sigma_minus
    L2 = np.sqrt(gamma_phi) * sigma_z
    L_list = [L1, L2]

    # GKLS generator in computational basis
    K = build_gkls_superoperator(H, L_list)

    # Stationary state
    rho_ss = find_stationary_state(K, N)
    print("Stationary state rho_ss in computational basis:")
    print(rho_ss)
    coh_mag = np.linalg.norm(rho_ss - np.diag(np.diag(rho_ss)))
    print(f"Off diagonal coherence norm (comp basis) = {coh_mag:.3e}")
    print()

    # Diagonalise rho_ss: rho_ss = U diag(pi) U^\dagger
    vals, U = eigh(rho_ss)
    pi = np.clip(vals.real, 0.0, None)
    pi = pi / pi.sum()
    print("Eigenvalues pi of rho_ss (stationary probabilities):")
    print(pi)
    print()

    # Transform K into rho_ss eigenbasis
    # vec(rho_comp) = (U* \otimes U) vec(rho_eig)
    # so K_eig = T^{-1} K T with T = U* \otimes U
    T = np.kron(U.conj(), U)
    T_inv = np.kron(U.T, U.conj().T)
    K_eig = T_inv @ K @ T

    # Check that diag(pi) is stationary in eigenbasis
    rho_ss_eig = np.diag(pi)
    stat_vec = K_eig @ vec(rho_ss_eig)
    stat_resid = np.linalg.norm(stat_vec)
    print(f"Stationarity residual in eigenbasis ||K_eig vec(diag(pi))|| = {stat_resid:.3e}")
    print()

    # Extract induced density generator Q_eff on diagonal subspace in eigenbasis
    # Basis of diagonals: E_11 and E_22.
    Q_eff = np.zeros((N, N), dtype=float)
    for j in range(N):
        p_basis = np.zeros(N, dtype=float)
        p_basis[j] = 1.0
        rho_basis = np.diag(p_basis)
        v_basis = vec(rho_basis)
        d_basis = K_eig @ v_basis
        drho = unvec(d_basis, N)
        dp = np.diag(drho).real
        Q_eff[:, j] = dp

    print("Induced density generator Q_eff in rho_ss eigenbasis (dp/dt = Q_eff p):")
    print(Q_eff)
    print()

    # Check generator properties
    col_sums = Q_eff.sum(axis=0)
    print(f"Column sums of Q_eff (should be near 0): {col_sums}")
    resid_Qpi = np.linalg.norm(Q_eff @ pi)
    print(f"Stationarity residual Q_eff pi: {resid_Qpi:.3e}")
    print()

    # Reconstruct rates and check detailed balance.
    # For two levels any irreducible Q is reversible.
    w_eff = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                w_eff[i, j] = Q_eff[j, i]

    print("Effective rates w_eff[i,j] = Q_eff[j,i]:")
    print(w_eff)
    print()

    db_resid = abs(pi[0] * w_eff[0, 1] - pi[1] * w_eff[1, 0])
    print(f"Detailed balance residual pi_0 w_01 - pi_1 w_10 = {db_resid:.3e}")
    print()

    # BKM metric in rho_ss eigenbasis
    C = bkm_weights(pi)
    # Vec ordering is column stacking, so M_diag corresponds to C reshaped in Fortran order.
    M_diag = C.reshape(-1, order="F")
    M_inv = 1.0 / M_diag

    # Metric adjoint and symmetric part G_eig for K_eig
    Kdag = K_eig.conj().T
    Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
    G_eig = 0.5 * (K_eig + Ksharp)

    # Compare GKLS density Dirichlet vs classical Fisher Dirichlet

    rng = np.random.default_rng(12345)
    n_tests = 50
    max_abs_err = 0.0
    max_rel_err = 0.0
    min_Ec = float("inf")
    min_Eg = float("inf")
    max_Ec = 0.0
    max_Eg = 0.0

    for _ in range(n_tests):
        # Random mass conserving perturbation delta p
        x = rng.normal(size=N)
        x -= x.mean()
        # Scale to a modest size to avoid numerical issues, but scale drops out since both are quadratic
        delta_p = x

        Ec = classical_dirichlet(delta_p, pi, Q_eff)
        Eg = gkls_dirichlet_density(delta_p, G_eig, M_diag, N=N)

        max_abs_err = max(max_abs_err, abs(Eg - Ec))
        if abs(Ec) > 1e-14:
            rel = abs(Eg - Ec) / abs(Ec)
            max_rel_err = max(max_rel_err, rel)

        min_Ec = min(min_Ec, Ec)
        min_Eg = min(min_Eg, Eg)
        max_Ec = max(max_Ec, Ec)
        max_Eg = max(max_Eg, Eg)

    print("Dirichlet comparison over random density perturbations:")
    print(f"  Max |E_GKLS - E_classical| = {max_abs_err:.3e}")
    print(f"  Max relative error         = {max_rel_err:.3e}")
    print(f"  Min classical Dirichlet    = {min_Ec:.3e}")
    print(f"  Min GKLS Dirichlet         = {min_Eg:.3e}")
    print(f"  Max classical Dirichlet    = {max_Ec:.3e}")
    print(f"  Max GKLS Dirichlet         = {max_Eg:.3e}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
    