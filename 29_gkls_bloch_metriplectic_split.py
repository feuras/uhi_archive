#!/usr/bin/env python3
"""
29_gkls_bloch_metriplectic_split.py

Metriplectic split of a driven, damped qubit GKLS generator in Bloch space.

We take the same qubit GKLS model as in script 28:

    H = 0.5 (Omega σ_x + Delta σ_z)
    L_1   = sqrt(gamma) σ_-
    L_phi = sqrt(gamma_phi) σ_z

and:

  1) Build the exact 4x4 real generator K_exact in the Pauli basis
     {I, σ_x, σ_y, σ_z}, acting on coordinates α with dα/dt = K_exact α.

  2) Solve for the stationary Bloch coordinates α_ss, reconstruct the stationary
     density matrix ρ_ss, and diagonalise it to obtain eigenvalues λ and
     eigenvectors U.

  3) Construct the BKM metric at ρ_ss in the eigenbasis, where the metric on
     matrix elements in the basis |m><n| is diagonal with weights

         c_{mn} =
             1 / λ_m,                     if m = n,
             (log λ_m - log λ_n)/(λ_m - λ_n),  if m ≠ n.

     This gives a diagonal metric diag(C_flat) on vec space in the eigenbasis.

  4) Transform the Pauli basis {I, σ_x, σ_y, σ_z} to the eigenbasis of ρ_ss,
     vectorise, and build the 4x4 metric matrix M in Bloch coordinates via

         M_ab = <Σ_a, Σ_b>_BKM
              = vec(Σ_a')^† diag(C_flat) vec(Σ_b'),

     where Σ_a' = U^† Σ_a U.

  5) Compute the metric adjoint of K_exact with respect to M,

         K_sharp = M^{-1} K_exact^T M,

     and split

         G = 0.5 (K_exact + K_sharp),
         J = 0.5 (K_exact - K_sharp).

  6) Check the defining properties:

       - K_exact = G + J,
       - M G is symmetric,
       - M J is skew-symmetric,

     and print the resulting matrices and diagnostic norms.

This script shows explicitly that the full qubit GKLS generator admits a
metriplectic split K = G + J in Bloch space with respect to the BKM metric
at the stationary state, extending the density-sector results to the full
operator space including coherences.
"""

import numpy as np


# ----------------------------------------------------------------------
# Pauli basis and GKLS generator (as in script 28)
# ----------------------------------------------------------------------

SIGMA_X = np.array([[0.0, 1.0],
                    [1.0, 0.0]], dtype=complex)

SIGMA_Y = np.array([[0.0, -1.0j],
                    [1.0j, 0.0]], dtype=complex)

SIGMA_Z = np.array([[1.0, 0.0],
                    [0.0, -1.0]], dtype=complex)

IDENTITY = np.eye(2, dtype=complex)

SIGMAS = [IDENTITY, SIGMA_X, SIGMA_Y, SIGMA_Z]


def apply_gkls_qubit(rho, Omega=1.0, Delta=0.7, gamma=1.0, gamma_phi=0.4):
    """
    Apply the driven, damped qubit GKLS generator to a 2x2 density matrix rho.

        dρ/dt = -i [H, ρ]
                + Σ_k ( L_k ρ L_k^\dag - 0.5 {L_k^\dag L_k, ρ} ).
    """
    H = 0.5 * (Omega * SIGMA_X + Delta * SIGMA_Z)

    sigma_minus = np.array([[0.0, 0.0],
                            [1.0, 0.0]], dtype=complex)
    L1 = np.sqrt(gamma) * sigma_minus
    Lphi = np.sqrt(gamma_phi) * SIGMA_Z

    drho = -1j * (H @ rho - rho @ H)

    for L in [L1, Lphi]:
        LdL = L.conj().T @ L
        drho += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)

    return drho


def coords_from_op(X):
    """
    Represent a Hermitian operator X in the {I, σ_x, σ_y, σ_z} basis:

        X = α_0 I + α_x σ_x + α_y σ_y + α_z σ_z,

    with α_μ = 0.5 tr(Σ_μ X). For Hermitian X these are real.
    """
    alphas = []
    for Sigma in SIGMAS:
        val = 0.5 * np.trace(Sigma @ X)
        alphas.append(val)
    # Discard tiny imaginary parts from numerical noise
    arr = np.array(alphas, dtype=complex)
    return arr.real


def op_from_coords(alpha):
    """
    Reconstruct an operator X from coordinates α in the Pauli basis:

        X = α_0 I + α_x σ_x + α_y σ_y + α_z σ_z.
    """
    X = np.zeros((2, 2), dtype=complex)
    for a, Sigma in zip(alpha, SIGMAS):
        X += a * Sigma
    return X


def vec(mat):
    """Column-stack a 2x2 matrix."""
    return np.asarray(mat, dtype=complex).reshape(-1, order="F")


def build_exact_bloch_generator(Omega=1.0, Delta=0.7, gamma=1.0, gamma_phi=0.4):
    """
    Build the exact 4x4 generator K_exact in the Pauli basis by acting
    with the GKLS generator on the basis elements {I, σ_x, σ_y, σ_z}.
    """
    K_exact = np.zeros((4, 4), dtype=float)

    for j, Sigma_j in enumerate(SIGMAS):
        dSigma_j = apply_gkls_qubit(Sigma_j,
                                    Omega=Omega,
                                    Delta=Delta,
                                    gamma=gamma,
                                    gamma_phi=gamma_phi)
        alpha_dot = coords_from_op(dSigma_j)
        K_exact[:, j] = alpha_dot

    return K_exact


# ----------------------------------------------------------------------
# Main metriplectic split
# ----------------------------------------------------------------------

def main():
    Omega = 1.0
    Delta = 0.7
    gamma = 1.0
    gamma_phi = 0.4

    print("29_gkls_bloch_metriplectic_split.py")
    print("------------------------------------")
    print(f"Omega                         = {Omega:.3f}")
    print(f"Delta                         = {Delta:.3f}")
    print(f"gamma (amplitude damping)     = {gamma:.3f}")
    print(f"gamma_phi (dephasing)         = {gamma_phi:.3f}")
    print()

    # 1) Exact Bloch generator
    K_exact = build_exact_bloch_generator(Omega=Omega,
                                          Delta=Delta,
                                          gamma=gamma,
                                          gamma_phi=gamma_phi)

    np.set_printoptions(precision=6, suppress=True)
    print("Exact generator K_exact in {I, σ_x, σ_y, σ_z} coordinates:")
    print(K_exact)
    print()

    # Check trace-preserving structure
    row0_norm = np.linalg.norm(K_exact[0, :])
    print(f"Trace-preserving check, ||row 0||      = {row0_norm:.3e}")
    print()

    # 2) Stationary Bloch coordinates α_ss and stationary state ρ_ss
    alpha0_ss = 0.5  # tr(ρ)/2 = 1/2 for any density matrix

    B = K_exact[1:, 1:]     # 3x3 block on (σ_x, σ_y, σ_z)
    c_vec = K_exact[1:, 0]  # driving from α_0 into traceless part

    r_ss = -np.linalg.solve(B, c_vec * alpha0_ss)
    alpha_ss = np.concatenate([[alpha0_ss], r_ss])

    rho_ss = op_from_coords(alpha_ss)
    print("Stationary Bloch coordinates α_ss:")
    print(alpha_ss)
    print()
    print("Stationary state ρ_ss in computational basis:")
    print(rho_ss)
    print(f"trace(ρ_ss)                         = {np.trace(rho_ss):.6f}")
    print()

    evals, U = np.linalg.eigh(rho_ss)
    print("Eigenvalues of ρ_ss (λ):")
    print(evals)
    print()

    # 3) BKM metric weights in eigenbasis of ρ_ss
    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            if abs(evals[i] - evals[j]) < 1e-14:
                C[i, j] = 1.0 / evals[i]
            else:
                C[i, j] = (np.log(evals[i]) - np.log(evals[j])) / (evals[i] - evals[j])

    C_flat = C.reshape(-1, order="F")
    print("BKM weights C_{mn} in eigenbasis of ρ_ss:")
    print(C)
    print()

    # 4) Metric matrix M in Pauli/Bloch coordinates
    M = np.zeros((4, 4), dtype=float)
    u_vecs = []

    # Precompute transformed, vectorised Pauli basis in eigenbasis
    for a, Sigma_a in enumerate(SIGMAS):
        Sigma_a_eig = U.conj().T @ Sigma_a @ U
        u_a = vec(Sigma_a_eig)
        u_vecs.append(u_a)

    for a in range(4):
        u_a = u_vecs[a]
        for b in range(4):
            u_b = u_vecs[b]
            # BKM inner product: <Σ_a, Σ_b> = Σ_{mn} C_{mn} Σ_a'^* Σ_b'
            val = np.vdot(u_a * C_flat, u_b)
            M[a, b] = val.real

    print("BKM metric matrix M in {I, σ_x, σ_y, σ_z} coordinates:")
    print(M)
    print(f"M symmetric?                        = {np.allclose(M, M.T)}")
    print(f"Eigenvalues of M                    = {np.linalg.eigvals(M)}")
    print()

    # 5) Metric adjoint, G and J
    M_inv = np.linalg.inv(M)
    K_sharp = M_inv @ K_exact.T @ M

    G = 0.5 * (K_exact + K_sharp)
    J = 0.5 * (K_exact - K_sharp)

    print("Metric adjoint K_sharp = M^{-1} K^T M:")
    print(K_sharp)
    print()

    print("Symmetric part G (dissipative channel):")
    print(G)
    print()

    print("Skew part J (reversible channel):")
    print(J)
    print()

    # 6) Metriplectic diagnostics
    MG = M @ G
    MJ = M @ J

    sym_MG = np.linalg.norm(MG - MG.T)
    skew_MJ = np.linalg.norm(MJ + MJ.T)
    K_reconstructed = G + J
    diff_K = np.linalg.norm(K_reconstructed - K_exact)

    print("Diagnostics:")
    print(f"  ||K_exact - (G + J)||             = {diff_K:.3e}")
    print(f"  ||M G - (M G)^T|| (should be ~0)  = {sym_MG:.3e}")
    print(f"  ||M J + (M J)^T|| (should be ~0)  = {skew_MJ:.3e}")
    print()

    # Optional: traceless sub-block
    G_tr = G[1:, 1:]
    J_tr = J[1:, 1:]

    print("G restricted to traceless subspace {σ_x, σ_y, σ_z}:")
    print(G_tr)
    print()
    print("J restricted to traceless subspace {σ_x, σ_y, σ_z}:")
    print(J_tr)
    print()


if __name__ == "__main__":
    main()
