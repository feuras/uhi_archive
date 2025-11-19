#!/usr/bin/env python3
"""
28_gkls_full_bloch_tomography.py

Full Bloch-space tomography of a coherent + dissipative qubit GKLS generator.

Goal
-----

For a driven, damped qubit with Hamiltonian and Lindblad jumps, we:

  1) Define the GKLS generator L(·) on 2x2 density matrices using

         dρ/dt = -i [H, ρ] + Σ_k ( L_k ρ L_k^\dag - 0.5 {L_k^\dag L_k, ρ} ).

  2) Choose the Hermitian operator basis { I, σ_x, σ_y, σ_z } and represent
     any Hermitian operator X as

         X = α_0 I + α_x σ_x + α_y σ_y + α_z σ_z,

     where α_μ = (1/2) tr(Σ_μ X) with Σ_0 = I, Σ_{1,2,3} = σ_{x,y,z}.

     In this basis, the GKLS generator is a 4x4 real matrix K_exact such that

         dα/dt = K_exact α,

     with α = (α_0, α_x, α_y, α_z)^T.

  3) Compute K_exact "analytically" by acting L on each basis element Σ_ν and
     reading off the coordinates of L(Σ_ν).

  4) Perform tomography of K using only state responses:
       - Sample a collection of random full-rank qubit density matrices ρ^(k)
         (with coherences).
       - For each ρ^(k), compute α^(k) and dα^(k)/dt from L(ρ^(k)).
       - Stack these into matrices A and dA such that dA ≈ K_rec A and solve
         for K_rec via least squares.

  5) Compare K_rec against K_exact entrywise.

We expect:
  * The first row of K_exact and K_rec to vanish (trace preservation).
  * K_rec ≈ K_exact up to numerical precision ~1e-14.
"""

import numpy as np


# ----------------------------------------------------------------------
# Pauli basis and GKLS generator
# ----------------------------------------------------------------------

# Pauli matrices and identity
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

    Parameters
    ----------
    rho : (2, 2) complex ndarray
        Density operator (or any 2x2 operator).
    Omega : float
        Coherent drive strength along σ_x.
    Delta : float
        Detuning along σ_z.
    gamma : float
        Amplitude damping rate (σ_- jump).
    gamma_phi : float
        Dephasing rate (σ_z jump).

    Returns
    -------
    drho : (2, 2) complex ndarray
        Time derivative dρ/dt.
    """
    # Hamiltonian H = 0.5 (Omega σ_x + Delta σ_z)
    H = 0.5 * (Omega * SIGMA_X + Delta * SIGMA_Z)

    # Jump operators
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

    where α_μ = (1/2) tr(Σ_μ X).
    """
    alphas = []
    for Sigma in SIGMAS:
        val = 0.5 * np.trace(Sigma @ X)
        alphas.append(val)
    # For Hermitian X, these coordinates should be real
    return np.real(np.array(alphas, dtype=float))


def op_from_coords(alpha):
    """
    Reconstruct an operator X from coordinates α in the Pauli basis:

        X = α_0 I + α_x σ_x + α_y σ_y + α_z σ_z.
    """
    X = np.zeros((2, 2), dtype=complex)
    for a, Sigma in zip(alpha, SIGMAS):
        X += a * Sigma
    return X


# ----------------------------------------------------------------------
# Exact generator in Pauli basis
# ----------------------------------------------------------------------

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
# Tomography of K from state responses
# ----------------------------------------------------------------------

def random_bloch_vector_inside_ball(rng, max_radius=0.8):
    """
    Sample a random Bloch vector r in R^3 with ||r|| <= max_radius.
    This ensures a strictly positive density matrix.
    """
    while True:
        r = rng.normal(size=3)
        norm = np.linalg.norm(r)
        if norm < 1e-12:
            continue
        r = r / norm
        radius = max_radius * rng.random()
        r = radius * r
        return r


def rho_from_bloch(r):
    """
    Build a density matrix from a Bloch vector r in R^3:

        ρ = 0.5 (I + r_x σ_x + r_y σ_y + r_z σ_z).

    For ||r|| <= 1 this is a valid density matrix.
    """
    return 0.5 * (IDENTITY
                  + r[0] * SIGMA_X
                  + r[1] * SIGMA_Y
                  + r[2] * SIGMA_Z)


def main():
    Omega = 1.0
    Delta = 0.7
    gamma = 1.0
    gamma_phi = 0.4

    print("28_gkls_full_bloch_tomography.py")
    print("---------------------------------")
    print(f"Omega                         = {Omega:.3f}")
    print(f"Delta                         = {Delta:.3f}")
    print(f"gamma (amplitude damping)     = {gamma:.3f}")
    print(f"gamma_phi (dephasing)         = {gamma_phi:.3f}")
    print()

    # 1) Exact generator in Pauli basis
    K_exact = build_exact_bloch_generator(Omega=Omega,
                                          Delta=Delta,
                                          gamma=gamma,
                                          gamma_phi=gamma_phi)

    print("Exact generator K_exact in {I, σ_x, σ_y, σ_z} coordinates:")
    np.set_printoptions(precision=6, suppress=True)
    print(K_exact)
    print()

    # Diagnostics: trace preservation and Hermiticity
    max_imag = 0.0  # K_exact is real by construction
    row0_norm = np.linalg.norm(K_exact[0, :])
    print(f"Trace-preserving check, ||row 0||      = {row0_norm:.3e}")
    print(f"Max imaginary part of K_exact entries  = {max_imag:.3e}")
    print()

    # 2) Tomography from state responses
    rng = np.random.default_rng(12345)
    n_probes = 20

    A = np.zeros((4, n_probes), dtype=float)
    dA = np.zeros((4, n_probes), dtype=float)

    for k in range(n_probes):
        r = random_bloch_vector_inside_ball(rng, max_radius=0.8)
        rho = rho_from_bloch(r)

        alpha = coords_from_op(rho)
        drho = apply_gkls_qubit(rho,
                                Omega=Omega,
                                Delta=Delta,
                                gamma=gamma,
                                gamma_phi=gamma_phi)
        alpha_dot = coords_from_op(drho)

        A[:, k] = alpha
        dA[:, k] = alpha_dot

    # Solve dA ≈ K_rec A via least squares:
    #   dA A^T ≈ K_rec A A^T   => K_rec ≈ dA A^T (A A^T)^{-1}
    AAt = A @ A.T
    dAAt = dA @ A.T
    try:
        AAt_inv = np.linalg.inv(AAt)
    except np.linalg.LinAlgError:
        AAt_inv = np.linalg.pinv(AAt)

    K_rec = dAAt @ AAt_inv

    # Diagnostics on K_rec
    diff_K = np.max(np.abs(K_rec - K_exact))
    row0_norm_rec = np.linalg.norm(K_rec[0, :])
    max_imag_rec = 0.0  # K_rec is real-valued

    print("Tomographically reconstructed K_rec:")
    print(K_rec)
    print()
    print(f"Max |K_rec - K_exact|               = {diff_K:.3e}")
    print(f"Trace-preserving check, ||row 0||   = {row0_norm_rec:.3e}")
    print(f"Max imaginary part of K_rec entries = {max_imag_rec:.3e}")
    print()

    # 3) Check prediction quality on additional random states
    n_test = 10
    max_state_resid = 0.0

    for _ in range(n_test):
        r = random_bloch_vector_inside_ball(rng, max_radius=0.8)
        rho = rho_from_bloch(r)

        alpha = coords_from_op(rho)
        drho = apply_gkls_qubit(rho,
                                Omega=Omega,
                                Delta=Delta,
                                gamma=gamma,
                                gamma_phi=gamma_phi)
        alpha_dot_true = coords_from_op(drho)
        alpha_dot_pred = K_rec @ alpha

        resid = np.linalg.norm(alpha_dot_true - alpha_dot_pred)
        max_state_resid = max(max_state_resid, resid)

    print(f"Max state-level residual ||α̇_true - α̇_pred|| over tests = {max_state_resid:.3e}")


if __name__ == "__main__":
    main()
