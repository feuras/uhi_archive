#!/usr/bin/env python3
"""
09_gkls_K_splitting_qubit_checks.py

GKLS K-splitting for a thermal qubit with coherences.

Goal:
  Show that for a simple qubit GKLS model with a Hamiltonian H and thermal
  up/down jumps (plus optional dephasing), the linear generator K on the
  reduced real state vector

      u = (p0, p1, Re c, Im c)

  can be decomposed, with respect to a ρ_ss-weighted inner product, into

      K = G + J,

  where
    - G is the symmetric (dissipative) part and numerically matches the
      contribution from the GKLS dissipator alone,
    - J is the antisymmetric (reversible) part and numerically matches the
      contribution from the Hamiltonian commutator alone.

Model:
  - Two level system, basis |0>, |1|.
  - Hamiltonian H = (Delta/2) sigma_z.
  - Thermal stationary state ρ_ss = diag(pi0, pi1), where
        pi0 ∝ exp(+beta Delta/2),
        pi1 ∝ exp(-beta Delta/2).
  - Jump operators:
        L_down = sqrt(gamma_down) |0><1|
        L_up   = sqrt(gamma_up)   |1><0|
    with detailed balance gamma_up / gamma_down = exp(-beta Delta).
  - Optional pure dephasing L_phi = sqrt(gamma_phi) sigma_z.

State coordinates:
  - u0 = p0 = ρ_00 (population of |0>)
  - u1 = p1 = ρ_11 (population of |1>)
  - u2 = Re c = Re ρ_01
  - u3 = Im c = Im ρ_01

Inner product:
  - Define basis variations B_i = ∂ρ/∂u_i at ρ_ss.
  - Use ρ_ss-weighted Hilbert space inner product
        <A, B> = Tr(A ρ_ss^{-1} B),
    with ρ_ss^{-1} well defined since ρ_ss is full rank.
  - This induces an inner product on u space
        (x, y) = x^T M y,
    where M_{ij} = Tr(B_i ρ_ss^{-1} B_j).

Metric adjoint and K-splitting:
  - For generator K on u with inner product matrix M, the metric adjoint is
        K_sharp = M^{-1} K^T M.
  - The symmetric and antisymmetric parts with respect to this metric are
        G = 0.5 (K + K_sharp),
        J = 0.5 (K - K_sharp).

Checks:
  1) Build K_H from the Hamiltonian term alone, K_D from the dissipator alone.
  2) Build full K = K_H + K_D.
  3) Construct M at ρ_ss, compute K_sharp, G, J.
  4) Verify:
       - G ≈ K_D (symmetric part equals dissipative generator),
       - J ≈ K_H (antisymmetric part equals Hamiltonian generator),
       - G is symmetric and J is antisymmetric in the metric sense.

The script reports diagnostics and a final PASS / FAIL summary.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Basic qubit model: H, jump operators, GKLS pieces
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
    # Hamiltonian H = (Delta/2) sigma_z
    sigma_z = np.array([[1.0, 0.0],
                        [0.0, -1.0]], dtype=complex)
    H = 0.5 * Delta * sigma_z

    # Thermal stationary distribution for energy levels ±Delta/2
    # Energies: E0 = -Delta/2, E1 = +Delta/2
    E0 = -0.5 * Delta
    E1 = 0.5 * Delta
    w0 = np.exp(-beta * E0)
    w1 = np.exp(-beta * E1)
    Z = w0 + w1
    pi0 = w0 / Z
    pi1 = w1 / Z
    rho_ss = np.array([[pi0, 0.0],
                       [0.0, pi1]], dtype=complex)

    # Detailed balance for jumps between |1> and |0|
    # gamma_up / gamma_down = exp(-beta Delta) = pi1 / pi0
    # Choose gamma_down and gamma_up consistent with this ratio.
    gamma_down = gamma_down_base
    gamma_up = gamma_down * np.exp(-beta * Delta)

    # Jump operators
    # L_down = sqrt(gamma_down) |0><1|
    # L_up   = sqrt(gamma_up)   |1><0|
    L_down = np.zeros((2, 2), dtype=complex)
    L_down[0, 1] = np.sqrt(gamma_down)

    L_up = np.zeros((2, 2), dtype=complex)
    L_up[1, 0] = np.sqrt(gamma_up)

    L_ops = [L_down, L_up]

    # Optional pure dephasing L_phi = sqrt(gamma_phi) sigma_z
    if gamma_phi > 0.0:
        L_phi = np.sqrt(gamma_phi) * sigma_z
        L_ops.append(L_phi)

    return H, rho_ss, L_ops


def apply_Hamiltonian(H, rho):
    """
    Hamiltonian part of GKLS generator:
        dρ/dt = -i [H, ρ]
    """
    return -1j * (H @ rho - rho @ H)


def apply_dissipator(L_ops, rho):
    """
    GKLS dissipator:
        D(ρ) = sum_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ})

    rho is 2x2 complex.
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


# ---------------------------------------------------------------------------
# Coordinate maps u <-> rho
# ---------------------------------------------------------------------------

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
    Build the 4x4 real generator K_real on u space so that:

        du/dt = K_real u

    by applying the GKLS generator to basis u vectors.
    """
    dim_u = 4
    K_total = np.zeros((dim_u, dim_u), dtype=float)
    K_H = np.zeros((dim_u, dim_u), dtype=float)
    K_D = np.zeros((dim_u, dim_u), dtype=float)

    for j in range(dim_u):
        # Basis vector e_j
        e_j = np.zeros(dim_u, dtype=float)
        e_j[j] = 1.0

        # Corresponding rho
        rho_j = u_to_rho(e_j)

        # Hamiltonian part
        d_rho_H = apply_Hamiltonian(H, rho_j)
        du_H = rho_to_u(d_rho_H)

        # Dissipator part
        d_rho_D = apply_dissipator(L_ops, rho_j)
        du_D = rho_to_u(d_rho_D)

        # Total
        du_total = du_H + du_D

        K_H[:, j] = du_H
        K_D[:, j] = du_D
        K_total[:, j] = du_total

    return K_total, K_H, K_D


# ---------------------------------------------------------------------------
# Metric induced by rho_ss and basis variations B_i = ∂rho/∂u_i
# ---------------------------------------------------------------------------

def build_metric_matrix(rho_ss):
    """
    Build 4x4 metric matrix M on u space from the inner product

        <A, B> = Tr(A rho_ss^{-1} B),

    evaluated on basis variations B_i = ∂ρ/∂u_i.

    For mapping u -> rho given by u_to_rho, the basis variations at rho_ss are:

      B0 = ∂ρ/∂p0 = [[1, 0],
                     [0, 0]]

      B1 = ∂ρ/∂p1 = [[0, 0],
                     [0, 1]]

      B2 = ∂ρ/∂x  = [[0, 1],
                     [1, 0]]  (sigma_x)

      B3 = ∂ρ/∂y  = [[0, i],
                     [-i, 0]] (sigma_y)

    We then set M_{ij} = Tr(B_i rho_ss^{-1} B_j).
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
    Metric adjoint of K with respect to inner product (x, y) = x^T M y.

    For du/dt = K u, the adjoint K_sharp satisfies:
        (x, K y) = (K_sharp x, y)
    which leads to:
        K_sharp = M^{-1} K^T M.
    """
    Minv = np.linalg.inv(M)
    return Minv @ K.T @ M


# ---------------------------------------------------------------------------
# Main test: K-splitting checks
# ---------------------------------------------------------------------------

def run():
    print("GKLS K-splitting for coherent thermal qubit")
    print("-------------------------------------------")

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

    print("Stationary state rho_ss (should be diagonal thermal):")
    print(rho_ss)
    print("")

    # Check stationarity: GKLS applied to rho_ss should vanish
    d_rho_H_ss = apply_Hamiltonian(H, rho_ss)
    d_rho_D_ss = apply_dissipator(L_ops, rho_ss)
    d_rho_ss = d_rho_H_ss + d_rho_D_ss

    norm_H_ss = float(np.linalg.norm(d_rho_H_ss, ord="fro"))
    norm_D_ss = float(np.linalg.norm(d_rho_D_ss, ord="fro"))
    norm_total_ss = float(np.linalg.norm(d_rho_ss, ord="fro"))

    print("Stationarity diagnostics:")
    print(f"  ||-i[H, rho_ss]||_F ≈ {norm_H_ss:.3e}")
    print(f"  ||D(rho_ss)||_F     ≈ {norm_D_ss:.3e}")
    print(f"  ||GKLS(rho_ss)||_F  ≈ {norm_total_ss:.3e} (should be near 0)")
    print("")

    # Build real generators K_total, K_H, K_D on u space
    K_total, K_H, K_D = build_generator_real(H, L_ops)

    print("Real generator matrices on u = (p0, p1, Re c, Im c):")
    print("  K_H (Hamiltonian part):")
    print(K_H)
    print("")
    print("  K_D (dissipative part):")
    print(K_D)
    print("")
    print("  K_total = K_H + K_D:")
    print(K_total)
    print("")

    # Metric matrix M from rho_ss
    M = build_metric_matrix(rho_ss)
    print("Metric matrix M induced by rho_ss:")
    print(M)
    print("")

    # Metric adjoint of K_total
    K_sharp = metric_adjoint(K_total, M)

    # Symmetric and antisymmetric parts with respect to M
    G = 0.5 * (K_total + K_sharp)
    J = 0.5 * (K_total - K_sharp)

    # Helper for Frobenius norm
    frob = lambda A: float(np.linalg.norm(A, ord="fro"))

    # Compare G to K_D and J to K_H
    diff_G_D = G - K_D
    diff_J_H = J - K_H

    rel_err_G_D = frob(diff_G_D) / max(frob(K_D), 1e-16)
    rel_err_J_H = frob(diff_J_H) / max(frob(K_H), 1e-16)

    print("K-splitting diagnostics:")
    print(f"  ||G - K_D||_F / ||K_D||_F ≈ {rel_err_G_D:.3e}")
    print(f"  ||J - K_H||_F / ||K_H||_F ≈ {rel_err_J_H:.3e}")
    print("")

    # Check metric symmetry properties explicitly
    G_sharp = metric_adjoint(G, M)
    J_sharp = metric_adjoint(J, M)

    sym_resid_G = frob(G - G_sharp) / max(frob(G), 1e-16)
    skew_resid_J = frob(J + J_sharp) / max(frob(J), 1e-16)

    print("Metric symmetry diagnostics:")
    print(f"  G symmetric in metric?  ||G - G_sharp||/||G|| ≈ {sym_resid_G:.3e}")
    print(f"  J antisymmetric?        ||J + J_sharp||/||J|| ≈ {skew_resid_J:.3e}")
    print("")

    # PASS / FAIL criteria
    tol_stationary = 1e-10
    tol_split = 1e-10
    tol_sym = 1e-10

    pass_stationary = norm_total_ss < tol_stationary
    pass_G = rel_err_G_D < tol_split
    pass_J = rel_err_J_H < tol_split
    pass_sym = sym_resid_G < tol_sym and skew_resid_J < tol_sym

    print("Summary:")
    print(f"  rho_ss stationary for full GKLS?          {pass_stationary} (tol = {tol_stationary})")
    print(f"  Symmetric part G ≈ dissipative K_D?       {pass_G} (tol = {tol_split})")
    print(f"  Antisymmetric part J ≈ Hamiltonian K_H?   {pass_J} (tol = {tol_split})")
    print(f"  G, J satisfy metric symmetry conditions?  {pass_sym} (tol = {tol_sym})")
    print("")

    if pass_stationary and pass_G and pass_J and pass_sym:
        print("GKLS K-splitting CHECK: PASS")
        print("  The reduced real generator K on (p0, p1, Re c, Im c), when")
        print("  decomposed using the rho_ss-weighted metric M, splits cleanly")
        print("  into a symmetric part G that matches the GKLS dissipator and")
        print("  an antisymmetric part J that matches the Hamiltonian commutator.")
        print("  This realises the K = G + J metriplectic splitting explicitly")
        print("  in a coherent quantum model.")
    else:
        print("GKLS K-splitting CHECK: FAIL (see diagnostics above).")


if __name__ == "__main__":
    run()
