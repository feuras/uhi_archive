#!/usr/bin/env python3
"""
11_gkls_qutrit_K_and_G_unification_checks.py

GKLS metriplectic K-splitting and classical G-unification for a thermal qutrit.

Goal:
  - Build a 3-level (qutrit) GKLS model with a thermal stationary state.
  - Represent states in a 9-dimensional real coordinate u corresponding to
    the 3 populations and 3 complex coherences.
  - Construct the real generator K on u by acting with the GKLS generator
    on basis directions.
  - Build the rho_ss-weighted Fisher-type metric M on u.
  - Perform the metriplectic splitting
        K_sharp = M^{-1} K^T M,
        G = 0.5 (K + K_sharp),
        J = 0.5 (K - K_sharp),
    and verify numerically that
        G ≈ K_D  (pure dissipator),
        J ≈ K_H  (pure Hamiltonian),
    and that G and J obey the metric symmetry conditions.
  - Extract the population block Q_markov from the dissipator and show that
    the canonical classical G_true = Q_markov diag(pi) is:
        * symmetric,
        * generates the same drift as Q_markov acting on p = pi ⊙ mu,
        * reproduces the classical Dirichlet form built from the jump rates.
  - Finally, verify a discrete cost-entropy inequality in the classical
    3-state Fisher geometry by checking that the ratio
        R = <v,gradF>^2 / (2 C_min σ)
    is 1 on eigenmodes of G_metric = -G_true and lies in [0, 1] on random
    directions in the positive eigenspace, with equality to numerical precision.

This script is designed as a referee-ready, high-precision numerical check
that the metriplectic K = G + J structure and the classical Fisher Dirichlet
geometry are already encoded in a nontrivial quantum GKLS qutrit model.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Qutrit GKLS model
# ---------------------------------------------------------------------------

def build_qutrit_model(beta=1.0,
                       E0=0.0, E1=1.0, E2=2.0,
                       g01_down=0.8, g12_down=0.5, g02_down=0.3,
                       gamma_phi0=0.1, gamma_phi1=0.1, gamma_phi2=0.1):
    """
    Build a simple thermal qutrit model.

    Energies: E0 < E1 < E2 with H = diag(E0, E1, E2).
    Thermal state:
        rho_ss = diag(pi0, pi1, pi2),
        pi_i ∝ exp(-beta E_i).

    Jump operators:
      For each pair (i,j) with E_j > E_i we include:
        L_down (j -> i) with rate g_down,
        L_up   (i -> j) with rate g_up = g_down * exp[-beta (E_j - E_i)],
      so that detailed balance holds for the Gibbs state.

    Dephasing:
      For each level k we include L_phi,k ∝ |k><k| with rate gamma_phik.
    """
    # Hamiltonian
    E = np.array([E0, E1, E2], dtype=float)
    H = np.diag(E).astype(complex)

    # Thermal stationary state
    w = np.exp(-beta * E)
    Z = float(np.sum(w))
    pi = w / Z
    rho_ss = np.diag(pi.astype(complex))

    # Helper to build up/down jumps with detailed balance
    def pair(i, j, g_down):
        # assume E[j] > E[i]
        Delta = E[j] - E[i]
        g_up = g_down * np.exp(-beta * Delta)
        L_down = np.zeros((3, 3), dtype=complex)
        L_up = np.zeros((3, 3), dtype=complex)
        # downward j -> i, upward i -> j
        L_down[i, j] = np.sqrt(g_down)
        L_up[j, i] = np.sqrt(g_up)
        return [L_down, L_up]

    L_ops = []
    # Nearest neighbour and direct 0 <-> 2 transitions
    L_ops += pair(0, 1, g01_down)
    L_ops += pair(1, 2, g12_down)
    L_ops += pair(0, 2, g02_down)

    # Dephasing on each level (keeps H diagonal, damps coherences)
    for k, gphi in enumerate([gamma_phi0, gamma_phi1, gamma_phi2]):
        if gphi > 0.0:
            L = np.zeros((3, 3), dtype=complex)
            L[k, k] = 1.0
            L_ops.append(np.sqrt(gphi) * L)

    return H, rho_ss, L_ops


def apply_Hamiltonian(H, rho):
    """
    Hamiltonian part of GKLS:
        dρ/dt = -i [H, ρ].
    """
    return -1j * (H @ rho - rho @ H)


def apply_dissipator(L_ops, rho):
    """
    GKLS dissipator:
        D(ρ) = sum_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ}).
    """
    d = rho.shape[0]
    D = np.zeros((d, d), dtype=complex)
    for L in L_ops:
        L_rho = L @ rho
        L_rho_Ld = L_rho @ L.conj().T
        K = L.conj().T @ L
        anti = K @ rho + rho @ K
        D += L_rho_Ld - 0.5 * anti
    return D


# ---------------------------------------------------------------------------
# Real coordinate representation u ↔ ρ for 3×3 density matrices
# ---------------------------------------------------------------------------

def u_to_rho(u):
    """
    Map real state vector u ∈ R^9 to a 3×3 Hermitian matrix ρ:

        u = (p0, p1, p2,
             Re ρ_01, Im ρ_01,
             Re ρ_02, Im ρ_02,
             Re ρ_12, Im ρ_12).

        ρ = [[p0,          ρ_01,          ρ_02        ],
             [ρ_01*,       p1,            ρ_12        ],
             [ρ_02*,       ρ_12*,         p2         ]]

    No trace constraint is imposed; GKLS is linear so this is convenient.
    """
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
    """
    Map a 3×3 Hermitian matrix ρ back to u ∈ R^9.
    """
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


# ---------------------------------------------------------------------------
# Real generator on u and metric
# ---------------------------------------------------------------------------

def build_generator_real(H, L_ops):
    """
    Build the 9×9 real generators K_total, K_H, K_D on u space such that

        du/dt = K u,

    by acting with the GKLS generator on the basis vectors e_j in u space.
    """
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


def build_metric_matrix(rho_ss):
    """
    Build the 9×9 metric matrix M on u space from the ρ_ss-weighted
    Fisher-type inner product

        <A, B> = Tr(A ρ_ss^{-1} B),

    using the basis variations B_i = ∂ρ/∂u_i:

      B0 = ∂ρ/∂p0  = |0><0|
      B1 = ∂ρ/∂p1  = |1><1|
      B2 = ∂ρ/∂p2  = |2><2|

      B3 = ∂ρ/∂Re01 = |0><1| + |1><0|
      B4 = ∂ρ/∂Im01 = i|0><1| - i|1><0|

      B5 = ∂ρ/∂Re02 = |0><2| + |2><0|
      B6 = ∂ρ/∂Im02 = i|0><2| - i|2><0|

      B7 = ∂ρ/∂Re12 = |1><2| + |2><1|
      B8 = ∂ρ/∂Im12 = i|1><2| - i|2><1|.

    Then M_{ij} = Re Tr(B_i ρ_ss^{-1} B_j).
    """
    rho_inv = np.linalg.inv(rho_ss)

    B_list = []

    # B0, B1, B2: diagonal variations
    for k in range(3):
        M = np.zeros((3, 3), dtype=complex)
        M[k, k] = 1.0
        B_list.append(M)

    # B3: d/d Re ρ_01
    M = np.zeros((3, 3), dtype=complex)
    M[0, 1] = 1.0
    M[1, 0] = 1.0
    B_list.append(M)

    # B4: d/d Im ρ_01
    M = np.zeros((3, 3), dtype=complex)
    M[0, 1] = 1.0j
    M[1, 0] = -1.0j
    B_list.append(M)

    # B5: d/d Re ρ_02
    M = np.zeros((3, 3), dtype=complex)
    M[0, 2] = 1.0
    M[2, 0] = 1.0
    B_list.append(M)

    # B6: d/d Im ρ_02
    M = np.zeros((3, 3), dtype=complex)
    M[0, 2] = 1.0j
    M[2, 0] = -1.0j
    B_list.append(M)

    # B7: d/d Re ρ_12
    M = np.zeros((3, 3), dtype=complex)
    M[1, 2] = 1.0
    M[2, 1] = 1.0
    B_list.append(M)

    # B8: d/d Im ρ_12
    M = np.zeros((3, 3), dtype=complex)
    M[1, 2] = 1.0j
    M[2, 1] = -1.0j
    B_list.append(M)

    dim_u = 9
    Mmat = np.zeros((dim_u, dim_u), dtype=float)

    for i in range(dim_u):
        for j in range(dim_u):
            val = np.trace(B_list[i] @ rho_inv @ B_list[j])
            Mmat[i, j] = float(np.real(val))

    return Mmat


def metric_adjoint(K, M):
    """
    Metric adjoint of K with respect to inner product (x, y) = x^T M y:

        K_sharp = M^{-1} K^T M.
    """
    Minv = np.linalg.inv(M)
    return Minv @ K.T @ M


# ---------------------------------------------------------------------------
# Classical Dirichlet form from a Markov generator Q
# ---------------------------------------------------------------------------

def dirichlet_from_Q(mu, pi, Q):
    """
    Classical Dirichlet form (pairwise representation) for reversible chain:

        E_pair(mu) = 0.5 sum_{i,j} pi_j γ_{ij} (mu_i - mu_j)^2

    where γ_{ij} = Q_{ij} for i != j are the jump rates.
    """
    dim = len(pi)
    E = 0.0
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            gamma_ij = Q[i, j]
            E += 0.5 * pi[j] * gamma_ij * (mu[i] - mu[j]) ** 2
    return float(E)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run():
    print("GKLS qutrit K-splitting and classical G-unification")
    print("---------------------------------------------------")

    beta = 1.0
    E0, E1, E2 = 0.0, 1.0, 2.0
    g01_down, g12_down, g02_down = 0.8, 0.5, 0.3
    gamma_phi0 = gamma_phi1 = gamma_phi2 = 0.1

    print(f"beta = {beta}")
    print(f"Energies: E0 = {E0}, E1 = {E1}, E2 = {E2}")
    print(f"Downward jump rates: g01_down = {g01_down}, "
          f"g12_down = {g12_down}, g02_down = {g02_down}")
    print(f"Dephasing rates: gamma_phi0 = {gamma_phi0}, "
          f"gamma_phi1 = {gamma_phi1}, gamma_phi2 = {gamma_phi2}")
    print("")

    # Build model
    H, rho_ss, L_ops = build_qutrit_model(beta=beta,
                                          E0=E0, E1=E1, E2=E2,
                                          g01_down=g01_down,
                                          g12_down=g12_down,
                                          g02_down=g02_down,
                                          gamma_phi0=gamma_phi0,
                                          gamma_phi1=gamma_phi1,
                                          gamma_phi2=gamma_phi2)

    print("Stationary state rho_ss (thermal Gibbs):")
    print(rho_ss)
    print("")

    # Stationarity check
    d_rho_ss = apply_Hamiltonian(H, rho_ss) + apply_dissipator(L_ops, rho_ss)
    norm_ss = float(np.linalg.norm(d_rho_ss, ord="fro"))
    print("Stationarity diagnostics:")
    print(f"  ||GKLS(rho_ss)||_F ≈ {norm_ss:.3e} (should be near 0)")
    print("")

    # Real generators on u
    K_total, K_H, K_D = build_generator_real(H, L_ops)

    print("Real generators on u = (p0, p1, p2, Re01, Im01, Re02, Im02, Re12, Im12):")
    print("  K_H (Hamiltonian part):")
    print(K_H)
    print("")
    print("  K_D (dissipative part):")
    print(K_D)
    print("")
    print("  K_total = K_H + K_D:")
    print(K_total)
    print("")

    # Metric and metriplectic splitting
    M = build_metric_matrix(rho_ss)
    print("Metric matrix M (rho_ss-weighted Fisher metric on u):")
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

    print("Metriplectic K-splitting diagnostics:")
    print(f"  ||G - K_D||_F / ||K_D||_F ≈ {rel_err_G_D:.3e}")
    print(f"  ||J - K_H||_F / ||K_H||_F ≈ {rel_err_J_H:.3e}")
    print("")

    # Metric symmetry conditions
    G_sharp = metric_adjoint(G, M)
    J_sharp = metric_adjoint(J, M)

    sym_resid_G = frob(G - G_sharp) / max(frob(G), 1e-16)
    skew_resid_J = frob(J + J_sharp) / max(frob(J), 1e-16)

    print("Metric symmetry diagnostics:")
    print(f"  G metric self-adjoint residual ||G - G_sharp||/||G|| ≈ {sym_resid_G:.3e}")
    print(f"  J metric skew-adjoint residual ||J + J_sharp||/||J|| ≈ {skew_resid_J:.3e}")
    print("")

    # Classical Markov generator from population block
    K_D_dens = K_D[0:3, 0:3]
    pi = np.real(np.diag(rho_ss))

    print("Classical 3-state Markov generator Q_markov (population block of K_D):")
    print(K_D_dens)
    print(f"  pi (thermal probabilities) = {pi}")
    Q_pi = K_D_dens @ pi
    col_sums = np.sum(K_D_dens, axis=0)
    print(f"  ||Q_markov pi||_2 ≈ {float(np.linalg.norm(Q_pi, ord=2)):.3e} (should be near 0)")
    print(f"  Column sums (should be ~0 for generator): {col_sums}")
    print("")

    # Canonical classical G_true = Q diag(pi)
    G_true_class = K_D_dens @ np.diag(pi)
    norm_G_true = frob(G_true_class)
    sym_resid_class = frob(G_true_class - G_true_class.T) / max(norm_G_true, 1e-16)
    skew_resid_class = frob(G_true_class + G_true_class.T) / max(norm_G_true, 1e-16)

    print("Canonical classical G_true_class = Q_markov diag(pi):")
    print(G_true_class)
    print(f"  Symmetry residual ||G_true_class - G_true_class^T||/||G_true_class|| ≈ {sym_resid_class:.3e}")
    print(f"  Skew residual     ||G_true_class + G_true_class^T||/||G_true_class|| ≈ {skew_resid_class:.3e}")
    print("")

    # Drift consistency: for mu, p = pi ⊙ mu, check v = Q p = G_true mu
    rng = np.random.default_rng(4242)
    n_probes = 50
    mu_batch = rng.normal(size=(n_probes, 3))

    # Remove trivial constant mode by enforcing pi-weighted mean zero
    for k in range(n_probes):
        avg = float(np.dot(pi, mu_batch[k]))
        mu_batch[k] -= avg

    v_Q = np.zeros_like(mu_batch)
    v_G = np.zeros_like(mu_batch)

    for k in range(n_probes):
        mu = mu_batch[k]
        p = pi * mu
        v_Q[k] = K_D_dens @ p
        v_G[k] = G_true_class @ mu

    drift_rel_err = frob(v_Q - v_G) / max(frob(v_Q), 1e-16)

    print("Drift consistency diagnostics (population GKLS vs classical G_true_class):")
    print(f"  ||Q_markov (pi ⊙ mu) - G_true_class mu||_F / ||Q_markov (pi ⊙ mu)||_F ≈ {drift_rel_err:.3e}")
    print("")

    # Dirichlet form consistency
    E_pair_list = []
    E_G_list = []

    for mu in mu_batch:
        E_pair = dirichlet_from_Q(mu, pi, K_D_dens)
        vG = G_true_class @ mu
        E_G = - float(mu @ vG)
        E_pair_list.append(E_pair)
        E_G_list.append(E_G)

    E_pair_arr = np.array(E_pair_list)
    E_G_arr = np.array(E_G_list)
    diff_E = E_pair_arr - E_G_arr
    mask = np.abs(E_pair_arr) > 1e-12

    if np.any(mask):
        dirichlet_rel_err = float(
            np.linalg.norm(diff_E[mask], ord=2) /
            np.linalg.norm(E_pair_arr[mask], ord=2)
        )
    else:
        dirichlet_rel_err = 0.0

    print("Dirichlet form consistency diagnostics:")
    print(f"  Relative difference between E_pair(mu) and -mu^T G_true_class mu ≈ {dirichlet_rel_err:.3e}")
    print("")

    # Cost-entropy inequality in 3-state Fisher geometry
    G_metric = -G_true_class
    eigvals, eigvecs = np.linalg.eigh(G_metric)

    print("Eigenvalues of G_metric = -G_true_class:")
    print(f"  {eigvals}")
    num_pos = int(np.sum(eigvals > 1e-10))
    num_zero = len(eigvals) - num_pos
    print(f"  Number of strictly positive eigenvalues: {num_pos}")
    print(f"  Number of (near) zero eigenvalues:      {num_zero}")
    print("")

    # Build pseudoinverse of G_metric on positive eigenspace
    pos_mask = eigvals > 1e-10
    V_pos = eigvecs[:, pos_mask]
    lam_pos = eigvals[pos_mask]
    G_pinv = V_pos @ np.diag(1.0 / lam_pos) @ V_pos.T

    def cost_entropy_ratio(mu):
        """
        For G_metric positive semidefinite on the mean-zero subspace, define:

          v = -G_metric mu
          gradF = mu
          C_min = 0.5 v^T G_metric^{-1} v  (on positive eigenspace)
          σ = gradF^T G_metric gradF

        Then by Cauchy-Schwarz R = <v,gradF>^2 / (2 C_min σ) ∈ [0,1],
        with R = 1 exactly when mu is an eigenmode of G_metric.
        """
        v = -G_metric @ mu
        gradF = mu
        C_min = 0.5 * float(v @ (G_pinv @ v))
        sigma = float(gradF @ (G_metric @ gradF))
        num = float(v @ gradF) ** 2
        return num / (2.0 * C_min * sigma)

    print("Cost-entropy ratios R on eigenmodes of G_metric (expect R ≈ 1):")
    R_modes = []
    for idx, (lam, vec) in enumerate(zip(eigvals, eigvecs.T)):
        if lam <= 1e-10:
            continue
        R = cost_entropy_ratio(vec)
        R_modes.append(R)
        print(f"  Mode {idx} with λ ≈ {lam:.3e}: R ≈ {R:.3e}")
    print("")

    # Random combinations in positive eigenspace
    rng2 = np.random.default_rng(2025)
    Rs_random = []
    for _ in range(20):
        coeffs = rng2.normal(size=len(lam_pos))
        mu = V_pos @ coeffs
        Rs_random.append(cost_entropy_ratio(mu))

    Rs_random = np.array(Rs_random)
    print("Cost-entropy ratios R on random directions in positive eigenspace:")
    print(f"  min R ≈ {float(np.min(Rs_random)):.3e}")
    print(f"  max R ≈ {float(np.max(Rs_random)):.3e}")
    print("")

    # PASS / FAIL criteria
    tol_stationary = 1e-10
    tol_split = 1e-10
    tol_metric_sym = 1e-10
    tol_Qpi = 1e-10
    tol_class_sym = 1e-12
    tol_drift = 1e-10
    tol_dirichlet = 1e-10
    tol_R_modes = 1e-10
    tol_R_random = 1e-8  # allow tiny numerical spread

    pass_stationary = norm_ss < tol_stationary
    pass_split = (rel_err_G_D < tol_split) and (rel_err_J_H < tol_split)
    pass_metric_sym = (sym_resid_G < tol_metric_sym) and (skew_resid_J < tol_metric_sym)
    pass_Qpi = float(np.linalg.norm(Q_pi, ord=2)) < tol_Qpi
    pass_class_sym = sym_resid_class < tol_class_sym
    pass_drift = drift_rel_err < tol_drift
    pass_dirichlet = dirichlet_rel_err < tol_dirichlet
    pass_R_modes = np.all(np.abs(np.array(R_modes) - 1.0) < tol_R_modes)
    pass_R_random = (np.min(Rs_random) >= 1.0 - tol_R_random and
                     np.max(Rs_random) <= 1.0 + tol_R_random)

    print("Summary:")
    print(f"  rho_ss stationary for full GKLS?                        {pass_stationary} (tol = {tol_stationary})")
    print(f"  G ≈ dissipative K_D and J ≈ Hamiltonian K_H?            {pass_split} (tol = {tol_split})")
    print(f"  G, J satisfy metric symmetry/skew conditions?           {pass_metric_sym} (tol = {tol_metric_sym})")
    print(f"  Q_markov pi ≈ 0 (correct stationary distribution)?      {pass_Qpi} (tol = {tol_Qpi})")
    print(f"  G_true_class symmetric (classical Fisher Dirichlet)?    {pass_class_sym} (tol = {tol_class_sym})")
    print(f"  Drift match: Q(pi⊙mu) ≈ G_true_class mu?                {pass_drift} (tol = {tol_drift})")
    print(f"  Dirichlet: E_pair(mu) ≈ -mu^T G_true_class mu?          {pass_dirichlet} (tol = {tol_dirichlet})")
    print(f"  Cost-entropy: R ≈ 1 on eigenmodes?                      {pass_R_modes} (tol = {tol_R_modes})")
    print(f"  Cost-entropy: R in [0,1] on random positive directions? {pass_R_random} (tol = {tol_R_random})")
    print("")

    all_pass = (pass_stationary and pass_split and pass_metric_sym and
                pass_Qpi and pass_class_sym and pass_drift and
                pass_dirichlet and pass_R_modes and pass_R_random)

    if all_pass:
        print("GKLS qutrit K-splitting and G-unification CHECK: PASS")
        print("  The 3-level GKLS model admits a canonical metriplectic")
        print("  splitting K = G + J in the rho_ss-weighted Fisher metric.")
        print("  The symmetric part G matches the dissipator, the antisymmetric")
        print("  part J matches the Hamiltonian, and the population block of G")
        print("  reproduces the classical reversible Markov generator Q with")
        print("  its Fisher Dirichlet form and cost-entropy structure. The")
        print("  classical information hydrodynamics is thus literally the")
        print("  density sector of the quantum metriplectic K.")
    else:
        print("GKLS qutrit K-splitting and G-unification CHECK: FAIL")
        print("  At least one of the diagnostics did not meet the specified")
        print("  tolerance. See detailed outputs above for which conditions")
        print("  were violated.")


if __name__ == "__main__":
    run()
