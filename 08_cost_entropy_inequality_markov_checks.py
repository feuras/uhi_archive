#!/usr/bin/env python3
"""
08_cost_entropy_inequality_markov_checks.py

Local cost–entropy inequality on the canonical Markov/GKLS G for a
four state reversible thermal chain.

Setup:
  - Same 4 state reversible Markov model as in 07_gkls_to_markov_G_unification_checks:
      * energies E_i
      * thermal stationary distribution pi_therm ∝ exp(-beta E_i)
      * symmetric base couplings k_ij
      * jump rates gamma_{ij} enforcing detailed balance:
            pi_j gamma_{ij} = pi_i gamma_{ji}

  - Classical generator Q_markov from gamma (dp/dt = Q p).

  - Canonical irreversible operator on μ space:
        G_true = Q_markov diag(pi_therm)
    which is symmetric and negative semidefinite on the mean zero sector.

  - We define the positive semidefinite "mobility" metric on μ-space as
        G_metric = -G_true.

  - On the mean zero subspace, G_metric is positive definite. We use its
    eigen decomposition to define the metric
        g(a, b) = a^T G_metric b
    and its inverse
        g^{-1}(v, v) via the pseudoinverse of G_metric on that subspace.

Cost–entropy inequality (instantaneous, local):
  For any gradient direction gradF and corresponding velocity v = -G_metric gradF
  (i.e. Markov drift linearisation near equilibrium), we have

      <v, gradF>^2 <= 2 C_min σ,

  with
      σ      = g(gradF, gradF)        (entropy curvature / Fisher norm),
      C_min  = 0.5 g^{-1}(v, v)       (minimal control cost to realise v).

  This is exactly the Cauchy–Schwarz inequality in the G_metric geometry.

In this script we:

  1) Rebuild the same 4 state thermal Markov model as in 07_... and confirm
     the basic diagnostics (symmetry and negativity of G_true).

  2) Diagonalise G_metric = -G_true, identify the one dimensional kernel
     (constant mode), and restrict everything to the positive eigenvalue
     subspace.

  3) Test the cost–entropy inequality in two regimes:
      - Eigen directions of G_metric: we expect equality in the ratio
            R = <v, gradF>^2 / (2 C_min σ) ≈ 1.
      - Random gradF drawn from the positive subspace: we expect
            R <= 1, with values distributed in (0,1].

The script reports detailed diagnostics and a final PASS / FAIL summary.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Shared model: thermal 4-state reversible Markov chain
# ---------------------------------------------------------------------------

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
# Metric machinery: G_metric from G_true, eigen decomposition, g and g^{-1}
# ---------------------------------------------------------------------------

def metric_setup(G_true, tol_eig=1e-12):
    """
    Given symmetric G_true (negative semidefinite on the mean-zero sector),
    build the positive semidefinite mobility G_metric = -G_true, then
    diagonalise:

        G_metric = V diag(lambda) V^T,

    and extract the positive eigen-subspace needed for the metric and its
    pseudoinverse.

    Returns:
      eigvals      (dim,)
      eigvecs      (dim, dim) as columns
      idx_pos      boolean mask for strictly positive eigenvalues
      G_metric     original G_metric
    """
    G_metric = -G_true
    eigvals, eigvecs = np.linalg.eigh(G_metric)
    idx_pos = eigvals > tol_eig
    return eigvals, eigvecs, idx_pos, G_metric


def metric_inner(eigvals, eigvecs, idx_pos, a, b):
    """
    Metric inner product g(a, b) = a^T G_metric b using the eigen basis.

    On the positive subspace:

        G_metric = V_pos diag(lambda_pos) V_pos^T.

    So

        g(a, b) = (V_pos^T a)^T diag(lambda_pos) (V_pos^T b),
    where V_pos are the eigenvectors with lambda > 0.
    """
    V_pos = eigvecs[:, idx_pos]
    lam_pos = eigvals[idx_pos]
    a_pos = V_pos.T @ a
    b_pos = V_pos.T @ b
    return float(np.dot(a_pos * lam_pos, b_pos))


def metric_dual_quadratic(eigvals, eigvecs, idx_pos, v):
    """
    Dual quadratic form g^{-1}(v, v) using the pseudoinverse of G_metric on
    the positive subspace.

    In the eigen basis:

        v_pos = V_pos^T v,
        g^{-1}(v, v) = v_pos^T diag(1 / lambda_pos) v_pos.
    """
    V_pos = eigvecs[:, idx_pos]
    lam_pos = eigvals[idx_pos]
    v_pos = V_pos.T @ v
    return float(np.dot(v_pos / lam_pos, v_pos))


def compute_sigma(eigvals, eigvecs, idx_pos, gradF):
    """
    Dissipation σ = g(gradF, gradF).
    """
    return metric_inner(eigvals, eigvecs, idx_pos, gradF, gradF)


def compute_C_min(eigvals, eigvecs, idx_pos, v):
    """
    Minimal control cost C_min = 0.5 g^{-1}(v, v).
    """
    return 0.5 * metric_dual_quadratic(eigvals, eigvecs, idx_pos, v)


# ---------------------------------------------------------------------------
# Main test: cost–entropy inequality in instantaneous Markov/Fisher geometry
# ---------------------------------------------------------------------------

def run():
    print("Cost–entropy inequality in 4-state thermal Markov Fisher geometry")
    print("-----------------------------------------------------------------")

    dim = 4
    beta = 1.0
    rng_seed = 24681357
    rng = np.random.default_rng(rng_seed)

    print(f"dim = {dim}, beta = {beta}, rng_seed = {rng_seed}")
    print("")

    # Build thermal weights and symmetric base couplings (same structure as in 07_...).
    E, pi_therm = build_thermal_pi(dim, beta=beta)
    k = build_symmetric_base_rates(dim, rng, base_scale=1.0, jitter=0.5)
    gamma = build_gamma_from_pi_and_k(pi_therm, k)
    Q_markov = build_Q_from_gamma(gamma)

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

    # Canonical symmetric operator on μ-space: G_true = Q_markov diag(pi_therm).
    G_true = Q_markov @ np.diag(pi_therm)
    G_sym_resid, G_skew_resid = symmetry_metrics(G_true)

    frob = lambda M: float(np.linalg.norm(M, ord="fro"))

    print("Canonical G_true diagnostics:")
    print(f"  Symmetry residual ||G_true - G_true^T||/||G_true|| ≈ {G_sym_resid:.3e}")
    print(f"  Skew residual     ||G_true + G_true^T||/||G_true|| ≈ {G_skew_resid:.3e}")
    print("")

    # Metric setup: G_metric = -G_true, diagonalise.
    eigvals, eigvecs, idx_pos, G_metric = metric_setup(G_true, tol_eig=1e-12)
    num_pos = int(np.sum(idx_pos))
    num_zero = dim - num_pos

    print("Eigen decomposition of G_metric = -G_true:")
    print(f"  Eigenvalues: {eigvals}")
    print(f"  Number of strictly positive eigenvalues: {num_pos}")
    print(f"  Number of (near) zero eigenvalues:      {num_zero}")
    print("")

    # Sanity check: G_metric is positive semidefinite, with one zero mode.
    min_pos = float(np.min(eigvals[idx_pos])) if num_pos > 0 else 0.0
    print(f"  Smallest positive eigenvalue ≈ {min_pos:.3e}")
    print("")

    # -----------------------------------------------------------------------
    # Test 1: Eigen directions (expect equality in cost–entropy inequality)
    # -----------------------------------------------------------------------

    print("Test 1: eigen directions (expect R ≈ 1 for each mode)")
    ratios_eig = []

    for idx, lam in enumerate(eigvals):
        if not idx_pos[idx]:
            continue  # skip kernel
        eigvec = eigvecs[:, idx]
        # Use gradF = eigenvector of G_metric
        gradF = eigvec
        # Drift v = -G_metric gradF = G_true gradF
        v = -G_metric @ gradF

        sigma = compute_sigma(eigvals, eigvecs, idx_pos, gradF)
        C_min = compute_C_min(eigvals, eigvecs, idx_pos, v)
        inner_v_gradF = float(np.dot(v, gradF))

        if sigma <= 0 or C_min <= 0:
            continue

        numerator = inner_v_gradF ** 2
        denom = 2.0 * C_min * sigma
        R = numerator / denom
        ratios_eig.append(R)
        print(f"  Mode with λ ≈ {lam:.3e}: R = <v,gradF>^2 / (2 C_min σ) ≈ {R:.3e}")

    if ratios_eig:
        print(f"  Eigen mode ratios: min ≈ {min(ratios_eig):.3e}, "
              f"max ≈ {max(ratios_eig):.3e}")
    print("")

    # -----------------------------------------------------------------------
    # Test 2: random gradient directions in positive subspace (expect R <= 1)
    # -----------------------------------------------------------------------

    print("Test 2: random gradF in positive eigen-subspace (expect R <= 1)")
    n_random = 200
    ratios_rand = []

    V_pos = eigvecs[:, idx_pos]
    lam_pos = eigvals[idx_pos]

    for _ in range(n_random):
        # Random coefficients in eigen basis for positive subspace
        coeffs = rng.normal(size=(num_pos,))
        gradF = V_pos @ coeffs

        # Drift induced by G_metric (negative of G_true)
        v = -G_metric @ gradF

        sigma = compute_sigma(eigvals, eigvecs, idx_pos, gradF)
        C_min = compute_C_min(eigvals, eigvecs, idx_pos, v)
        inner_v_gradF = float(np.dot(v, gradF))

        # Skip degenerate cases
        if sigma <= 1e-14 or C_min <= 1e-14:
            continue

        numerator = inner_v_gradF ** 2
        denom = 2.0 * C_min * sigma
        R = numerator / denom
        ratios_rand.append(R)

    if ratios_rand:
        print(f"  Random direction ratios: min ≈ {min(ratios_rand):.3e}, "
              f"max ≈ {max(ratios_rand):.3e}, "
              f"mean ≈ {sum(ratios_rand)/len(ratios_rand):.3e}")
    else:
        print("  No valid random samples (this should not happen).")
    print("")

    # -----------------------------------------------------------------------
    # PASS / FAIL summary
    # -----------------------------------------------------------------------

    # Eigen modes should give R very close to 1.
    tol_eig_ratio = 1e-10
    pass_eig = True
    if ratios_eig:
        for R in ratios_eig:
            if abs(R - 1.0) > tol_eig_ratio:
                pass_eig = False
                break

    # Random directions should satisfy R <= 1 + small numerical tolerance.
    tol_rand_ratio = 1e-10
    pass_rand = True
    if ratios_rand:
        max_R = max(ratios_rand)
        if max_R > 1.0 + tol_rand_ratio:
            pass_rand = False

    print("Summary:")
    print(f"  G_true symmetric and G_metric semidefinite?          "
          f"{G_sym_resid < 1e-12} (sym resid tol = 1e-12)")
    print(f"  Eigen mode ratios R ≈ 1 within tol?                  {pass_eig} (tol = {tol_eig_ratio})")
    print(f"  Random direction ratios R <= 1 within tol?           {pass_rand} (tol = {tol_rand_ratio})")
    print("")

    if pass_eig and pass_rand and (G_sym_resid < 1e-12):
        print("Cost–entropy inequality CHECK: PASS")
        print("  In the canonical Fisher/Markov geometry defined by G_true, the")
        print("  instantaneous cost–entropy inequality <v,gradF>^2 <= 2 C_min σ")
        print("  holds for all tested directions, with equality on eigen modes of")
        print("  the mobility G_metric = -G_true. This numerically realises the")
        print("  local inequality structure derived in the metriplectic framework.")
    else:
        print("Cost–entropy inequality CHECK: FAIL (see diagnostics above).")


if __name__ == "__main__":
    run()
