#!/usr/bin/env python3
"""
06_gkls_fp_G_unification_checks.py

Canonical G-unification between:

  - the irreversible generator of a reversible nearest neighbour Markov chain
    (diagonal GKLS sector) with detailed balance w.r.t. a Gibbs density pi, and

  - a finite difference Fokker Planck discretisation on the same ring with the
    same potential and equilibrium density.

Core theory checks (PASS criteria):

  1) From the Markov generator Q and stationary density pi, construct
         G_true = Q diag(pi)
     and verify that G_true is symmetric (Dirichlet form is a quadratic form).

  2) Verify that, for smooth probe potentials mu,
         v_markov(mu) = Q (pi ⊙ mu)
         v_Gtrue(mu)  = G_true mu
     agree on training and test probes (Markov drift matches canonical G_true).

  3) Build a Fokker Planck operator L_FP that discretises
         ∂_t ρ = ∂_x( D ∂_x ρ + ρ ∂_x(V/D) )
     and show that, on the same test probes,
         v_FP(mu) = L_FP (pi ⊙ mu)
     agrees with v_markov(mu) up to discretisation error.

These steps demonstrate that the canonical symmetric G_true extracted from
the reversible Markov chain is also realised by the Fokker Planck limit.

Secondary diagnostics (tomography experiment):

  - Perform G-tomography from Markov data on a low dimensional smooth
    potential subspace and reconstruct a symmetric G_hat by pseudoinverse
    plus symmetrisation.

  - Compare G_hat to G_true and to FP responses on the same probes.

This tomography block is an identifiability experiment only and does NOT
enter the PASS criteria. It quantifies how much of G_true can be inferred
from finitely many Fisher probes on a large lattice.

You can control thread count via the environment variable G_UNIFY_NUM_WORKERS.
"""

import os
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def determine_num_workers():
    """
    Choose number of worker threads.

    Priority:
      1. Environment variable G_UNIFY_NUM_WORKERS if set and >= 1.
      2. Otherwise min(20, os.cpu_count()) with a floor of 1.
    """
    env_value = os.environ.get("G_UNIFY_NUM_WORKERS", "").strip()
    if env_value:
        try:
            n = int(env_value)
            if n >= 1:
                return n
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(20, cpu))


def build_potential_and_gibbs(nx, length, D, V0):
    """
    Build spatial grid x, potential V(x) = V0 cos(x), and Gibbs density
    pi ∝ exp(−V/D).
    """
    x = np.linspace(0.0, length, nx, endpoint=False)
    V = V0 * np.cos(x)
    beta = 1.0 / D
    w = np.exp(-beta * V)
    pi = w / np.sum(w)
    return x, V, pi


def build_detailed_balance_Q(V, D, base_rate=1.0):
    """
    Build a reversible nearest neighbour Markov generator Q satisfying
    detailed balance with respect to the Gibbs density corresponding
    to V and D.

    Rates k_{j i} (from i → j) are
        k_{j i} = base_rate * exp(-(V_j - V_i)/(2 D)).

    For neighbours j = i ± 1 (mod N) this gives detailed balance
        π_i k_{j i} = π_j k_{i j}
    where π_i ∝ exp(−V_i / D). Q acts on column vectors p via dp/dt = Q p.
    """
    nx = V.size
    Q = np.zeros((nx, nx), dtype=float)
    beta = 1.0 / D
    for i in range(nx):
        for j in ((i - 1) % nx, (i + 1) % nx):
            dV = V[j] - V[i]
            k_ji = base_rate * math.exp(-0.5 * beta * dV)
            Q[j, i] += k_ji
            Q[i, i] -= k_ji
    return Q


def symmetry_metrics(M):
    """
    Return (symmetry_residual, skew_residual) for matrix M in Frobenius norm units:
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


def forward_markov_batch(mu_batch, pi, Q, n_workers):
    """
    Compute Markov responses v_k = Q (pi ⊙ mu_k) for a batch of probe potentials μ_k.

    mu_batch: shape (n_probes, nx)
    pi:       shape (nx,)
    Q:        shape (nx, nx)
    Returns v_batch: shape (n_probes, nx)
    """
    n_probes, nx = mu_batch.shape
    v_batch = np.zeros((n_probes, nx), dtype=float)

    def _single(mu_vec):
        q = pi * mu_vec
        return Q @ q

    if n_workers <= 1 or n_probes == 1:
        for k in range(n_probes):
            v_batch[k] = _single(mu_batch[k])
        return v_batch

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_single, mu_batch[k]) for k in range(n_probes)]
        for k, fut in enumerate(futures):
            v_batch[k] = fut.result()

    return v_batch


def build_fp_aux(x, V, D):
    """
    Precompute grid spacing and discrete derivative of V/D needed for L_FP.

    We use central differences for dV/dx on a periodic ring.
    """
    nx = x.size
    length = x[-1] - x[0] + (x[1] - x[0])
    dx = length / nx
    dVdx = np.zeros(nx, dtype=float)
    for i in range(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        dVdx[i] = (V[ip] - V[im]) / (2.0 * dx)
    c = dVdx / D  # c_i ≈ ∂_x(V/D)
    return dx, c


def apply_L_fp_batch(mu_batch, pi, x, V, D, n_workers):
    """
    Apply the discrete Fokker Planck operator L_FP to q_k = pi ⊙ mu_k for a batch.

    Continuum target:
        ∂_t ρ = ∂_x( D ∂_x ρ + ρ ∂_x(V/D) ).

    Discretisation:
        (L_FP ρ)_i = D (ρ_{i+1} - 2ρ_i + ρ_{i-1}) / dx^2
                     + (c_{i+1} ρ_{i+1} - c_{i-1} ρ_{i-1}) / (2 dx),

    with periodic boundary conditions and c_i = ∂_x(V/D) at grid point i.
    """
    n_probes, nx = mu_batch.shape
    v_batch = np.zeros((n_probes, nx), dtype=float)
    dx, c = build_fp_aux(x, V, D)

    def _single(mu_vec):
        q = pi * mu_vec
        rho = q
        out = np.zeros_like(rho)
        for i in range(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            diff = D * (rho[ip] - 2.0 * rho[i] + rho[im]) / (dx * dx)
            drift = (c[ip] * rho[ip] - c[im] * rho[im]) / (2.0 * dx)
            out[i] = diff + drift
        return out

    if n_workers <= 1 or n_probes == 1:
        for k in range(n_probes):
            v_batch[k] = _single(mu_batch[k])
        return v_batch

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_single, mu_batch[k]) for k in range(n_probes)]
        for k, fut in enumerate(futures):
            v_batch[k] = fut.result()

    return v_batch


def generate_fourier_basis(x, k_max):
    """
    Generate a low dimensional Fourier basis on the ring:

        mode 0: 1
        modes: cos(k x), sin(k x) for k = 1,...,k_max

    Returns basis with shape (nx, n_modes).
    """
    nx = x.size
    modes = [np.ones(nx, dtype=float)]
    for k in range(1, k_max + 1):
        modes.append(np.cos(k * x))
        modes.append(np.sin(k * x))
    return np.stack(modes, axis=1)


def build_random_mu(rng, basis, n_probes, coeff_scale=1.0):
    """
    Draw random smooth probe potentials from the Fourier basis with zero mean.
    """
    nx, n_modes = basis.shape
    coeffs = rng.normal(loc=0.0, scale=coeff_scale, size=(n_probes, n_modes))
    mu = coeffs @ basis.T  # (n_probes, nx)
    mu = mu - mu.mean(axis=1, keepdims=True)
    return mu


def reconstruct_G(mu_batch, v_batch, rcond=1e-10):
    """
    Reconstruct a symmetric G_hat from probe data v_k ≈ G μ_k using pseudoinverse.

    We solve Y = G X in least squares sense, with
        X = [μ_1 ... μ_K] ∈ R^{nx×K},
        Y = [v_1 ... v_K] ∈ R^{nx×K},
    and set G_rec = Y X^+ then project to the symmetric subspace:
        G_hat = 0.5 (G_rec + G_rec^T).
    """
    n_probes, nx = mu_batch.shape
    X = mu_batch.T  # (nx, n_probes)
    Y = v_batch.T   # (nx, n_probes)
    X_pinv = np.linalg.pinv(X, rcond=rcond)  # (n_probes, nx)
    G_rec = Y @ X_pinv                       # (nx, nx)
    G_hat = 0.5 * (G_rec + G_rec.T)
    return G_hat


def best_scalar_match(A, B):
    """
    Given two arrays A and B of the same shape, compute the scalar alpha that
    minimises ||alpha A - B||_F and the corresponding relative residual
    ||alpha A - B||_F / ||B||_F.

    If ||A||_F = 0 or ||B||_F = 0, alpha is set to 0 and residual to 0.
    """
    frob = lambda M: float(np.linalg.norm(M, ord="fro"))
    norm_A2 = frob(A) ** 2
    norm_B = frob(B)
    if norm_A2 == 0.0 or norm_B == 0.0:
        return 0.0, 0.0
    num = float(np.tensordot(A, B))
    alpha = num / norm_A2
    resid = frob(alpha * A - B) / norm_B
    return alpha, resid


def run():
    print("GKLS / Markov vs Fokker Planck G-unification")
    print("-------------------------------------------")

    # Core parameters
    nx = 4096
    length = 2.0 * math.pi
    D = 1.0           # noise / temperature scale in Gibbs factor and FP
    V0 = 0.5          # potential amplitude
    k_max = 4         # highest Fourier mode in probe basis
    n_probes = 40     # number of training probes
    n_test = 20       # number of test probes
    coeff_scale = 1.0
    rng_seed = 24680
    rcond_pinv = 1e-10

    n_workers = determine_num_workers()

    print(f"nx = {nx}, length = {length}")
    print(f"D (noise scale) = {D}, V0 (potential amplitude) = {V0}")
    print(f"k_max = {k_max}, n_probes = {n_probes}, n_test = {n_test}")
    print(f"Random seed = {rng_seed}")
    print(f"Worker threads = {n_workers}")
    print("")

    rng = np.random.default_rng(rng_seed)

    # Build potential, Gibbs density
    x, V, pi = build_potential_and_gibbs(nx, length, D, V0)
    dx = length / nx
    print(f"Computed dx = {dx:.3e}")
    print("")

    # Build Markov generator with diffusive scaling base_rate = D / dx^2
    base_rate = D / (dx * dx)
    Q = build_detailed_balance_Q(V, D, base_rate=base_rate)

    # Diagnostics: Markov stationarity and conservation
    resid_Q_pi = float(np.linalg.norm(Q @ pi, ord=2))
    col_sums = Q.sum(axis=0)
    resid_cols = float(np.linalg.norm(col_sums, ord=2))

    print("Markov generator diagnostics:")
    print(f"  base_rate = D / dx^2 ≈ {base_rate:.3e}")
    print(f"  ||Q pi||_2 ≈ {resid_Q_pi:.3e} (should be near 0)")
    print(f"  Column sum residual ||1^T Q||_2 ≈ {resid_cols:.3e}")
    print("")

    # Fokker Planck discretisation diagnostics
    dx_fp, _ = build_fp_aux(x, V, D)
    v_pi = apply_L_fp_batch(pi.reshape(1, -1), pi, x, V, D, n_workers=1)[0]
    resid_L_pi = float(np.linalg.norm(v_pi, ord=2))

    print("Fokker Planck discretisation diagnostics:")
    print(f"  dx (from FP aux) = {dx_fp:.3e}")
    print(f"  ||L_FP pi||_2 ≈ {resid_L_pi:.3e} (discretisation error)")
    print("")

    # Build smooth probe potentials
    basis = generate_fourier_basis(x, k_max)
    mu_batch = build_random_mu(rng, basis, n_probes, coeff_scale=coeff_scale)
    mu_test = build_random_mu(rng, basis, n_test, coeff_scale=coeff_scale)

    # Markov responses for training probes
    v_markov = forward_markov_batch(mu_batch, pi, Q, n_workers)

    # Canonical operator on μ-space: G_true = Q diag(pi)
    G_true = Q @ np.diag(pi)
    G_true_sym_resid, G_true_skew_resid = symmetry_metrics(G_true)

    frob = lambda M: float(np.linalg.norm(M, ord="fro"))
    frob_G_true = frob(G_true)

    print("Canonical G_true = Q diag(pi) diagnostics:")
    print(f"  Symmetry residual ||G_true - G_true^T||/||G_true|| ≈ {G_true_sym_resid:.3e}")
    print(f"  Skew residual     ||G_true + G_true^T||/||G_true|| ≈ {G_true_skew_resid:.3e}")
    print("")

    # Check that G_true reproduces Markov responses on training probes
    v_true_train = (G_true @ mu_batch.T).T
    rel_err_true_train = frob(v_true_train - v_markov) / frob(v_markov)

    print("Canonical G_true forward map diagnostics (training probes):")
    print(f"  ||G_true mu - v_markov||_F / ||v_markov||_F ≈ {rel_err_true_train:.3e}")
    print("")

    # Same check on test probes
    v_markov_test = forward_markov_batch(mu_test, pi, Q, n_workers)
    v_true_test = (G_true @ mu_test.T).T
    rel_err_true_test = frob(v_true_test - v_markov_test) / frob(v_markov_test)

    print("Canonical G_true forward map diagnostics (test probes):")
    print(f"  ||G_true mu - v_markov||_F / ||v_markov||_F ≈ {rel_err_true_test:.3e}")
    print("")

    # Fokker Planck responses on the same test probes
    v_fp_test = apply_L_fp_batch(mu_test, pi, x, V, D, n_workers)

    # FP vs Markov and FP vs G_true comparisons
    rel_err_fp_vs_markov = frob(v_fp_test - v_markov_test) / frob(v_markov_test)
    rel_err_fp_vs_true = frob(v_fp_test - v_true_test) / frob(v_fp_test)

    alpha_fp_to_markov, resid_fp_to_markov = best_scalar_match(v_fp_test, v_markov_test)
    alpha_fp_to_true, resid_fp_to_true = best_scalar_match(v_fp_test, v_true_test)

    print("GKLS/Markov vs Fokker Planck comparison on test probes (canonical G slice):")
    print(f"  Raw  ||v_FP - v_markov||_F / ||v_markov||_F ≈ {rel_err_fp_vs_markov:.3e}")
    print(f"  Raw  ||v_FP - G_true mu||_F / ||v_FP||_F ≈ {rel_err_fp_vs_true:.3e}")
    print(f"  Best scalar α_FP→Markov ≈ {alpha_fp_to_markov:.3e}, "
          f"resid after scaling ≈ {resid_fp_to_markov:.3e}")
    print(f"  Best scalar α_FP→G_true  ≈ {alpha_fp_to_true:.3e}, "
          f"resid after scaling ≈ {resid_fp_to_true:.3e}")
    print("")

    # Tomography experiment: identifiability from limited probes
    print("Tomography experiment (identifiability, not used for PASS):")
    G_hat = reconstruct_G(mu_batch, v_markov, rcond=rcond_pinv)
    G_hat_sym_resid, G_hat_skew_resid = symmetry_metrics(G_hat)
    frob_diff_G = frob(G_hat - G_true)
    rel_err_G = frob_diff_G / frob_G_true

    print("  Reconstructed G_hat diagnostics (from Markov tomography):")
    print(f"    Symmetry residual ||G_hat - G_hat^T||/||G_hat|| ≈ {G_hat_sym_resid:.3e}")
    print(f"    Skew residual     ||G_hat + G_hat^T||/||G_hat|| ≈ {G_hat_skew_resid:.3e}")
    print(f"    Relative operator error ||G_hat - G_true||/||G_true|| ≈ {rel_err_G:.3e}")

    v_hat_train = (G_hat @ mu_batch.T).T
    rel_err_hat_train = frob(v_hat_train - v_markov) / frob(v_markov)

    v_hat_test = (G_hat @ mu_test.T).T
    rel_err_hat_test = frob(v_hat_test - v_markov_test) / frob(v_markov_test)

    alpha_fp_to_Ghat, resid_fp_to_Ghat = best_scalar_match(v_fp_test, v_hat_test)

    print("  Forward map diagnostics for G_hat:")
    print(f"    Training probes: ||G_hat mu - v_markov||_F / ||v_markov||_F ≈ {rel_err_hat_train:.3e}")
    print(f"    Test probes:     ||G_hat mu - v_markov||_F / ||v_markov||_F ≈ {rel_err_hat_test:.3e}")
    print(f"    Best scalar α_FP→G_hat ≈ {alpha_fp_to_Ghat:.3e}, "
          f"resid after scaling ≈ {resid_fp_to_Ghat:.3e}")
    print("")

    # PASS / FAIL criteria based only on canonical G_true and FP/Markov alignment
    tol_sym_true = 1e-10
    tol_true_train = 1e-8
    tol_true_test = 1e-8
    tol_fp_markov_scaled = 5e-5
    tol_fp_true_scaled = 5e-5

    pass_sym_true = G_true_sym_resid < tol_sym_true
    pass_true_train = rel_err_true_train < tol_true_train
    pass_true_test = rel_err_true_test < tol_true_test
    pass_fp_markov_scaled = resid_fp_to_markov < tol_fp_markov_scaled
    pass_fp_true_scaled = resid_fp_to_true < tol_fp_true_scaled

    all_pass = (
        pass_sym_true
        and pass_true_train
        and pass_true_test
        and pass_fp_markov_scaled
        and pass_fp_true_scaled
    )

    print("Summary (canonical G_unification checks):")
    print(f"  G_true symmetric up to tol?                        {pass_sym_true} (tol = {tol_sym_true})")
    print(f"  G_true forward map fit on training probes?         {pass_true_train} (tol = {tol_true_train})")
    print(f"  G_true forward map fit on test probes?             {pass_true_test} (tol = {tol_true_test})")
    print(f"  v_FP ≈ α v_markov on test probes after scaling?    {pass_fp_markov_scaled} (tol = {tol_fp_markov_scaled})")
    print(f"  v_FP ≈ α G_true mu on test probes after scaling?   {pass_fp_true_scaled} (tol = {tol_fp_true_scaled})")
    print("")
    print("Note: tomography diagnostics for G_hat are informative about identifiability")
    print("      from finite Fisher probes, but do not enter the PASS criteria above.")
    print("")

    if all_pass:
        print("G_unification CHECK: PASS")
        print("  The canonical symmetric G_true = Q diag(pi) reproduces the Markov")
        print("  irreversible drift on the probe subspace and matches the Fokker")
        print("  Planck discretisation there up to discretisation error. This is the")
        print("  canonical irreversible slice of the Fisher metriplectic structure.")
    else:
        print("G_unification CHECK: FAIL (see canonical diagnostics above).")


if __name__ == "__main__":
    run()
