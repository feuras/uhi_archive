#!/usr/bin/env python3
"""
24_gkls_random_qutrit_density_ensemble.py

Ensemble of random *genuinely coherent* qutrit GKLS models:

  - N = 3 levels, computational basis |0>,|1>,|2>.
  - Random Hermitian H (generic coherent mixing).
  - Lindblad operators:
        0 <-> 1:  L01_down = sqrt(g01_down) |0><1|,
                   L01_up   = sqrt(g01_up)   |1><0|
        1 <-> 2:  L12_down = sqrt(g12_down) |1><2|,
                   L12_up   = sqrt(g12_up)   |2><1|
        Dephasing: P0 = |0><0|, P1 = |1><1|, P2 = |2><2|
                   Ld_k = sqrt(g_phi_k) Pk

For each random model we:

  1. Build the GKLS superoperator K on vec(rho) (3x3 system).
  2. Find a stationary state rho_ss with K vec(rho_ss) = 0, then
     hermitise, enforce positivity, and normalise.
  3. Require rho_ss to be full rank and nondegenerate:
        min(pi) > pi_min,  min_{i<j} |pi_i - pi_j| > gap_min
     where pi are eigenvalues of rho_ss.
  4. Diagonalise rho_ss = U diag(pi) U^dagger and transform K into this
     eigenbasis:
        K_eig = T^{-1} K T,  T = U* ⊗ U.
  5. Extract the induced density generator Q_eff on the diagonal subspace
     in the rho_ss eigenbasis by acting K_eig on the diagonal basis
     elements E_11, E_22, E_33 and taking the diagonal of K_eig(E_jj).
  6. Check:
        - column sums of Q_eff are near zero,
        - Q_eff pi ≈ 0,
        - detailed balance residuals
              max_{i<j} |pi_i w_ij - pi_j w_ji|
          are small, where w_ij = Q_eff[j,i].
  7. Build the BKM metric at diag(pi), construct K_eig^sharp and G_eig,
     and compare the GKLS density-sector Dirichlet form

           E_GKLS(delta p) = - <delta u, M G_eig delta u>

     with the classical Fisher Dirichlet energy for the 3-state chain
     (pi, Q_eff),

           E_classical(delta p)
             = 0.5 sum_{i,j} pi_i w_ij (phi_j - phi_i)^2,

     where phi_i = delta p_i / pi_i, over many random mass-conserving
     perturbations delta p.

We then summarise max residuals and Dirichlet mismatches over accepted models.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng
from numpy.linalg import eig, eigh


# --------------------------------------------------------------------
# Basic utilities
# --------------------------------------------------------------------

def vec(rho):
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def build_random_hermitian(N, rng, scale=1.0):
    """
    Random Hermitian H = A + A^dagger with complex Gaussian entries.
    """
    A = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)
    H = A + A.conj().T
    return 0.5 * scale * H


def build_qutrit_lindblads(rng,
                           g01_range=(0.2, 1.5),
                           g12_range=(0.2, 1.5),
                           gphi_range=(0.2, 1.5)):
    """
    Build a set of qutrit Lindblad operators:

      L01_down = sqrt(g01_down) |0><1|
      L01_up   = sqrt(g01_up)   |1><0|
      L12_down = sqrt(g12_down) |1><2|
      L12_up   = sqrt(g12_up)   |2><1|
      Ld_k     = sqrt(g_phi_k)  |k><k|   for k=0,1,2
    """
    N = 3
    # Ladder operators
    L01_down = np.zeros((N, N), dtype=complex)
    L01_down[0, 1] = 1.0
    L01_up = np.zeros((N, N), dtype=complex)
    L01_up[1, 0] = 1.0

    L12_down = np.zeros((N, N), dtype=complex)
    L12_down[1, 2] = 1.0
    L12_up = np.zeros((N, N), dtype=complex)
    L12_up[2, 1] = 1.0

    # Random rates
    g01_down = rng.uniform(*g01_range)
    g01_up = rng.uniform(*g01_range)
    g12_down = rng.uniform(*g12_range)
    g12_up = rng.uniform(*g12_range)

    L01_down *= np.sqrt(g01_down)
    L01_up *= np.sqrt(g01_up)
    L12_down *= np.sqrt(g12_down)
    L12_up *= np.sqrt(g12_up)

    # Dephasing projectors
    P0 = np.zeros((N, N), dtype=complex)
    P1 = np.zeros((N, N), dtype=complex)
    P2 = np.zeros((N, N), dtype=complex)
    P0[0, 0] = 1.0
    P1[1, 1] = 1.0
    P2[2, 2] = 1.0

    g_phi0 = rng.uniform(*gphi_range)
    g_phi1 = rng.uniform(*gphi_range)
    g_phi2 = rng.uniform(*gphi_range)

    Ld0 = np.sqrt(g_phi0) * P0
    Ld1 = np.sqrt(g_phi1) * P1
    Ld2 = np.sqrt(g_phi2) * P2

    L_list = [L01_down, L01_up, L12_down, L12_up, Ld0, Ld1, Ld2]
    return L_list


def build_gkls_superoperator(H, L_list):
    """
    GKLS superoperator K on vec(rho):

        d/dt vec(rho) = K vec(rho)

    with column stacking convention.
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    dim = N * N
    I_N = np.eye(N, dtype=complex)

    K = np.zeros((dim, dim), dtype=complex)

    # Hamiltonian part
    K_H = -1j * (np.kron(I_N, H) - np.kron(H.T, I_N))
    K += K_H

    # Dissipative part
    for L in L_list:
        L = np.asarray(L, dtype=complex)
        Ld = L.conj().T
        LdL = Ld @ L
        term_jump = np.kron(L.conj(), L)
        term_left = np.kron(I_N, LdL)
        term_right = np.kron(LdL.T, I_N)
        K += term_jump - 0.5 * (term_left + term_right)

    return K


def find_stationary_state(K, N, tol=1e-12):
    """
    Find stationary rho_ss with K vec(rho_ss) ≈ 0, enforce Hermiticity,
    positivity, and trace 1.
    """
    evals, evecs = eig(K)
    idx = np.argmin(np.abs(evals))
    v_ss = evecs[:, idx]
    rho_ss = unvec(v_ss, N)
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    vals, vecs = eigh(rho_ss)
    vals_clipped = np.clip(vals.real, 0.0, None)
    if vals_clipped.sum() < tol:
        raise RuntimeError("Stationary state eigenvalues too small after clipping")
    vals_clipped = vals_clipped / vals_clipped.sum()
    rho_ss_pos = vecs @ np.diag(vals_clipped) @ vecs.conj().T
    rho_ss_pos = 0.5 * (rho_ss_pos + rho_ss_pos.conj().T)
    return rho_ss_pos


def bkm_weights(pi):
    """
    BKM (Kubo-Mori) weights c_ij for Hessian of S(rho || diag(pi))
    at rho = diag(pi) in the eigenbasis of rho_ss.
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
    Classical Fisher-Dirichlet for a 3-state chain with generator Q_eff
    (dp/dt = Q_eff p) and stationary pi.

      E = 0.5 sum_{i,j} pi_i w_ij (phi_j - phi_i)^2
      w_ij = Q_eff[j,i]
    """
    delta_p = np.asarray(delta_p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    Q_eff = np.asarray(Q_eff, dtype=float)
    N = len(pi)
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


def gkls_dirichlet_density(delta_p, G_eig, M_diag, N):
    """
    GKLS density-sector Dirichlet in rho_ss eigenbasis:

        E_GKLS = - <delta u, M G_eig delta u>
    """
    delta_p = np.asarray(delta_p, dtype=float)
    delta_rho = np.diag(delta_p)
    delta_u = vec(delta_rho)
    MG_du = M_diag * (G_eig @ delta_u)
    E = -np.vdot(delta_u, MG_du).real
    return float(E)


# --------------------------------------------------------------------
# Single model experiment
# --------------------------------------------------------------------

def run_single_model(model_index,
                     pi_min=1e-3,
                     gap_min=5e-3,
                     n_dirichlet_tests=50,
                     seed=12345):
    rng = default_rng(seed + 1000 * model_index)
    N = 3

    max_tries = 10
    for attempt in range(max_tries):
        # Random Hermitian H
        H = build_random_hermitian(N, rng, scale=1.0)

        # Random qutrit Lindblads
        L_list = build_qutrit_lindblads(rng)

        # GKLS superoperator
        K = build_gkls_superoperator(H, L_list)

        # Stationary state
        rho_ss = find_stationary_state(K, N)

        # Coherence norm in computational basis
        rho_diag = np.diag(np.diag(rho_ss))
        coh_norm = float(np.linalg.norm(rho_ss - rho_diag))

        # Diagonalise rho_ss
        vals, U = eigh(rho_ss)
        pi = np.clip(vals.real, 0.0, None)
        if pi.sum() <= 0.0:
            continue
        pi = pi / pi.sum()

        # Full rank and nondegenerate
        if np.min(pi) < pi_min:
            continue
        gaps = []
        for i in range(N):
            for j in range(i + 1, N):
                gaps.append(abs(pi[i] - pi[j]))
        if min(gaps) < gap_min:
            continue

        # Transform K into rho_ss eigenbasis
        T = np.kron(U.conj(), U)
        T_inv = np.kron(U.T, U.conj().T)
        K_eig = T_inv @ K @ T

        # Stationarity in eigenbasis
        rho_ss_eig = np.diag(pi)
        stat_resid_eig = float(np.linalg.norm(K_eig @ vec(rho_ss_eig)))

        # Induced density generator Q_eff
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

        col_sums = Q_eff.sum(axis=0)
        col_resid_Q = float(np.linalg.norm(col_sums, ord=np.inf))
        stat_resid_Q = float(np.linalg.norm(Q_eff @ pi))

        # Rates and detailed balance residuals
        w_eff = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j:
                    w_eff[i, j] = Q_eff[j, i]

        db_resid = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                db_resid = max(db_resid, abs(pi[i] * w_eff[i, j] - pi[j] * w_eff[j, i]))

        # BKM metric and symmetric G_eig
        C = bkm_weights(pi)
        M_diag = C.reshape(-1, order="F")
        M_inv = 1.0 / M_diag
        Kdag = K_eig.conj().T
        Ksharp = (M_inv[:, None]) * Kdag * (M_diag[None, :])
        G_eig = 0.5 * (K_eig + Ksharp)

        # Dirichlet comparisons
        max_abs_err = 0.0
        max_rel_err = 0.0
        min_Ec = float("inf")
        min_Eg = float("inf")
        max_Ec = 0.0
        max_Eg = 0.0

        for _ in range(n_dirichlet_tests):
            x = rng.normal(size=N)
            x -= x.mean()
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

        return {
            "accepted": True,
            "attempts": attempt + 1,
            "coh_norm": coh_norm,
            "pi": pi,
            "stat_resid_eig": stat_resid_eig,
            "col_resid_Q": col_resid_Q,
            "stat_resid_Q": stat_resid_Q,
            "db_resid": db_resid,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "min_Ec": min_Ec,
            "min_Eg": min_Eg,
            "max_Ec": max_Ec,
            "max_Eg": max_Eg,
        }

    return {
        "accepted": False,
        "attempts": max_tries,
    }


# --------------------------------------------------------------------
# Main ensemble driver
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Random qutrit GKLS ensemble: density-sector Fisher–Markov checks."
    )
    parser.add_argument("--n_models", type=int, default=10,
                        help="Number of random GKLS models to attempt.")
    parser.add_argument("--pi_min", type=float, default=1e-3,
                        help="Minimum eigenvalue of rho_ss to accept.")
    parser.add_argument("--gap_min", type=float, default=5e-3,
                        help="Minimum spectral gap |pi_i - pi_j| to accept.")
    parser.add_argument("--n_dirichlet_tests", type=int, default=50,
                        help="Number of random density perturbations per model.")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Base random seed.")
    parser.add_argument("--n_jobs", type=int, default=0,
                        help="Number of parallel workers (0 for auto / sequential).")

    args = parser.parse_args()

    n_models = args.n_models
    pi_min = args.pi_min
    gap_min = args.gap_min
    n_dirichlet_tests = args.n_dirichlet_tests
    seed = args.seed

    if args.n_jobs > 0:
        n_jobs = args.n_jobs
    else:
        cpu_count = mp.cpu_count()
        n_jobs = min(20, cpu_count)

    print("24_gkls_random_qutrit_density_ensemble.py")
    print("-----------------------------------------")
    print(f"n_models                   = {n_models}")
    print(f"pi_min                     = {pi_min}")
    print(f"gap_min                    = {gap_min}")
    print(f"n_dirichlet_tests          = {n_dirichlet_tests}")
    print(f"base random seed           = {seed}")
    print(f"parallel workers n_jobs    = {n_jobs}")
    print()

    start_time = time.time()

    worker_args = [
        (m, pi_min, gap_min, n_dirichlet_tests, seed)
        for m in range(n_models)
    ]

    if n_jobs == 1:
        results = [run_single_model(*wa) for wa in worker_args]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_model, worker_args)

    elapsed = time.time() - start_time

    accepted = [r for r in results if r.get("accepted", False)]
    n_accepted = len(accepted)

    print(f"Models accepted (full rank, nondegenerate rho_ss): {n_accepted}/{n_models}")
    if n_accepted == 0:
        print("No acceptable models found; consider relaxing pi_min/gap_min.")
        print(f"Elapsed wall clock time              = {elapsed:.2f} s")
        return

    coh_norms = np.array([r["coh_norm"] for r in accepted])
    max_abs_errs = np.array([r["max_abs_err"] for r in accepted])
    max_rel_errs = np.array([r["max_rel_err"] for r in accepted])
    stat_resid_eig = np.array([r["stat_resid_eig"] for r in accepted])
    col_resid_Q = np.array([r["col_resid_Q"] for r in accepted])
    stat_resid_Q = np.array([r["stat_resid_Q"] for r in accepted])
    db_resids = np.array([r["db_resid"] for r in accepted])

    print()
    print("Summary over accepted models:")
    print(f"  Mean coherence norm in comp basis  = {np.mean(coh_norms):.3e}")
    print(f"  Min, max coherence norm            = {np.min(coh_norms):.3e}, {np.max(coh_norms):.3e}")
    print()
    print(f"  Max stationarity residual (eigen)  = {np.max(stat_resid_eig):.3e}")
    print(f"  Max column sum residual Q_eff      = {np.max(col_resid_Q):.3e}")
    print(f"  Max stationarity residual Q_eff pi = {np.max(stat_resid_Q):.3e}")
    print(f"  Max detailed balance residual      = {np.max(db_resids):.3e}")
    print()
    print(f"  Max |E_GKLS - E_classical|         = {np.max(max_abs_errs):.3e}")
    print(f"  Max relative error                 = {np.max(max_rel_errs):.3e}")
    print()
    print(f"Elapsed wall clock time              = {elapsed:.2f} s")


if __name__ == "__main__":
    main()
