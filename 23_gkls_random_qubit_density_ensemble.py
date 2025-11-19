#!/usr/bin/env python3
"""
23_gkls_random_qubit_density_ensemble.py

Ensemble of random *genuinely coherent* qubit GKLS models:

  - H = 0.5 (h_x sigma_x + h_y sigma_y + h_z sigma_z) with random h.
  - Lindblad operators:
        L1 = sqrt(gamma1) sigma_minus
        L2 = sqrt(gamma2) sigma_plus
        L3 = sqrt(gamma3) sigma_z

For each random model, we:

  1. Build the GKLS generator K on vec(rho) (2x2 system).
  2. Find a stationary state rho_ss with K vec(rho_ss) = 0, then
     hermitise, enforce positivity, and normalise.
  3. Require rho_ss to be full rank and nondegenerate:
        min(pi) > pi_min,  |pi0 - pi1| > gap_min
     where pi are eigenvalues of rho_ss.
  4. Diagonalise rho_ss = U diag(pi) U^dagger and transform K into this
     eigenbasis:
        K_eig = T^{-1} K T, T = U* ⊗ U.
  5. Extract the induced density generator Q_eff on the diagonal subspace
     in the rho_ss eigenbasis by acting K_eig on E_11 and E_22 and taking
     the diagonal of K_eig(E_jj).
  6. Check:
        - column sums of Q_eff are near zero,
        - Q_eff pi ≈ 0,
        - detailed balance residual pi_0 w_01 - pi_1 w_10 is small,
     where w_ij = Q_eff[j,i].
  7. Build the BKM metric at diag(pi), construct K_eig^sharp and G_eig, and
     compare the GKLS density-sector Dirichlet form

           E_GKLS(delta p) = - <delta u, M G_eig delta u>

     with the classical Fisher Dirichlet energy for the 2-state chain
     (pi, Q_eff), over many random mass-conserving perturbations delta p.

We then summarise max residuals and Dirichlet mismatches over accepted models.
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng
from numpy.linalg import eig, eigh


# --------------------------------------------------------------------
# Basic 2x2 building blocks
# --------------------------------------------------------------------

def pauli_matrices():
    sigma_x = np.array([[0.0, 1.0],
                        [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j],
                        [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0],
                        [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def ladder_operators():
    # |0> = [1,0]^T, |1> = [0,1]^T
    sigma_minus = np.array([[0.0, 0.0],
                            [1.0, 0.0]], dtype=complex)
    sigma_plus = np.array([[0.0, 1.0],
                           [0.0, 0.0]], dtype=complex)
    return sigma_minus, sigma_plus


def build_gkls_superoperator(H, L_list):
    """
    Build GKLS superoperator K acting on vec(rho) with column stacking:

        d/dt vec(rho) = K vec(rho)

    H is 2x2 Hermitian, L_list contains 2x2 Lindblad operators.
    """
    H = np.asarray(H, dtype=complex)
    N = H.shape[0]
    dim = N * N
    I_N = np.eye(N, dtype=complex)

    K = np.zeros((dim, dim), dtype=complex)

    # Hamiltonian part: -i [H, rho]
    K_H = -1j * (np.kron(I_N, H) - np.kron(H.T, I_N))
    K += K_H

    # Dissipative part: sum L rho L^dag - 0.5 {L^dag L, rho}
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
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def unvec(v, N):
    return np.asarray(v, dtype=complex).reshape((N, N), order="F")


def find_stationary_state(K, N, tol=1e-12):
    """
    Find a stationary density matrix rho_ss from K:

        K vec(rho_ss) = 0

    using the eigenvector of K with eigenvalue closest to zero.
    Enforce Hermiticity, positivity, and trace 1.
    """
    evals, evecs = eig(K)
    idx = np.argmin(np.abs(evals))
    v_ss = evecs[:, idx]
    rho_ss = unvec(v_ss, N)
    # Hermitise
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    # Diagonalise and clip spectrum
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
    at rho = diag(pi), in the eigenbasis of rho_ss.

    For 2 levels:

      c_ii = 1 / pi_i
      c_ij = (log pi_i - log pi_j) / (pi_i - pi_j)  (i != j)
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
    Classical Fisher-Dirichlet for a 2 state chain with generator Q_eff
    and stationary distribution pi (2-vector).

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


def gkls_dirichlet_density(delta_p, G_eig, M_diag, N=2):
    """
    GKLS density sector Dirichlet energy in rho_ss eigenbasis:

        E_GKLS = - <delta u, M G_eig delta u>

    where delta u = vec(diag(delta_p)) in that basis (length 4 for N=2),
    M is diagonal with entries M_diag in the E_ij basis, and G_eig is the
    symmetric part of K^sharp.
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
    sigma_x, sigma_y, sigma_z = pauli_matrices()
    sigma_minus, sigma_plus = ladder_operators()

    N = 2

    # Try a few times to get a well-conditioned rho_ss
    max_tries = 10
    for attempt in range(max_tries):
        # Random Hamiltonian
        h = rng.normal(size=3)
        H = 0.5 * (h[0] * sigma_x + h[1] * sigma_y + h[2] * sigma_z)

        # Random positive rates for sigma_minus, sigma_plus, sigma_z
        # Sample in a moderate range to avoid extremely stiff cases.
        gamma1 = rng.uniform(0.2, 1.5)
        gamma2 = rng.uniform(0.2, 1.5)
        gamma3 = rng.uniform(0.2, 1.5)

        L1 = np.sqrt(gamma1) * sigma_minus
        L2 = np.sqrt(gamma2) * sigma_plus
        L3 = np.sqrt(gamma3) * sigma_z
        L_list = [L1, L2, L3]

        K = build_gkls_superoperator(H, L_list)

        # Find stationary state
        rho_ss = find_stationary_state(K, N)
        # Diagonalise
        vals, U = eigh(rho_ss)
        pi = np.clip(vals.real, 0.0, None)
        pi_sum = pi.sum()
        if pi_sum <= 0.0:
            continue
        pi = pi / pi_sum

        # Require full rank and nondegenerate spectrum
        if np.min(pi) < pi_min:
            continue
        if abs(pi[0] - pi[1]) < gap_min:
            continue

        # Also check that rho_ss has non-trivial coherence in computational basis
        coh_norm = np.linalg.norm(rho_ss - np.diag(np.diag(rho_ss)))
        # We do not require it to be large, but record it.

        # Transform K to rho_ss eigenbasis
        T = np.kron(U.conj(), U)
        T_inv = np.kron(U.T, U.conj().T)
        K_eig = T_inv @ K @ T

        # Check stationarity in eigenbasis
        rho_ss_eig = np.diag(pi)
        stat_resid = float(np.linalg.norm(K_eig @ vec(rho_ss_eig)))

        # Extract Q_eff
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
        col_resid = float(np.linalg.norm(col_sums, ord=np.inf))
        stat_resid_Q = float(np.linalg.norm(Q_eff @ pi))

        # Reconstruct rates and detailed balance residual
        w_eff = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j:
                    w_eff[i, j] = Q_eff[j, i]
        db_resid = abs(pi[0] * w_eff[0, 1] - pi[1] * w_eff[1, 0])

        # BKM metric and symmetric part G_eig
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
            # Random mass-conserving perturbation
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
            "coh_norm": float(coh_norm),
            "pi": pi,
            "stat_resid_eig": stat_resid,
            "col_resid_Q": col_resid,
            "stat_resid_Q": stat_resid_Q,
            "db_resid": db_resid,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "min_Ec": min_Ec,
            "min_Eg": min_Eg,
            "max_Ec": max_Ec,
            "max_Eg": max_Eg,
        }

    # If we get here, no acceptable rho_ss was found
    return {
        "accepted": False,
        "attempts": max_tries,
    }


# --------------------------------------------------------------------
# Main ensemble driver
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Random qubit GKLS ensemble: density-sector Fisher–Markov checks."
    )
    parser.add_argument("--n_models", type=int, default=20,
                        help="Number of random GKLS models to attempt.")
    parser.add_argument("--pi_min", type=float, default=1e-3,
                        help="Minimum eigenvalue of rho_ss to accept.")
    parser.add_argument("--gap_min", type=float, default=5e-3,
                        help="Minimum spectral gap |pi0 - pi1| of rho_ss to accept.")
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

    print("23_gkls_random_qubit_density_ensemble.py")
    print("----------------------------------------")
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
        print("No acceptable models found; adjust pi_min/gap_min or parameters.")
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
