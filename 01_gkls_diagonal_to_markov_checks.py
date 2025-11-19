#!/usr/bin/env python3
"""
gkls_diagonal_to_markov_checks.py

Referee proof numerical check that a finite dimensional GKLS generator with
diagonal jump structure

    L_ij = sqrt(k_ij) |i><j|,

reduces exactly to a classical reversible Markov chain on the diagonal sector,
and that the quantum relative entropy decay S(rho_t || pî) matches the
classical relative entropy decay D_KL(p(t) || pi) to machine precision.

The script:

  1. Draws a random strictly positive stationary distribution pi over N levels.
  2. Constructs a symmetric "conductance" matrix W_ij > 0 and defines rates

        k_ij = W_ij / pi_j  for i != j,

     which ensures detailed balance and makes pi stationary for the chain.
  3. Builds the classical Markov generator R from k_ij.
  4. Builds the GKLS superoperator L_super corresponding to jump operators
     L_ij = sqrt(k_ij) |i><j|.
  5. Evolves a random diagonal initial state rho_0 = diag(p0) under the GKLS
     semigroup and under the Markov semigroup, using expm_multiply for both.
  6. Compares:
        - p_quantum(t) = diag(rho_t) vs p_classical(t),
        - S_quantum(t) vs S_classical(t),
        - column sums of R, and R @ pi,
        - positivity and mass conservation,
        - monotonicity of the entropy curves.

The script runs multiple random test instances in parallel using up to 20
worker threads (or fewer if fewer cores are available). If parallel
execution fails for any reason, it falls back to sequential execution.

Dependencies:
    - numpy
    - scipy (for scipy.sparse.linalg.expm_multiply)

Usage:
    python gkls_diagonal_to_markov_checks.py

Optional arguments:
    --N           Hilbert space dimension (default: 5)
    --tests       Number of random test instances (default: 20)
    --T           Final time for evolution (default: 5.0)
    --num-times   Number of time samples (default: 80)
    --workers     Number of worker threads (default: min(20, cpu_count))

Exit status:
    0 if all checks pass, 1 otherwise.
"""

import argparse
import concurrent.futures
import os
import sys

import numpy as np
from scipy.sparse.linalg import expm_multiply


def rel_entropy(p, pi):
    """
    Classical relative entropy (Kullback–Leibler divergence)

        D(p || pi) = sum_i p_i log(p_i / pi_i),

    with natural logarithms. Assumes pi_i > 0. Any tiny negative entries in p
    should be cleaned before calling (we clip to zero upstream).
    """
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(pi[mask]))))


def build_k_matrix(N, rng):
    """
    Construct a random detailed balance set of rates k_ij >= 0 on N states.

    Steps:
      - Draw a random strictly positive stationary distribution pi.
      - Draw a symmetric "conductance" matrix W_ij > 0 with zero diagonal.
      - Set rates

            k_ij = W_ij / pi_j  for i != j,   k_ii = 0,

        which ensures detailed balance with respect to pi for the Markov chain:

            pi_j * k_ij = pi_i * k_ji  (through W_ij symmetry).
    """
    pi_raw = rng.random(N)
    pi = pi_raw / pi_raw.sum()

    # Symmetric positive weights W_ij; W_ii = 0
    W = rng.uniform(0.1, 1.0, size=(N, N))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    k = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                k[i, j] = W[i, j] / pi[j]

    return pi, k


def build_R_from_k(k):
    """
    Build the classical Markov generator R from rates k_ij, where k_ij is the
    rate from state j to state i.

    For i != j:
        R_ij = k_ij
    For i == j:
        R_ii = - sum_{j != i} k_ji   (minus total rate out of state i)

    This ensures that the column sums of R vanish (probability conservation).
    """
    N = k.shape[0]
    R = np.zeros((N, N), dtype=float)

    # Off-diagonal entries: inflow rates from j to i
    for i in range(N):
        for j in range(N):
            if i != j:
                R[i, j] = k[i, j]

    # Diagonal entries: minus total outgoing from i
    for i in range(N):
        R[i, i] = -np.sum(k[:, i])

    col_sums = R.sum(axis=0)
    return R, col_sums


def build_lindblad_super_from_k(k):
    """
    Build the GKLS superoperator L_super (in Liouville form) corresponding to
    jump operators

        L_ij = sqrt(k_ij) |i><j|,  for i != j,

    and zero Hamiltonian. We use column-stacking vec(ρ) and the identity

        vec(A ρ B) = (B^T ⊗ A) vec(ρ).

    The dissipator is

        D(ρ) = Σ_ij [ L_ij ρ L_ij† - 0.5 { L_ij† L_ij, ρ } ].

    In Liouville form this becomes

        L_super = Σ_ij [ (L_ij†)^T ⊗ L_ij
                         - 0.5 ( I ⊗ L_ij† L_ij + (L_ij† L_ij)^T ⊗ I ) ].
    """
    N = k.shape[0]
    dim = N * N
    L_super = np.zeros((dim, dim), dtype=complex)
    I = np.eye(N, dtype=complex)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            kij = k[i, j]
            if kij <= 0.0:
                continue

            # L_ij = sqrt(k_ij) |i><j|
            L = np.zeros((N, N), dtype=complex)
            L[i, j] = np.sqrt(kij)

            # D = L† L = k_ij |j><j|
            D = L.conj().T @ L

            # L ρ L† term: (L†)^T ⊗ L
            term1 = np.kron(L.conj(), L)

            # Anticommutator terms: -0.5 (I ⊗ D + D^T ⊗ I)
            term2 = -0.5 * (np.kron(I, D) + np.kron(D.T, I))

            L_super += term1 + term2

    return L_super


def run_single_test(test_id, seed, N=5, T=5.0, num_times=80):
    """
    Run a single random consistency test:

      - Draw pi and k_ij.
      - Build R and L_super.
      - Evolve p0 and rho0 = diag(p0) up to time T on num_times points.
      - Compare p_quantum vs p_class, and S_quantum vs S_class.
      - Check mass conservation, positivity, stationarity, and entropy monotonicity.
    """
    rng = np.random.default_rng(seed)

    # Construct detailed balance rates and stationary distribution
    pi, k = build_k_matrix(N, rng)

    # Classical generator and basic checks (column sums)
    R, col_sums = build_R_from_k(k)
    col_err = float(np.max(np.abs(col_sums)))
    stat_err = float(np.max(np.abs(R @ pi)))

    # Random initial probability vector p0 (strictly positive)
    p0_raw = rng.random(N)
    p0 = p0_raw / p0_raw.sum()

    # Initial density matrix and its vectorisation (column stacking)
    rho0 = np.diag(p0.astype(complex))
    rho0_vec = rho0.reshape(N * N, order="F")

    # Build Lindblad superoperator
    L_super = build_lindblad_super_from_k(k)

    # Time grid with expm_multiply (linear spacing between 0 and T)
    rho_t = expm_multiply(L_super, rho0_vec,
                          start=0.0, stop=T, num=num_times, endpoint=True)

    # Extract quantum populations p_quantum(t) = diag(rho(t))
    p_quantum = np.empty((num_times, N), dtype=float)
    for idx, rv in enumerate(rho_t):
        rho = rv.reshape((N, N), order="F")
        diag = np.clip(np.real(np.diag(rho)), 0.0, None)
        s = diag.sum()
        if s <= 0.0:
            raise RuntimeError("Quantum diagonal lost positivity or mass in test "
                               f"{test_id}.")
        p_quantum[idx] = diag / s

    # Classical evolution: dp/dt = R p using expm_multiply
    p_t = expm_multiply(R, p0,
                        start=0.0, stop=T, num=num_times, endpoint=True)
    p_t = np.clip(p_t, 0.0, None)
    p_t /= p_t.sum(axis=1, keepdims=True)

    # Compare populations
    diff_p = p_quantum - p_t
    max_diff_p = float(np.max(np.abs(diff_p)))

    # Relative entropy curves (quantum == classical for diagonal states)
    S_q = np.array([rel_entropy(p, pi) for p in p_quantum])
    S_c = np.array([rel_entropy(p, pi) for p in p_t])
    max_diff_S = float(np.max(np.abs(S_q - S_c)))

    # Mass conservation and positivity
    max_mass_q = float(np.max(np.abs(p_quantum.sum(axis=1) - 1.0)))
    max_mass_c = float(np.max(np.abs(p_t.sum(axis=1) - 1.0)))
    min_p_q = float(np.min(p_quantum))
    min_p_c = float(np.min(p_t))

    # Monotonicity of entropy in time
    S_q_monotone = bool(np.all(S_q[1:] <= S_q[:-1] + 1e-10))
    S_c_monotone = bool(np.all(S_c[1:] <= S_c[:-1] + 1e-10))

    return dict(
        test_id=test_id,
        N=N,
        T=T,
        num_times=num_times,
        col_err=col_err,
        stat_err=stat_err,
        max_diff_p=max_diff_p,
        max_diff_S=max_diff_S,
        max_mass_q=max_mass_q,
        max_mass_c=max_mass_c,
        min_p_q=min_p_q,
        min_p_c=min_p_c,
        S_q_monotone=S_q_monotone,
        S_c_monotone=S_c_monotone,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check that a diagonal GKLS model with jump operators "
            "L_ij = sqrt(k_ij)|i><j| reduces exactly to a reversible Markov chain "
            "on the diagonal sector, and that quantum vs classical relative entropy "
            "decay coincide."
        )
    )
    parser.add_argument("--N", type=int, default=5,
                        help="Hilbert space dimension (>= 2). Default: 5.")
    parser.add_argument("--tests", type=int, default=20,
                        help="Number of random test instances. Default: 20.")
    parser.add_argument("--T", type=float, default=5.0,
                        help="Final time for evolution. Default: 5.0.")
    parser.add_argument("--num-times", type=int, default=80,
                        help="Number of time samples between 0 and T. Default: 80.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads to use. "
             "Default: min(20, cpu_count)."
    )

    args = parser.parse_args()

    if args.N < 2:
        print("Error: N must be at least 2.", file=sys.stderr)
        return False

    base_seed = 123456

    # Determine number of workers (up to 20), with fallback to 1
    if args.workers is not None:
        workers = max(1, min(20, args.workers))
    else:
        cpu = os.cpu_count() or 1
        workers = max(1, min(20, cpu))

    print(
        f"Running {args.tests} tests with "
        f"N={args.N}, T={args.T}, num_times={args.num_times}, workers={workers}"
    )

    results = []

    def run_all_sequential():
        out = []
        for i in range(args.tests):
            out.append(
                run_single_test(
                    test_id=i,
                    seed=base_seed + i,
                    N=args.N,
                    T=args.T,
                    num_times=args.num_times,
                )
            )
        return out

    if workers > 1:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        run_single_test,
                        i,
                        base_seed + i,
                        args.N,
                        args.T,
                        args.num_times,
                    )
                    for i in range(args.tests)
                ]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
        except Exception as e:
            print("Parallel execution failed, falling back to sequential execution.")
            print("Reason:", repr(e))
            results = run_all_sequential()
    else:
        results = run_all_sequential()

    # Aggregate statistics over all tests
    max_diff_p = max(r["max_diff_p"] for r in results)
    max_diff_S = max(r["max_diff_S"] for r in results)
    max_mass_q = max(r["max_mass_q"] for r in results)
    max_mass_c = max(r["max_mass_c"] for r in results)
    min_p_q = min(r["min_p_q"] for r in results)
    min_p_c = min(r["min_p_c"] for r in results)
    max_col_err = max(r["col_err"] for r in results)
    max_stat_err = max(r["stat_err"] for r in results)
    all_S_q_monotone = all(r["S_q_monotone"] for r in results)
    all_S_c_monotone = all(r["S_c_monotone"] for r in results)

    print("")
    print("Summary over all tests:")
    print(f"  max |R column sum|          = {max_col_err:.3e}")
    print(f"  max |R pi| (stationarity)   = {max_stat_err:.3e}")
    print(f"  max |p_quantum - p_class|   = {max_diff_p:.3e}")
    print(f"  max |S_quantum - S_class|   = {max_diff_S:.3e}")
    print(f"  max mass error (quantum)    = {max_mass_q:.3e}")
    print(f"  max mass error (classical)  = {max_mass_c:.3e}")
    print(f"  min p (quantum)             = {min_p_q:.3e}")
    print(f"  min p (classical)           = {min_p_c:.3e}")
    print(f"  S_quantum monotone in t?    = {all_S_q_monotone}")
    print(f"  S_classical monotone in t?  = {all_S_c_monotone}")

    # Tolerances for PASS/FAIL (tight but robust)
    tol_col = 1e-12
    tol_stat = 1e-12
    tol_p = 1e-10
    tol_S = 1e-10
    tol_mass = 1e-12

    success = (
        max_col_err <= tol_col
        and max_stat_err <= tol_stat
        and max_diff_p <= tol_p
        and max_diff_S <= tol_S
        and max_mass_q <= tol_mass
        and max_mass_c <= tol_mass
        and min_p_q >= -1e-14
        and min_p_c >= -1e-14
        and all_S_q_monotone
        and all_S_c_monotone
    )

    print("")
    if success:
        print("GKLS diagonal-to-Markov consistency check: PASS")
    else:
        print("GKLS diagonal-to-Markov consistency check: FAIL")

    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
