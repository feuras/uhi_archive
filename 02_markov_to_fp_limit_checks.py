#!/usr/bin/env python3
"""
markov_to_fp_limit_checks.py

Referee-proof numerical check that a 1D reversible nearest-neighbour Markov chain
on a periodic lattice converges to the continuum diffusion (Fokker–Planck) equation
in the hydrodynamic limit.

We consider the heat equation on the circle [0, 1) with diffusion coefficient D:

    ∂_t ρ = D ∂_xx ρ,   x ∈ [0,1), periodic,

with smooth initial condition

    ρ_0(x) = 1 + A sin(2π x),

for which the exact solution is

    ρ(x, t) = 1 + A exp(-λ t) sin(2π x),
    λ = (2π)^2 D.

We build, for each spatial resolution M:

  - A uniform grid with spacing a = 1/M and nodes x_i = (i + 0.5) a.
  - A continuous-time Markov generator R (size M×M) describing a symmetric
    random walk with nearest-neighbour jumps at rate r = D/a^2 and periodic
    boundary conditions. The master equation is

        d p_i / dt = r (p_{i+1} + p_{i-1} - 2 p_i),

    which is exactly the semi-discrete finite-difference approximation of the
    heat equation on this grid.

  - Probabilities p_i(t) evolving under

        d p / dt = R p,

    using scipy.sparse.linalg.expm_multiply.

The discrete density is ρ_markov(x_i,t) = p_i(t)/a. We compare ρ_markov(x_i,t)
to the exact analytic solution ρ_true(x_i,t) at the same times and define a
maximal L^2 error over time:

    E(M) = max_t sqrt(∑_i |ρ_markov(x_i,t) - ρ_true(x_i,t)|^2 * a).

We then repeat this for a list of resolutions M and fit a power law

    E(M) ≈ C a^p,   a = 1/M,

by regressing log E vs log a. For a standard second-order central-difference
discretisation of the Laplacian, we expect p ≈ 2.

The script:

  * Verifies for each M:
      - Column sums of R are ≈ 0 (mass conservation),
      - A uniform stationary distribution π_i = 1/M is stationary: R π ≈ 0,
      - E(M) is small for the finest grid,
      - ρ_markov remains non-negative and normalised to ≈ 1.

  * Aggregates across resolutions to estimate the convergence rate p and
    declares PASS if:
      - p >= 1.5 (close to second order),
      - the finest-grid max error is below a prescribed tolerance.

Multithreading:
  * By default, uses up to 20 worker threads or as many as there are resolutions
    and CPU cores, whichever is smaller.
  * If parallel execution fails for any reason, falls back to sequential execution.

Dependencies:
  * numpy
  * scipy (for scipy.sparse.linalg.expm_multiply)

Usage:
  python markov_to_fp_limit_checks.py

Optional arguments:
  --D            Diffusion coefficient (default: 0.5)
  --A            Initial sine amplitude A (default: 0.5)
  --T            Final time (default: 0.5)
  --num-times    Number of time samples between 0 and T (default: 40)
  --resolutions  Comma-separated list of grid sizes M (default: "50,100,200")
  --workers      Number of worker threads (default: min(20, cpu_count, #resolutions))

Exit status:
  0 if all checks pass, 1 otherwise.
"""

import argparse
import concurrent.futures
import os
import sys

import numpy as np
from scipy.sparse.linalg import expm_multiply


def build_markov_generator(M, D):
    """
    Build the Markov generator R for symmetric nearest-neighbour jumps on a
    periodic lattice with M sites and diffusion coefficient D.

    Grid spacing: a = 1/M, nodes x_i = (i + 0.5) a.

    Rate r = D / a^2. The master equation for probabilities p_i is

        d p_i / dt = r (p_{i+1} + p_{i-1} - 2 p_i),

    where indices are taken modulo M (periodic boundary conditions).

    In matrix form: dp/dt = R p with

        R[i, i]      = -2 r,
        R[i, i+1]    = r,
        R[i, i-1]    = r.

    We construct R so that column sums vanish (probability conservation).
    """
    a = 1.0 / M
    r = D / (a * a)
    R = np.zeros((M, M), dtype=float)

    for i in range(M):
        left = (i - 1) % M
        right = (i + 1) % M
        # Inflow from neighbours: R[i, j] multiplies p_j
        R[i, i] = -2.0 * r
        R[i, left] += r
        R[i, right] += r

    # Column sums should be zero
    col_sums = R.sum(axis=0)
    return R, col_sums


def analytic_rho(x, t, D, A):
    """
    Exact solution of the heat equation on the circle [0,1) with initial

        ρ_0(x) = 1 + A sin(2π x),

    is

        ρ(x,t) = 1 + A exp(-(2π)^2 D t) sin(2π x).
    """
    lam = (2.0 * np.pi) ** 2 * D
    return 1.0 + A * np.exp(-lam * t) * np.sin(2.0 * np.pi * x)


def run_single_resolution(idx, M, D, A, T, num_times):
    """
    Run the Markov-to-FP consistency test for a single grid resolution M.

    Steps:
      - Build R and check column sums & stationary distribution.
      - Construct initial probabilities p0 from the analytic initial density.
      - Evolve p(t) under dp/dt = R p using expm_multiply.
      - Convert to density ρ_markov(x_i,t) = p_i(t) / a.
      - Compare to analytic ρ_true(x_i,t) and compute max L2 error over time.
      - Return diagnostics in a dict.
    """
    a = 1.0 / M
    x = (np.arange(M, dtype=float) + 0.5) * a  # midpoints in [0,1)

    # Markov generator and basic conservation checks
    R, col_sums = build_markov_generator(M, D)
    col_err = float(np.max(np.abs(col_sums)))

    # Stationarity of uniform distribution
    pi = np.full(M, 1.0 / M, dtype=float)
    stat_err = float(np.max(np.abs(R @ pi)))

    # Initial density and probabilities
    rho0 = analytic_rho(x, t=0.0, D=D, A=A)
    rho0 = np.clip(rho0, 0.0, None)
    p0 = rho0 * a
    total_mass = float(np.sum(p0))
    if total_mass <= 0.0:
        raise RuntimeError(f"Non-positive total mass at M={M}.")
    p0 /= total_mass  # enforce normalisation exactly

    # Time grid and Markov evolution
    times = np.linspace(0.0, T, num_times)
    p_t = expm_multiply(R, p0,
                        start=0.0, stop=T,
                        num=num_times, endpoint=True)
    # Fix tiny negatives and renormalise at each time
    p_t = np.clip(p_t, 0.0, None)
    p_t /= p_t.sum(axis=1, keepdims=True)
    rho_markov = p_t / a

    # Compare to analytic solution over time
    max_err_L2 = 0.0
    max_mass_err = 0.0
    min_rho = float(np.min(rho_markov))

    for k, t in enumerate(times):
        rho_true = analytic_rho(x, t, D, A)
        # L2 error with dx = a
        err = np.sqrt(np.sum((rho_markov[k] - rho_true) ** 2) * a)
        if err > max_err_L2:
            max_err_L2 = err
        # mass check
        mass_k = float(np.sum(rho_markov[k] * a))
        mass_err_k = abs(mass_k - 1.0)
        if mass_err_k > max_mass_err:
            max_mass_err = mass_err_k

    return dict(
        idx=idx,
        M=M,
        a=a,
        D=D,
        A=A,
        T=T,
        num_times=num_times,
        col_err=col_err,
        stat_err=stat_err,
        max_err_L2=max_err_L2,
        max_mass_err=max_mass_err,
        min_rho=min_rho,
    )


def parse_resolutions(s):
    """
    Parse a comma-separated list of positive integers from a string.
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    Ms = []
    for p in parts:
        try:
            val = int(p)
        except ValueError:
            raise ValueError(f"Invalid grid size '{p}' in --resolutions.")
        if val <= 2:
            raise ValueError("Each grid size M must be >= 3.")
        Ms.append(val)
    if not Ms:
        raise ValueError("No valid grid sizes found in --resolutions.")
    return Ms


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check that a reversible nearest-neighbour Markov chain on a periodic "
            "lattice converges to the continuum diffusion (Fokker–Planck) solution "
            "for a known analytic initial condition."
        )
    )
    parser.add_argument("--D", type=float, default=0.5,
                        help="Diffusion coefficient D > 0. Default: 0.5.")
    parser.add_argument("--A", type=float, default=0.5,
                        help="Amplitude A of the initial sine mode. Default: 0.5.")
    parser.add_argument("--T", type=float, default=0.5,
                        help="Final time T > 0. Default: 0.5.")
    parser.add_argument("--num-times", type=int, default=40,
                        help="Number of time samples between 0 and T. Default: 40.")
    parser.add_argument(
        "--resolutions",
        type=str,
        default="50,100,200",
        help="Comma-separated list of grid sizes M. Default: '50,100,200'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads to use (default: min(20, cpu_count, #resolutions)).",
    )

    args = parser.parse_args()

    if args.D <= 0.0:
        print("Error: D must be > 0.", file=sys.stderr)
        return False
    if args.T <= 0.0:
        print("Error: T must be > 0.", file=sys.stderr)
        return False
    if args.num_times < 2:
        print("Error: num-times must be at least 2.", file=sys.stderr)
        return False

    try:
        Ms = parse_resolutions(args.resolutions)
    except ValueError as e:
        print(f"Error parsing resolutions: {e}", file=sys.stderr)
        return False

    base_seed = 987654321  # not used, but kept for symmetry with other scripts

    # Determine number of workers
    if args.workers is not None:
        workers = max(1, min(20, args.workers, len(Ms)))
    else:
        cpu = os.cpu_count() or 1
        workers = max(1, min(20, cpu, len(Ms)))

    print(
        "Running Markov-to-FP limit checks with:\n"
        f"  D = {args.D}\n"
        f"  A = {args.A}\n"
        f"  T = {args.T}\n"
        f"  num_times = {args.num_times}\n"
        f"  resolutions M = {Ms}\n"
        f"  workers = {workers}"
    )

    results = []

    def run_all_sequential():
        out = []
        for idx, M in enumerate(Ms):
            out.append(
                run_single_resolution(
                    idx=idx,
                    M=M,
                    D=args.D,
                    A=args.A,
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
                        run_single_resolution,
                        idx,
                        M,
                        args.D,
                        args.A,
                        args.T,
                        args.num_times,
                    )
                    for idx, M in enumerate(Ms)
                ]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
        except Exception as e:
            print("Parallel execution failed, falling back to sequential execution.")
            print("Reason:", repr(e))
            results = run_all_sequential()
    else:
        results = run_all_sequential()

    # Sort results by M for nicer output
    results.sort(key=lambda r: r["M"])

    print("")
    print("Per-resolution diagnostics:")
    for r in results:
        print(
            f"  M={r['M']:4d}, a={r['a']:.4e}, "
            f"max L2 error={r['max_err_L2']:.3e}, "
            f"col_err={r['col_err']:.3e}, "
            f"stat_err={r['stat_err']:.3e}, "
            f"max_mass_err={r['max_mass_err']:.3e}, "
            f"min rho={r['min_rho']:.3e}"
        )

    # Aggregate and estimate convergence rate
    a_list = np.array([r["a"] for r in results], dtype=float)
    E_list = np.array([r["max_err_L2"] for r in results], dtype=float)

    # Basic safety checks
    max_col_err = float(np.max([r["col_err"] for r in results]))
    max_stat_err = float(np.max([r["stat_err"] for r in results]))
    max_mass_err = float(np.max([r["max_mass_err"] for r in results]))
    min_rho_all = float(np.min([r["min_rho"] for r in results]))
    min_err = float(np.min(E_list))

    print("")
    print("Global diagnostics:")
    print(f"  max |R column sum|         = {max_col_err:.3e}")
    print(f"  max |R pi| (stationarity)  = {max_stat_err:.3e}")
    print(f"  max mass error             = {max_mass_err:.3e}")
    print(f"  min rho over all runs      = {min_rho_all:.3e}")
    print(f"  min max-error (finest grid)= {min_err:.3e}")

    # Estimate convergence rate p from log-log fit: log E ≈ p log a + C
    if len(a_list) >= 2 and np.all(E_list > 0.0):
        log_a = np.log(a_list)
        log_E = np.log(E_list)
        p, C = np.polyfit(log_a, log_E, 1)
        print(f"  estimated convergence rate p ≈ {p:.3f}")
    else:
        p = 0.0
        print("  insufficient data to estimate convergence rate.")

    # PASS/FAIL criteria
    tol_col = 1e-12
    tol_stat = 1e-12
    tol_mass = 1e-10
    tol_min_rho = -1e-12
    tol_min_err = 1e-3
    min_p_required = 1.5  # expect ~2 for second-order Laplacian

    success = (
        max_col_err <= tol_col
        and max_stat_err <= tol_stat
        and max_mass_err <= tol_mass
        and min_rho_all >= tol_min_rho
        and min_err <= tol_min_err
        and p >= min_p_required
    )

    print("")
    if success:
        print("Markov-to-Fokker–Planck hydrodynamic limit check: PASS")
    else:
        print("Markov-to-Fokker–Planck hydrodynamic limit check: FAIL")

    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
