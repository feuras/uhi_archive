#!/usr/bin/env python3
"""
04_gkls_coherence_elimination_checks.py

Numerical demonstration of coherence elimination in a nontrivial GKLS qubit.

Model:
    Qubit with Hamiltonian
        H = (omega/2) * sigma_x

    and pure dephasing Lindblad operator
        L = sqrt(gamma) * sigma_z,

    so the master equation is
        dρ/dt = -i [H, ρ] + gamma (sigma_z ρ sigma_z - ρ).

In Bloch coordinates (x, y, z) with
    ρ = (1/2) (I + x σ_x + y σ_y + z σ_z),
the equations of motion are
    dx/dt = -2 gamma x
    dy/dt = -2 gamma y - omega z
    dz/dt =  omega y.

This is a genuinely coherent GKLS model: Hamiltonian generates rotations
about the x axis, while the Lindblad operator damps coherences in the z basis.

In the strong dephasing regime gamma >> omega, one can adiabatically eliminate
the fast-decaying coherence y to obtain an effective slow dynamics for z:
    y ≈ -(omega / (2 gamma)) z,
    dz/dt ≈ -(omega^2 / (2 gamma)) z.

This is exactly the equation for a symmetric two-state Markov chain on the
σ_z eigenbasis with effective rate
    a_eff = omega^2 / (4 gamma),
since for a symmetric chain the population difference z obeys
    dz/dt = -2 a_eff z.

This script numerically verifies that:

  1. For large gamma, the exact GKLS evolution of z(t) is well approximated
     by the effective Markov prediction
        z_eff(t) = exp(-kappa t),   kappa = omega^2 / (2 gamma),
     after an initial decoherence transient.

  2. Coherences y(t) are small in the slow tail, consistent with adiabatic
     elimination.

  3. In the tail, the quantum relative entropy S_q(t) of ρ(t) with respect
     to the maximally mixed state I/2 is close to the classical relative
     entropy S_eff(t) of the effective two-state chain with populations
     p_eff(t) = ((1+z_eff(t))/2, (1-z_eff(t))/2) against the uniform target.

We run this for a list of gamma values, and enforce strict tolerances only
on the largest gamma (deep dephasing regime). Smaller gammas are reported
but not used for PASS/FAIL gating.

Implementation details:

  - State: Bloch vector v = (x, y, z) obeying dv/dt = A v for a 3x3 matrix A.
  - Solution: v(t) = exp(A t) v(0) computed via scipy.sparse.linalg.expm_multiply.
  - Initial state: excited σ_z eigenstate |1><1|, so v(0) = (0, 0, 1).
  - Time window: for each gamma, we choose T_end = T_factor / kappa with
    T_factor > 1 to see several slow decay times, and we discard an initial
    transient of length t_cut ≈ c / gamma (several dephasing times).
  - Quantum relative entropy: for a qubit with eigenvalues λ_{±} = (1 ± r)/2
    where r = sqrt(x^2 + y^2 + z^2), the relative entropy to I/2 is
        S_q = sum_i λ_i log(2 λ_i).
  - Classical relative entropy: for p_eff = (p1, p0) and π = (1/2, 1/2),
        S_eff = sum_i p_i log(2 p_i).

Multithreading:
  - Uses ThreadPoolExecutor with up to 20 workers by default (or a user
    supplied limit). If any error occurs during parallel execution, falls
    back to sequential evaluation.

Dependencies:
  - numpy
  - scipy (for scipy.sparse.linalg.expm_multiply)

Usage:
  python 04_gkls_coherence_elimination_checks.py

Optional arguments:
  --omega        Hamiltonian frequency omega (default: 1.0)
  --gammas       Comma-separated list of gamma values (default: "1.0,2.0,4.0,8.0")
  --T-factor     Dimensionless tail length in units of slow timescale 1/kappa
                 (default: 8.0)
  --num-times    Number of time samples (default: 400)
  --workers      Number of worker threads (default: min(20, cpu_count))

Exit status:
  0 if all checks pass, 1 otherwise.
"""

import argparse
import concurrent.futures
import os
import sys

import numpy as np
from scipy.sparse.linalg import expm_multiply


def build_bloch_generator(omega, gamma):
    """
    Build the 3x3 generator matrix A for the Bloch vector v = (x, y, z):

        dx/dt = -2 gamma x
        dy/dt = -2 gamma y - omega z
        dz/dt =  omega y.

    So A is:

        A = [[-2 gamma,    0,      0],
             [    0,   -2 gamma, -omega],
             [    0,     omega,    0   ]].
    """
    A = np.array(
        [
            [-2.0 * gamma, 0.0, 0.0],
            [0.0, -2.0 * gamma, -omega],
            [0.0, omega, 0.0],
        ],
        dtype=float,
    )
    return A


def bloch_to_density(x, y, z):
    """
    Convert Bloch coordinates (x, y, z) to a 2x2 density matrix:

        ρ = 0.5 * [[1+z, x - i y],
                   [x + i y, 1 - z]].
    """
    rho = 0.5 * np.array(
        [
            [1.0 + z, x - 1j * y],
            [x + 1j * y, 1.0 - z],
        ],
        dtype=complex,
    )
    return rho


def quantum_relative_entropy_to_max_mix(x, y, z):
    """
    Quantum relative entropy D(ρ || I/2) for a qubit in Bloch form.

    If ρ has Bloch radius r = sqrt(x^2 + y^2 + z^2), then the eigenvalues are

        λ_± = (1 ± r) / 2.

    For the reference state π = I/2 (eigenvalues 1/2, 1/2), the relative
    entropy is

        S_q = sum_i λ_i log(λ_i / (1/2)) = sum_i λ_i log(2 λ_i).

    We clamp r into [0,1] for numerical safety.
    """
    r = np.sqrt(x * x + y * y + z * z)
    r_clamped = float(np.clip(r, 0.0, 1.0))
    lam_plus = 0.5 * (1.0 + r_clamped)
    lam_minus = 0.5 * (1.0 - r_clamped)
    lams = np.array([lam_plus, lam_minus], dtype=float)
    lams = np.clip(lams, 0.0, 1.0)
    mask = lams > 0.0
    S = float(np.sum(lams[mask] * np.log(2.0 * lams[mask])))
    return S


def classical_relative_entropy_two_state(z):
    """
    Classical relative entropy D(p || π) for a two-state chain with populations

        p = ((1+z)/2, (1-z)/2),

    against the uniform target π = (1/2, 1/2):

        S_eff = sum_i p_i log(p_i / (1/2)) = sum_i p_i log(2 p_i).
    """
    p1 = 0.5 * (1.0 + z)
    p0 = 0.5 * (1.0 - z)
    p = np.array([p1, p0], dtype=float)
    p = np.clip(p, 0.0, 1.0)
    mask = p > 0.0
    S = float(np.sum(p[mask] * np.log(2.0 * p[mask])))
    return S


def run_single_gamma(idx, gamma, omega, T_factor, num_times):
    """
    Run the coherence-elimination check for a single gamma value.

    Steps:
      1. Build Bloch generator A for given (omega, gamma).
      2. Determine slow rate kappa = omega^2 / (2 gamma) and T_end = T_factor / kappa.
      3. Solve v(t) = exp(A t) v0 for v0 = (0, 0, 1).
      4. Compute exact z(t), y(t).
      5. Compute effective Markov prediction z_eff(t) = exp(-kappa t).
      6. Define a decoherence transient cutoff t_cut ~ c / gamma and evaluate:

         - max absolute error of z vs z_eff on t >= t_cut,
         - max ratio of |y| to |z| on t >= t_cut,
         - tail discrepancy between S_q(t) and S_eff(t).

    Returns a dict of diagnostics.
    """
    gamma = float(gamma)
    omega = float(omega)

    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    if omega <= 0.0:
        raise ValueError("omega must be positive.")

    # Slow rate and time window
    kappa = omega * omega / (2.0 * gamma)
    T_end = T_factor / kappa

    # Time grid
    times = np.linspace(0.0, T_end, num_times)

    # Build generator and initial Bloch vector v0 = (x0, y0, z0) = (0, 0, 1)
    A = build_bloch_generator(omega, gamma)
    v0 = np.array([0.0, 0.0, 1.0], dtype=float)

    # Evolve using expm_multiply
    v_t = expm_multiply(A, v0, start=0.0, stop=T_end, num=num_times, endpoint=True)
    v_t = np.array(v_t, dtype=float)
    x_t = v_t[:, 0]
    y_t = v_t[:, 1]
    z_t = v_t[:, 2]

    # Effective Markov prediction for z
    z_eff_t = np.exp(-kappa * times)

    # Decoherence transient cutoff: several times the dephasing timescale ~ 1/(2 gamma)
    # Choose t_cut = c / gamma with c = 5.
    t_cut = 5.0 / gamma
    mask_tail = times >= t_cut
    if not np.any(mask_tail):
        # If grid too short, use the last half of times as tail
        half = num_times // 2
        mask_tail = np.zeros(num_times, dtype=bool)
        mask_tail[half:] = True

    # Population comparison in tail
    z_tail = z_t[mask_tail]
    z_eff_tail = z_eff_t[mask_tail]
    abs_err_tail = np.abs(z_tail - z_eff_tail)
    max_abs_err_tail = float(np.max(abs_err_tail))

    # Avoid ill-defined relative error at very small z_eff
    # Restrict to points where |z_eff| is not extremely small.
    mask_rel = np.logical_and(mask_tail, np.abs(z_eff_t) > 1e-4)
    if np.any(mask_rel):
        rel_err_tail = np.abs(z_t[mask_rel] - z_eff_t[mask_rel]) / np.abs(z_eff_t[mask_rel])
        max_rel_err_tail = float(np.max(rel_err_tail))
    else:
        max_rel_err_tail = 0.0

    # Coherence vs population in tail
    y_tail = y_t[mask_tail]
    max_abs_y_tail = float(np.max(np.abs(y_tail)))
    max_abs_z_tail = float(np.max(np.abs(z_tail)))
    if max_abs_z_tail > 0.0:
        y_over_z_tail = max_abs_y_tail / max_abs_z_tail
    else:
        y_over_z_tail = 0.0

    # Quantum and classical relative entropies in tail
    S_q_tail = []
    S_eff_tail = []
    for k, t in enumerate(times):
        if not mask_tail[k]:
            continue
        x = x_t[k]
        y = y_t[k]
        z = z_t[k]
        S_q = quantum_relative_entropy_to_max_mix(x, y, z)
        S_eff = classical_relative_entropy_two_state(z_eff_t[k])
        S_q_tail.append(S_q)
        S_eff_tail.append(S_eff)

    S_q_tail = np.array(S_q_tail, dtype=float)
    S_eff_tail = np.array(S_eff_tail, dtype=float)
    if S_q_tail.size > 0:
        S_diff_tail = np.abs(S_q_tail - S_eff_tail)
        max_S_diff_tail = float(np.max(S_diff_tail))
    else:
        max_S_diff_tail = 0.0

    return dict(
        idx=idx,
        gamma=gamma,
        omega=omega,
        kappa=kappa,
        T_end=T_end,
        t_cut=t_cut,
        num_times=num_times,
        max_abs_err_z_tail=max_abs_err_tail,
        max_rel_err_z_tail=max_rel_err_tail,
        y_over_z_tail=y_over_z_tail,
        max_S_diff_tail=max_S_diff_tail,
    )


def parse_gammas(s):
    """
    Parse a comma-separated list of positive floats from a string.
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    gammas = []
    for p in parts:
        try:
            val = float(p)
        except ValueError:
            raise ValueError(f"Invalid gamma value '{p}' in --gammas.")
        if val <= 0.0:
            raise ValueError("All gamma values must be > 0.")
        gammas.append(val)
    if not gammas:
        raise ValueError("No valid gamma values found in --gammas.")
    return gammas


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check that a coherent GKLS qubit with H=(omega/2) sigma_x and "
            "L=sqrt(gamma) sigma_z admits an effective two-state Markov "
            "description for its populations in the strong-dephasing regime."
        )
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Hamiltonian frequency omega > 0. Default: 1.0.",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default="1.0,2.0,4.0,8.0",
        help="Comma-separated list of gamma values > 0. Default: '1.0,2.0,4.0,8.0'.",
    )
    parser.add_argument(
        "--T-factor",
        type=float,
        default=8.0,
        help=(
            "Dimensionless tail length factor. Evolution runs up to "
            "T_end = T_factor / kappa with kappa = omega^2/(2 gamma). "
            "Default: 8.0."
        ),
    )
    parser.add_argument(
        "--num-times",
        type=int,
        default=400,
        help="Number of time samples between 0 and T_end. Default: 400.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: min(20, cpu_count)).",
    )

    args = parser.parse_args()

    if args.omega <= 0.0:
        print("Error: omega must be > 0.", file=sys.stderr)
        return False
    if args.T_factor <= 0.0:
        print("Error: T-factor must be > 0.", file=sys.stderr)
        return False
    if args.num_times < 10:
        print("Error: num-times must be at least 10.", file=sys.stderr)
        return False

    try:
        gammas = parse_gammas(args.gammas)
    except ValueError as e:
        print(f"Error parsing gammas: {e}", file=sys.stderr)
        return False

    if args.workers is not None:
        workers = max(1, min(20, args.workers))
    else:
        cpu = os.cpu_count() or 1
        workers = max(1, min(20, cpu))

    print(
        "Running GKLS coherence-elimination checks with:\n"
        f"  omega = {args.omega}\n"
        f"  gammas = {gammas}\n"
        f"  T_factor = {args.T_factor}\n"
        f"  num_times = {args.num_times}\n"
        f"  workers = {workers}"
    )

    results = []

    def run_all_sequential():
        out = []
        for idx, gamma in enumerate(gammas):
            out.append(
                run_single_gamma(
                    idx=idx,
                    gamma=gamma,
                    omega=args.omega,
                    T_factor=args.T_factor,
                    num_times=args.num_times,
                )
            )
        return out

    if workers > 1:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        run_single_gamma,
                        idx,
                        gamma,
                        args.omega,
                        args.T_factor,
                        args.num_times,
                    )
                    for idx, gamma in enumerate(gammas)
                ]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
        except Exception as e:
            print("Parallel execution failed, falling back to sequential execution.")
            print("Reason:", repr(e))
            results = run_all_sequential()
    else:
        results = run_all_sequential()

    # Sort by gamma for nice output
    results.sort(key=lambda r: r["gamma"])

    print("")
    print("Per-gamma diagnostics:")
    for r in results:
        print(
            f"  gamma={r['gamma']:5.2f}, "
            f"kappa={r['kappa']:.4e}, "
            f"T_end={r['T_end']:.4e}, "
            f"t_cut={r['t_cut']:.4e}, "
            f"max_abs_err_z_tail={r['max_abs_err_z_tail']:.3e}, "
            f"max_rel_err_z_tail={r['max_rel_err_z_tail']:.3e}, "
            f"y_over_z_tail={r['y_over_z_tail']:.3e}, "
            f"max_S_diff_tail={r['max_S_diff_tail']:.3e}"
        )

    # Use the largest gamma (deepest dephasing) for strict PASS/FAIL criteria
    gamma_max = max(gammas)
    r_star = None
    for r in results:
        if abs(r["gamma"] - gamma_max) < 1e-12:
            r_star = r
            break

    if r_star is None:
        print("Internal error: could not identify largest-gamma result.", file=sys.stderr)
        return False

    print("")
    print("Gating diagnostics at largest gamma:")
    print(
        f"  gamma_max={r_star['gamma']:.2f}, "
        f"max_abs_err_z_tail={r_star['max_abs_err_z_tail']:.3e}, "
        f"max_rel_err_z_tail={r_star['max_rel_err_z_tail']:.3e}, "
        f"y_over_z_tail={r_star['y_over_z_tail']:.3e}, "
        f"max_S_diff_tail={r_star['max_S_diff_tail']:.3e}"
    )

    # Thresholds for deep-dephasing regime (largest gamma)
    tol_abs_z = 5e-2       # absolute population error in tail
    tol_rel_z = 3e-1       # relative error in tail
    tol_y_over_z = 2e-1    # coherence small compared to population in tail
    tol_S_diff = 1e-1      # entropy curves close in tail

    success = (
        r_star["max_abs_err_z_tail"] <= tol_abs_z
        and r_star["max_rel_err_z_tail"] <= tol_rel_z
        and r_star["y_over_z_tail"] <= tol_y_over_z
        and r_star["max_S_diff_tail"] <= tol_S_diff
    )

    print("")
    if success:
        print("GKLS coherence-elimination effective Markov limit check: PASS")
    else:
        print("GKLS coherence-elimination effective Markov limit check: FAIL")

    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
