#!/usr/bin/env python3
"""
33_uih_two_quadratures_visual_explorer.py

Visual explorer for unified reversible + irreversible K-flows in a
finite-dimensional universal information hydrodynamics (UIH) setting.

We work on R^n with a symmetric positive definite metric M and a
metriplectic split

    K = G + J,

with

    M G symmetric negative definite   (dissipative channel),
    M J skew symmetric                 (reversible channel).

We define the quadratic functional

    F(u) = 0.5 * u^T M u,

and visualise:

  * The evolution of F(t) under the full K-flow and the pure G-flow:
        du/dt = K u,   du/dt = G u,
  * The logarithms log F_K(t), log F_G(t), whose slopes encode decay rates,
  * The trajectory projected onto the slowest two dissipative modes of
    the generalised eigenproblem
        (-M G) v = lambda M v.

The reversible channel J should rotate the state on constant-F hypersurfaces
of the M-metric, while G drives the irreversible decay. Asymptotically, the
decay scale is set by the smallest positive dissipative eigenvalue
lambda_min, with an expected F-decay rate of roughly 2 * lambda_min.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def build_random_metric(n: int, seed: int = 123) -> np.ndarray:
    """
    Build a random symmetric positive definite metric matrix M of size n x n.

    We construct M = A^T A and normalise the eigenvalues into a moderate range
    to avoid ill conditioning.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    M = A.T @ A

    # Normalise eigenvalues to lie in [1, kappa_max] to control conditioning
    evals, evecs = la.eigh(M)
    evals = np.clip(evals, 1.0, None)
    M = (evecs * evals) @ evecs.T
    return M


def build_metriplectic_generator(
    M: np.ndarray,
    seed_G: int = 456,
    seed_J: int = 789,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a metric M, construct G and J such that:

        M G is symmetric negative definite (dissipative channel),
        M J is skew symmetric              (reversible channel),

    and return (G, J, K) with K = G + J.
    """
    n = M.shape[0]

    # Dissipative channel: choose S = B^T B positive definite and set
    # M G = -S => G = -M^{-1} S, so M G = -S is symmetric negative.
    rng_G = np.random.default_rng(seed_G)
    B = rng_G.normal(size=(n, n))
    S = B.T @ B  # symmetric positive definite
    G = la.solve(M, -S)  # M G = -S

    # Reversible channel: choose antisymmetric A and set J = M^{-1} A.
    # Then M J = A, so M J + J^T M = A + A^T = 0.
    rng_J = np.random.default_rng(seed_J)
    R = rng_J.normal(size=(n, n))
    A = 0.5 * (R - R.T)  # antisymmetric
    J = la.solve(M, A)

    K = G + J
    return G, J, K


def evolve_linear(K: np.ndarray, u0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Solve du/dt = K u with u(0) = u0 by diagonalisation of K.

    For small dimensions this is efficient and numerically accurate.
    """
    n = K.shape[0]
    u0 = np.asarray(u0, dtype=float).reshape(n)

    evals, evecs = la.eig(K)
    Vinv = la.inv(evecs)

    evals = evals.astype(complex)
    evecs = evecs.astype(complex)
    Vinv = Vinv.astype(complex)

    coeffs0 = Vinv @ u0.astype(complex)
    out = np.zeros((len(times), n), dtype=float)

    for k, t in enumerate(times):
        factors = np.exp(evals * t)
        ut = evecs @ (factors * coeffs0)
        out[k, :] = ut.real

    return out


def energy_F(M: np.ndarray, u: np.ndarray) -> float:
    """
    Entropy-like functional F(u) = 0.5 * u^T M u.
    """
    return 0.5 * float(u.T @ (M @ u))


def main():
    # Core parameters (kept modest for interactive plotting)
    n = 6
    seed_metric = 123
    seed_G = 456
    seed_J = 789

    # Build metric and generators
    M = build_random_metric(n, seed=seed_metric)
    G, J, K = build_metriplectic_generator(M, seed_G=seed_G, seed_J=seed_J)

    # Dissipative spectrum of (-M G, M)
    MG = M @ G
    evals, evecs = la.eig(-MG, M)
    evals = evals.real
    lam_pos = evals[evals > 1e-10]
    lam_pos_sorted = np.sort(lam_pos)

    if lam_pos_sorted.size < 2:
        raise RuntimeError("Need at least two positive dissipative eigenvalues.")

    lam_min = float(lam_pos_sorted[0])
    lam_second = float(lam_pos_sorted[1])

    # Extract eigenvectors for the two slowest dissipative modes
    # Find their indices in the full eigenvalue list
    idx_min = int(np.argmin(np.abs(evals - lam_min)))
    idx_second = int(np.argmin(np.abs(evals - lam_second)))
    v_min = evecs[:, idx_min].real
    v_second = evecs[:, idx_second].real

    # Time grid based on lambda_min
    t_max_factor = 6.0
    t_max = t_max_factor / lam_min
    n_times = 400
    times = np.linspace(0.0, t_max, n_times)

    # Single initial condition: random unit M-norm
    rng_init = np.random.default_rng(seed_metric + seed_G + seed_J)
    u0 = rng_init.normal(size=n)
    norm_M = float(np.sqrt(u0.T @ (M @ u0)))
    u0 /= norm_M

    # Evolve under full K and pure G
    traj_K = evolve_linear(K, u0, times)
    traj_G = evolve_linear(G, u0, times)

    # Energies F_K(t), F_G(t)
    F_K = np.array([energy_F(M, u) for u in traj_K])
    F_G = np.array([energy_F(M, u) for u in traj_G])

    # Projections onto slowest two dissipative modes
    # Use M-inner product <u, v>_M = u^T M v
    def m_inner(u, v):
        return float(u.T @ (M @ v))

    proj_K_min = np.array([m_inner(u, v_min) for u in traj_K])
    proj_K_second = np.array([m_inner(u, v_second) for u in traj_K])

    proj_G_min = np.array([m_inner(u, v_min) for u in traj_G])
    proj_G_second = np.array([m_inner(u, v_second) for u in traj_G])

    # Expected asymptotic decay rate for F(t) ~ exp(-r t) is r ~ 2 * lambda_min
    r_expected = 2.0 * lam_min

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 1) F(t) vs t
    ax = axes[0, 0]
    ax.plot(times, F_K, label="F_K(t): full K-flow")
    ax.plot(times, F_G, label="F_G(t): pure G-flow", linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel("F(t)")
    ax.set_title("Energy-like functional F(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) log F(t) vs t with expected slope
    ax = axes[0, 1]
    mask_FK = F_K > 0
    mask_FG = F_G > 0
    ax.plot(times[mask_FK], np.log(F_K[mask_FK]), label="log F_K(t)")
    ax.plot(times[mask_FG], np.log(F_G[mask_FG]), label="log F_G(t)", linestyle="--")

    # Reference line with slope -r_expected passing through the last log F_G point
    t_ref0 = times[-1]
    logF_ref0 = np.log(F_G[mask_FG][-1])
    t_ref_line = np.array([times[0], times[-1]])
    logF_ref_line = logF_ref0 - r_expected * (t_ref_line - t_ref0)
    ax.plot(t_ref_line, logF_ref_line, label=f"slope -2 λ_min ≈ {-r_expected:.3f}", linestyle=":")
    ax.set_xlabel("t")
    ax.set_ylabel("log F(t)")
    ax.set_title("Log-decay and expected asymptotic slope")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Phase portrait in slow dissipative plane for K-flow
    ax = axes[1, 0]
    ax.plot(proj_K_min, proj_K_second, label="K-flow")
    ax.plot(proj_G_min, proj_G_second, label="G-flow", linestyle="--")
    ax.set_xlabel("<u, v_min>_M")
    ax.set_ylabel("<u, v_second>_M")
    ax.set_title("Projection onto two slowest dissipative modes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Projections vs time to see rotation vs pure decay
    ax = axes[1, 1]
    ax.plot(times, proj_K_min, label="<u_K(t), v_min>_M")
    ax.plot(times, proj_K_second, label="<u_K(t), v_second>_M")
    ax.plot(times, proj_G_min, label="<u_G(t), v_min>_M", linestyle="--")
    ax.plot(times, proj_G_second, label="<u_G(t), v_second>_M", linestyle=":")
    ax.set_xlabel("t")
    ax.set_ylabel("projection")
    ax.set_title("Mode amplitudes: rotation (K) vs pure decay (G)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
