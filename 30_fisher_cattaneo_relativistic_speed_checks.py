#!/usr/bin/env python3
"""
30_fisher_cattaneo_relativistic_speed_checks.py

Test of the Fisher–Cattaneo idea that a relaxation time τ = ħ/(m c^2)
combined with a diffusion scale D = ħ/m produces a telegraph speed

    v_* = sqrt(D / τ) = c.

We evolve the 1D first order system

    ∂_t ρ = - ∂_x j
    ∂_t j = - (1/τ) j - (D/τ) ∂_x ρ

on a periodic interval using spectral derivatives and RK4, starting from
a localised bump in ρ and j = 0. At a sequence of times we estimate a
"front" position as the furthest point where |ρ| exceeds a small fraction
of its instantaneous maximum, and fit a line front(t) ≈ v_meas t.
"""

import numpy as np


def run_fisher_cattaneo_test(
    m=1.0,
    hbar=1.0,
    c=2.5,
    L=10.0,
    N=1024,
    t_final=1.5,
    dt=1.5e-3,
    front_frac=0.01,
    fit_t_min_frac=0.3,
):
    """
    Run a 1D Fisher–Cattaneo simulation and estimate front speed.

    Parameters
    ----------
    m, hbar, c : floats
        Mass, Planck constant and speed of light. We use these to set
        D = ħ/m and τ = ħ/(m c^2).
    L : float
        Half length of the spatial domain. Domain is [-L, L).
    N : int
        Number of spatial grid points.
    t_final : float
        Final simulation time.
    dt : float
        Time step for RK4.
    front_frac : float
        Relative amplitude threshold to define the "front", as a fraction
        of the instantaneous max(|ρ|).
    fit_t_min_frac : float
        We only fit the front speed on times t >= fit_t_min_frac * t_final.

    Returns
    -------
    v_expected : float
        Theoretical telegraph speed sqrt(D / τ).
    v_measured : float
        Fitted front speed from the simulation.
    """
    # Physical coefficients
    D = hbar / m
    tau = hbar / (m * c * c)
    v_expected = np.sqrt(D / tau)

    # Grid and spectral derivative
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ik = 1j * k

    def d_dx(f):
        return np.fft.ifft(ik * np.fft.fft(f)).real

    # Initial data: localised bump, zero current
    rho = np.exp(- (x ** 2) / (0.2 ** 2))
    j = np.zeros_like(rho)

    n_steps = int(round(t_final / dt))
    dt = t_final / n_steps  # snap dt to fit exactly

    print("Fisher–Cattaneo test")
    print("---------------------")
    print(f"m                     = {m:.6g}")
    print(f"hbar                  = {hbar:.6g}")
    print(f"c                     = {c:.6g}")
    print(f"D = hbar/m            = {D:.6g}")
    print(f"tau = hbar/(m c^2)    = {tau:.6g}")
    print(f"Predicted speed v_*   = sqrt(D/tau) = {v_expected:.6g}")
    print()
    print(f"Domain length L       = {L:.6g}")
    print(f"Grid points N         = {N}")
    print(f"dx                    = {dx:.6g}")
    print(f"t_final               = {t_final:.6g}")
    print(f"dt                    = {dt:.6g}")
    print(f"CFL = v_* dt/dx       = {v_expected * dt / dx:.6g}")
    print()

    def rhs(rho, j):
        drho = - d_dx(j)
        dj = - (1.0 / tau) * j - (D / tau) * d_dx(rho)
        return drho, dj

    t = 0.0
    times = []
    fronts = []

    # Time stepping
    for step in range(1, n_steps + 1):
        k1_r, k1_j = rhs(rho, j)
        k2_r, k2_j = rhs(rho + 0.5 * dt * k1_r, j + 0.5 * dt * k1_j)
        k3_r, k3_j = rhs(rho + 0.5 * dt * k2_r, j + 0.5 * dt * k2_j)
        k4_r, k4_j = rhs(rho + dt * k3_r, j + dt * k3_j)

        rho = rho + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        j = j + (dt / 6.0) * (k1_j + 2 * k2_j + 2 * k3_j + k4_j)
        t += dt

        # Record every few steps
        if step % 10 == 0:
            max_amp = np.max(np.abs(rho))
            if max_amp <= 0:
                front_pos = 0.0
            else:
                thresh = front_frac * max_amp
                idx = np.where(np.abs(rho) > thresh)[0]
                if idx.size == 0:
                    front_pos = 0.0
                else:
                    front_pos = np.max(np.abs(x[idx]))
            times.append(t)
            fronts.append(front_pos)

    times = np.array(times)
    fronts = np.array(fronts)

    # Fit front position vs time at intermediate to late times
    mask = times >= fit_t_min_frac * t_final
    if np.count_nonzero(mask) < 3:
        raise RuntimeError("Not enough points for speed fit")

    coeffs = np.polyfit(times[mask], fronts[mask], 1)
    v_measured = coeffs[0]

    print("Front tracking summary")
    print("----------------------")
    print(f"front_frac (relative)          = {front_frac:.3g}")
    print(f"fit_t_min_frac                 = {fit_t_min_frac:.3g}")
    print(f"Number of samples in fit       = {np.count_nonzero(mask)}")
    print(f"Measured front speed v_meas    = {v_measured:.6g}")
    print(f"Relative error (v_meas - c)/c  = {(v_measured - c)/c:.3g}")
    print()

    return v_expected, v_measured


if __name__ == "__main__":
    run_fisher_cattaneo_test()
