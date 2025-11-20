#!/usr/bin/env python3
"""
44_qapi_two_qubit_k_cptp_semigroup_decay_checks.py

Semigroup decay checks for a CPTP repaired two qubit K generator, using
BKM relative entropy and Fisher metric data produced by
43_qapi_two_qubit_k_cptp_repair_and_bkm_checks.py.

This script:

  * loads a *_cptp.npz file containing:

        - T_cp        : 16x16 Pauli transfer matrix (including identity),
        - K_cp        : 15x15 matrix log of T_cp,tr,
        - K_reg_cp    : 15x15 spectrally regularised generator,
        - v_stat_cp   : 16 component stationary Pauli vector,
        - rho_ss_cp   : 4x4 stationary density matrix,
        - M_bkm       : 15x15 BKM metric on traceless Pauli sector,
        - G_bkm       : 15x15 Fisher symmetric generator,
        - G_bkm_evals : eigenvalues of G_bkm in BKM orthonormal
                        coordinates (sorted),

  * locates the corresponding tomography npz by stripping the "_cptp"
    suffix, and loads V_in, the input Pauli expectation vectors,

  * constructs 8 physical initial density matrices from the first 8 rows
    of V_in, using

        rho = (1/4) sum_k v_k P_k

    in the two qubit Pauli basis,

  * optionally adds one additional "gap mode" initial state built by
    taking the eigenvector of G_bkm with eigenvalue closest to zero
    from below (in the Euclidean sense) and mapping it to a traceless
    Pauli perturbation of rho_ss_cp,

  * propagates the traceless Pauli components under the semigroup
    exp(t K_reg_cp) over a user controlled time grid, and reconstructs
    rho_t from the Pauli expectations,

  * computes the BKM relative entropy D_BKM(t) = D( rho_t || rho_ss_cp )
    along the orbit using the full quantum relative entropy,

  * fits log D_BKM(t) in a short time window [t_s_min, t_s_max] and a
    long time window [t_l_min, t_l_max] to extract two effective decay
    rates, and compares both to the Fisher gap (largest negative
    eigenvalue of G_bkm),

  * reports, for each initial state, D_BKM(0), D_BKM(t_max), the short
    and long fitted slopes, the Fisher gap, the ratios slope/gap, and
    the range of eigenvalues and traces of rho_t over the time grid,

  * prints a summary over all initial states, including mean slopes and
    the Fisher gap.

Usage
-----

    python 44_qapi_two_qubit_k_cptp_semigroup_decay_checks.py \
        ibmq_results/ibmq_two_qubit_k_tomography_ibm_fez_20251120_094042_8kshots_cptp.npz

If no argument is given, a default CPTP npz path is used.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------

@dataclass
class DecayCheckConfig:
    cptp_npz_path: str = (
        "ibmq_results/"
        "ibmq_two_qubit_k_tomography_ibm_fez_20251120_090502_6kshots_cptp.npz"
    )
    # Time grid parameters
    t_max: float = 60.0
    n_t: int = 241  # dt = 0.25

    # Fit windows for log D(t)
    fit_window_short: Tuple[float, float] = (0.5, 5.0)
    fit_window_long: Tuple[float, float] = (20.0, 60.0)

    # Whether to add a gap mode initial state aligned with a slow
    # eigenvector of G_bkm
    use_gap_mode_init: bool = True

    # Numerical floors and tolerances
    rho_eig_floor: float = 1e-6
    D_floor: float = 1e-10
    equal_tol: float = 1e-10

    verbose: bool = True


# ----------------------------------------------------------------------
# Two qubit Pauli basis and utilities
# ----------------------------------------------------------------------

def single_qubit_paulis() -> List[np.ndarray]:
    """Return [I, X, Y, Z] as 2x2 complex arrays."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]


def two_qubit_paulis() -> List[np.ndarray]:
    """
    Build the 16 two qubit Pauli matrices in row major
    {I, X, Y, Z} ⊗ {I, X, Y, Z} order:

        idx = 4 * i + j  <->  P_idx = P_i ⊗ P_j.
    """
    paulis_1q = single_qubit_paulis()
    ops: List[np.ndarray] = []
    for i in range(4):
        for j in range(4):
            ops.append(np.kron(paulis_1q[i], paulis_1q[j]))
    return ops


PAULIS_2Q = two_qubit_paulis()
D2Q = 4  # two qubits


def reconstruct_rho_from_vstat(
    v_stat: np.ndarray,
    pauli_ops: List[np.ndarray],
) -> np.ndarray:
    """
    Given a 16 component Pauli expectation vector v_stat with entries

        v_stat[k] = Tr(rho P_k),

    and a list of 16 two qubit Pauli operators P_k ordered as in the
    tomography scripts, reconstruct the density matrix

        rho = 1/4 * sum_k v_stat[k] P_k.
    """
    if v_stat.shape[0] != 16:
        raise ValueError(f"Expected v_stat of length 16, got {v_stat.shape[0]}")
    if len(pauli_ops) != 16:
        raise ValueError(f"Expected 16 Pauli operators, got {len(pauli_ops)}")

    rho = np.zeros((D2Q, D2Q), dtype=complex)
    for k in range(16):
        rho += v_stat[k] * pauli_ops[k]
    rho *= 0.25
    return rho


# ----------------------------------------------------------------------
# Quantum relative entropy (BKM functional)
# ----------------------------------------------------------------------

def quantum_relative_entropy(
    rho: np.ndarray,
    sigma: np.ndarray,
    eig_floor: float = 1e-8,
) -> float:
    """
    Quantum relative entropy

        D(rho || sigma) = Tr[rho (log rho - log sigma)],

    with eigenvalues of rho and sigma clipped below eig_floor to avoid
    log(0). Both rho and sigma are assumed Hermitian and positive.
    """
    # Diagonalise rho
    evals_rho, _ = np.linalg.eigh(rho)
    lam_rho = np.clip(evals_rho.real, eig_floor, None)
    term1 = float(np.sum(lam_rho * np.log(lam_rho)))

    # Diagonalise sigma
    evals_sigma, U_sigma = np.linalg.eigh(sigma)
    lam_sigma = np.clip(evals_sigma.real, eig_floor, None)
    log_sigma = U_sigma @ np.diag(np.log(lam_sigma)) @ U_sigma.conj().T

    term2 = float(np.real(np.trace(rho @ log_sigma)))
    return term1 - term2


# ----------------------------------------------------------------------
# Semigroup propagation exp(t K_reg_cp) on traceless Pauli sector
# ----------------------------------------------------------------------

def build_K_eigendecomposition(K_reg_cp: np.ndarray):
    """
    Diagonalise K_reg_cp (15x15). Returns eigenvalues and eigenvectors
    along with the inverse of the eigenvector matrix.
    """
    vals, vecs = np.linalg.eig(K_reg_cp.astype(complex))
    vecs_inv = np.linalg.inv(vecs)
    return vals, vecs, vecs_inv


def propagate_traceless_pauli(
    K_evals: np.ndarray,
    K_vecs: np.ndarray,
    K_vecs_inv: np.ndarray,
    w0: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Propagate traceless Pauli components w(t) under

        w(t) = exp(t K_reg_cp) w0

    using the spectral representation of K_reg_cp.

    Returns an array of shape (n_t, 15).
    """
    coeff = K_vecs_inv @ w0.astype(complex)
    n_t = t_grid.shape[0]
    w_t = np.zeros((n_t, w0.shape[0]), dtype=complex)
    for i, t in enumerate(t_grid):
        factors = np.exp(K_evals * t)
        w_t[i, :] = K_vecs @ (factors * coeff)
    return w_t.real


# ----------------------------------------------------------------------
# Fitting utilities
# ----------------------------------------------------------------------

def fit_log_decay_slope(
    t_grid: np.ndarray,
    D_vals: np.ndarray,
    window: Tuple[float, float],
    D_floor: float = 1e-10,
) -> float:
    """
    Fit log D_vals versus t over a time window [t_min, t_max], ignoring
    points where D_vals <= D_floor. Returns the slope of the best fit,
    or NaN if fewer than 2 points are available.
    """
    t_min, t_max = window
    mask = (t_grid >= t_min) & (t_grid <= t_max) & (D_vals > D_floor)
    if np.sum(mask) < 2:
        return float("nan")
    t_fit = t_grid[mask]
    logD = np.log(D_vals[mask])
    slope, _ = np.polyfit(t_fit, logD, 1)
    return float(slope)


# ----------------------------------------------------------------------
# Gap mode initial state construction
# ----------------------------------------------------------------------

def build_gap_mode_initial_state(
    rho_ss_cp: np.ndarray,
    G_bkm: np.ndarray,
    pauli_ops: List[np.ndarray],
    eig_floor: float = 1e-6,
    max_tries: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a gap mode initial density matrix and its Pauli
    expectation vector by:

      1) diagonalising the symmetric part of G_bkm in the Euclidean
         inner product,

      2) selecting the eigenvector corresponding to the negative
         eigenvalue closest to zero,

      3) mapping that 15 component eigenvector to a traceless Pauli
         perturbation

            delta_rho = sum_{i=1}^{15} u_i (P_i / 4),

      4) choosing an amplitude epsilon so that

            rho_init = rho_ss_cp + epsilon delta_rho

         remains positive definite with eigenvalues above eig_floor.

    Returns (rho_init_gap, v_init_gap) where v_init_gap is the 16
    component Pauli expectation vector of rho_init_gap.
    """
    # Symmetrise G_bkm
    G_sym = 0.5 * (G_bkm + G_bkm.T)
    evals, vecs = np.linalg.eigh(G_sym)

    neg_mask = evals < 0.0
    if not np.any(neg_mask):
        raise RuntimeError("G_bkm has no negative eigenvalues, cannot build gap mode.")

    neg_evals = evals[neg_mask]
    # Slowest mode: negative eigenvalue with smallest magnitude
    idx_local = int(np.argmin(np.abs(neg_evals)))
    lam_gap_mode = float(neg_evals[idx_local])

    # Recover global index
    indices = np.where(neg_mask)[0]
    idx_global = int(indices[idx_local])

    u_gap = vecs[:, idx_global].real  # 15 components

    # Build traceless perturbation delta_rho
    delta_rho = np.zeros_like(rho_ss_cp, dtype=complex)
    for i in range(1, 16):
        delta_rho += u_gap[i - 1] * (pauli_ops[i] / 4.0)

    # Normalise delta_rho and choose epsilon
    frob_norm = np.linalg.norm(delta_rho, ord="fro")
    if frob_norm == 0.0:
        raise RuntimeError("delta_rho has zero norm in gap mode construction.")
    # Aim for small perturbation in HS norm
    epsilon = 0.1 / frob_norm

    rho_init = rho_ss_cp.copy().astype(complex)
    for _ in range(max_tries):
        trial = rho_ss_cp + epsilon * delta_rho
        evals_trial = np.linalg.eigvalsh(trial)
        if np.min(evals_trial.real) > eig_floor:
            rho_init = trial
            break
        epsilon *= 0.5
    else:
        # Final attempt with very small epsilon, even if eigenvalues are
        # slightly below floor, they will be clipped for logs
        rho_init = rho_ss_cp + epsilon * delta_rho

    # Build Pauli expectation vector v_init_gap
    v_init = np.zeros(16, dtype=float)
    for k in range(16):
        v_init[k] = float(np.trace(rho_init @ pauli_ops[k]).real)

    return rho_init, v_init


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def run_decay_checks(cfg: DecayCheckConfig) -> None:
    t0 = time.time()
    print(f"[UIH-DECAY] Loading CPTP repaired data from: {cfg.cptp_npz_path}")
    if not os.path.exists(cfg.cptp_npz_path):
        raise FileNotFoundError(f"Cannot find CPTP npz file at '{cfg.cptp_npz_path}'")

    data_cptp = np.load(cfg.cptp_npz_path)

    required_keys = [
        "T_cp",
        "K_cp",
        "K_reg_cp",
        "v_stat_cp",
        "rho_ss_cp",
        "M_bkm",
        "G_bkm",
        "G_bkm_evals",
    ]
    for key in required_keys:
        if key not in data_cptp:
            raise KeyError(f"CPTP npz is missing required array '{key}'")

    T_cp = data_cptp["T_cp"]
    K_cp = data_cptp["K_cp"]
    K_reg_cp = data_cptp["K_reg_cp"]
    v_stat_cp = data_cptp["v_stat_cp"]
    rho_ss_cp = data_cptp["rho_ss_cp"]
    M_bkm = data_cptp["M_bkm"]
    G_bkm = data_cptp["G_bkm"]
    G_bkm_evals = data_cptp["G_bkm_evals"]

    print(f"[UIH-DECAY] Shapes: T_cp = {T_cp.shape}, K_cp = {K_cp.shape}, "
          f"K_reg_cp = {K_reg_cp.shape}")

    # Fisher spectrum diagnostics
    evals_F = np.array(G_bkm_evals, dtype=float)
    print("[UIH-DECAY] Fisher G_bkm eigenvalues (sorted):")
    for lam in evals_F:
        print(f"    {lam:.6f}")

    negs = evals_F[evals_F < 0.0]
    if negs.size > 0:
        fisher_gap = float(negs.max())   # largest negative (slowest decay)
        fisher_fast = float(negs.min())  # most negative (fastest decay)
    else:
        fisher_gap = 0.0
        fisher_fast = 0.0

    print(f"[UIH-DECAY] Fisher slowest dissipative rate (largest negative): {fisher_gap:.6f}")
    print(f"[UIH-DECAY] Fisher fastest dissipative rate (most negative)  : {fisher_fast:.6f}")

    # Consistency check exp(K_reg_cp) vs T_cp,tr at t = 1
    idx_non_id = list(range(1, 16))
    T_tr_cp = T_cp[np.ix_(idx_non_id, idx_non_id)]
    K_evals, K_vecs, K_vecs_inv = build_K_eigendecomposition(K_reg_cp)
    factors_1 = np.exp(K_evals * 1.0)
    expK1 = (K_vecs @ (np.diag(factors_1) @ K_vecs_inv)).real
    diff_expk1 = np.linalg.norm(expK1 - T_tr_cp)
    print(f"[UIH-DECAY] Consistency check ||exp(K_reg_cp) - T_cp,tr||_F at t=1: {diff_expk1:.3e}")

    # Stationary vector and rho_ss diagnostics
    v_stat_cp = v_stat_cp.real
    evals_rho_ss, _ = np.linalg.eigh(rho_ss_cp)
    evals_rho_ss_clipped = np.clip(evals_rho_ss.real, cfg.rho_eig_floor, None)

    print(f"[UIH-DECAY] Stationary eigenvalue of T_cp closest to 1: "
          f"{1.0 + 0j}")  # K_reg_cp already regularised
    print("[UIH-DECAY] Stationary rho_ss_cp eigenvalues (clipped for logs):")
    for lam in evals_rho_ss_clipped:
        print(f"    {lam:.6e}")

    # Time grid
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t)
    print(f"[UIH-DECAY] Time grid: {t_grid}")

    # Load original tomography npz to get V_in for initial states
    if cfg.cptp_npz_path.endswith("_cptp.npz"):
        tomo_npz_path = cfg.cptp_npz_path.replace("_cptp.npz", ".npz")
    else:
        # Fallback: try stripping trailing ".npz" and appending ".npz"
        base, _ = os.path.splitext(cfg.cptp_npz_path)
        tomo_npz_path = base + ".npz"

    if not os.path.exists(tomo_npz_path):
        raise FileNotFoundError(
            f"Cannot find tomography npz file at '{tomo_npz_path}' "
            "derived from CPTP path."
        )

    data_tomo = np.load(tomo_npz_path)
    if "V_in" not in data_tomo:
        raise KeyError("Tomography npz is missing 'V_in' array for initial states.")

    V_in = data_tomo["V_in"]
    if V_in.ndim != 2 or V_in.shape[1] != 16:
        raise ValueError(f"Expected V_in with shape (N, 16), got {V_in.shape}")

    n_phys_inits = min(8, V_in.shape[0])
    pauli_ops = PAULIS_2Q

    # Build list of initial Pauli vectors and density matrices
    v_init_list: List[np.ndarray] = []
    rho_init_list: List[np.ndarray] = []

    for i in range(n_phys_inits):
        v_init = V_in[i, :].real
        rho_init = reconstruct_rho_from_vstat(v_init, pauli_ops)
        v_init_list.append(v_init)
        rho_init_list.append(rho_init)

    # Optional gap mode initial state
    gap_mode_added = False
    if cfg.use_gap_mode_init:
        try:
            rho_gap, v_gap = build_gap_mode_initial_state(
                rho_ss_cp=rho_ss_cp,
                G_bkm=G_bkm,
                pauli_ops=pauli_ops,
                eig_floor=cfg.rho_eig_floor,
            )
            v_init_list.append(v_gap)
            rho_init_list.append(rho_gap)
            gap_mode_added = True
        except Exception as e:
            print(f"[UIH-DECAY] Warning: could not construct gap mode initial state: {e}")

    n_inits = len(v_init_list)

    # Per state diagnostics
    slopes_short: List[float] = []
    slopes_long: List[float] = []

    for idx in range(n_inits):
        v_init = v_init_list[idx]
        rho_init = rho_init_list[idx]

        # Deviation from stationary Pauli vector in traceless sector
        w0 = v_init[1:] - v_stat_cp[1:]

        # Propagate w(t)
        w_t = propagate_traceless_pauli(
            K_evals=K_evals,
            K_vecs=K_vecs,
            K_vecs_inv=K_vecs_inv,
            w0=w0,
            t_grid=t_grid,
        )

        # Reconstruct v_t and rho_t, compute D_BKM(t)
        D_vals = np.zeros_like(t_grid)
        min_eigs = []
        traces = []

        for j, t in enumerate(t_grid):
            v_t = np.zeros(16, dtype=float)
            v_t[0] = 1.0
            v_t[1:] = v_stat_cp[1:] + w_t[j, :]

            rho_t = reconstruct_rho_from_vstat(v_t, pauli_ops)
            # Hermitise softly
            rho_t = 0.5 * (rho_t + rho_t.conj().T)

            evals_t = np.linalg.eigvalsh(rho_t)
            min_eigs.append(float(evals_t.real.min()))
            traces.append(float(np.trace(rho_t).real))

            D_vals[j] = quantum_relative_entropy(
                rho=rho_t,
                sigma=rho_ss_cp,
                eig_floor=cfg.rho_eig_floor,
            )

        D0 = float(D_vals[0])
        D_end = float(D_vals[-1])
        min_eig_range = (float(np.min(min_eigs)), float(np.max(min_eigs)))
        tr_range = (float(np.min(traces)), float(np.max(traces)))

        slope_short = fit_log_decay_slope(
            t_grid=t_grid,
            D_vals=D_vals,
            window=cfg.fit_window_short,
            D_floor=cfg.D_floor,
        )
        slope_long = fit_log_decay_slope(
            t_grid=t_grid,
            D_vals=D_vals,
            window=cfg.fit_window_long,
            D_floor=cfg.D_floor,
        )

        slopes_short.append(slope_short)
        slopes_long.append(slope_long)

        label = f"Initial state {idx}"
        if gap_mode_added and idx == n_inits - 1:
            label += " (gap mode)"

        print(f"[UIH-DECAY] {label}:")
        print(f"    D_BKM(0)             = {D0:.6e}")
        print(f"    D_BKM(t_max)         = {D_end:.6e}")
        print(f"    Fitted short slope   = {slope_short:.6e} "
              f"(window {cfg.fit_window_short[0]:.2f} to {cfg.fit_window_short[1]:.2f})")
        print(f"    Fitted long slope    = {slope_long:.6e} "
              f"(window {cfg.fit_window_long[0]:.2f} to {cfg.fit_window_long[1]:.2f})")
        print(f"    Fisher gap           = {fisher_gap:.6e}")
        if fisher_gap != 0.0:
            print(f"    Short slope / gap    = {slope_short / fisher_gap:.3f}")
            print(f"    Long slope / gap     = {slope_long / fisher_gap:.3f}")
        print(f"    min eig(rho_t)       in [{min_eig_range[0]:.3e}, {min_eig_range[1]:.3e}]")
        print(f"    Tr(rho_t)            in [{tr_range[0]:.6f}, {tr_range[1]:.6f}]")

    # Summary over all initial states
    slopes_short_arr = np.array(slopes_short, dtype=float)
    slopes_long_arr = np.array(slopes_long, dtype=float)

    mean_short = float(np.nanmean(slopes_short_arr))
    mean_long = float(np.nanmean(slopes_long_arr))
    min_short = float(np.nanmin(slopes_short_arr))
    max_short = float(np.nanmax(slopes_short_arr))
    min_long = float(np.nanmin(slopes_long_arr))
    max_long = float(np.nanmax(slopes_long_arr))

    print("[UIH-DECAY] Summary of fitted slopes over all initial states:")
    print(f"    mean short slope = {mean_short:.6e}")
    print(f"    min  short slope = {min_short:.6e}")
    print(f"    max  short slope = {max_short:.6e}")
    print(f"    mean long slope  = {mean_long:.6e}")
    print(f"    min  long slope  = {min_long:.6e}")
    print(f"    max  long slope  = {max_long:.6e}")
    print(f"    Fisher gap       = {fisher_gap:.6e}")

    t1 = time.time()
    print(f"[UIH-DECAY] Semigroup decay analysis complete in {t1 - t0:.1f} seconds")


def main():
    if len(sys.argv) > 1:
        cptp_npz_path = sys.argv[1]
    else:
        cptp_npz_path = DecayCheckConfig().cptp_npz_path

    cfg = DecayCheckConfig(cptp_npz_path=cptp_npz_path)
    run_decay_checks(cfg)


if __name__ == "__main__":
    main()
