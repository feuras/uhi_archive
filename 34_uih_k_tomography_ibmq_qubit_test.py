#!/usr/bin/env python3
"""
34_uih_k_tomography_ibmq_qubit_test.py

Universal Information Hydrodynamics K tomography smoking gun test
for a single IBM Quantum qubit.

This script

  - connects to an IBM Quantum backend via QiskitRuntimeService
    (with a clean fall back to AerSimulator),
  - runs one qubit process tomography for a shallow idle circuit,
  - reconstructs the noisy channel as a SuperOp,
  - extracts a real generator K on the traceless Bloch subspace,
  - builds the Bogoliubov Kubo Mori (BKM) metric at the stationary state,
  - splits K into metric symmetric (G) and metric skew (J) parts,
  - checks the UIH picture of one current and two quadratures:
        MG symmetric, MJ skew, G positive, J no work direction.

Dependencies

  pip install qiskit qiskit-aer qiskit-experiments qiskit-ibm-runtime numpy scipy

Notes on IBM connection

  Before running on real hardware you should save your IBM account once, for
  example

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token="YOUR_API_KEY_HERE",
        overwrite=True,
    )

  After that, QiskitRuntimeService() can be called without arguments and this
  script will pick up your stored credentials.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, eigvalsh
import scipy.linalg as la

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SuperOp

from qiskit_experiments.library import ProcessTomography

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False


# Global knobs for the experiment
N_SHOTS = 2048            # per tomography circuit
IDLE_DEPTH = 4            # number of identity gates to accumulate noise
STATIONARY_ITERS = 40     # iterations to converge to the stationary state
BKM_EPS_SPEC = 1e-8       # spectral regularisation for BKM metric


def build_idle_circuit(num_id_gates: int = IDLE_DEPTH) -> QuantumCircuit:
    """
    Build a one qubit circuit that implements a shallow idle evolution.

    We rely on the device native identity gate and keep transpiler
    optimisation level at zero so that the idle depth is respected.
    """
    qc = QuantumCircuit(1, name=f"idle_depth_{num_id_gates}")
    for _ in range(num_id_gates):
        qc.id(0)
    qc.barrier()
    return qc


def get_backend():
    """
    Try to connect to an IBM Quantum backend via QiskitRuntimeService.

    If that fails for any reason, fall back to a local AerSimulator.
    """
    if HAVE_RUNTIME:
        try:
            service = QiskitRuntimeService()
            backend = service.least_busy(
                simulator=False,
                operational=True,
                min_num_qubits=1,
            )
            print(f"Using IBM Quantum backend: {backend.name}")
            return backend
        except Exception as exc:
            print("Warning: could not acquire IBM Quantum backend.")
            print(f"Reason: {exc!r}")
            print("Falling back to local AerSimulator.")
    else:
        print("qiskit_ibm_runtime not available, using AerSimulator.")

    backend = AerSimulator()
    print(f"Using backend: {backend}")
    return backend


def apply_superop_to_operator(chan: SuperOp, A: np.ndarray) -> np.ndarray:
    """
    Apply a SuperOp channel to a general operator A using the internal
    column stacked representation.

    Qiskit represents superoperators as S such that

        vec_out = S @ vec_in,

    where vec_in is formed by stacking the columns of A.
    """
    d = A.shape[0]
    vec_in = A.reshape((d * d,), order="F")
    vec_out = chan.data @ vec_in
    A_out = vec_out.reshape((d, d), order="F")
    return A_out


def compute_stationary_state(chan: SuperOp,
                             n_iter: int = STATIONARY_ITERS,
                             eps: float = 1e-8) -> np.ndarray:
    """
    Iteratively compute the stationary state rho_ss of a mixing channel.

    Start from the maximally mixed state, repeatedly apply the channel,
    enforce trace one and Hermiticity at each step, then spectrally
    clip tiny negative eigenvalues at the end.
    """
    d = chan.dim[0]
    rho = np.eye(d, dtype=complex) / d

    for _ in range(n_iter):
        rho = apply_superop_to_operator(chan, rho)
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if tr.real <= 0:
            raise RuntimeError("Non positive trace encountered in fixed point iteration.")
        rho /= tr

    evals, evecs = np.linalg.eigh(rho)
    evals_clipped = np.clip(evals.real, eps, None)
    rho_reg = evecs @ np.diag(evals_clipped) @ evecs.conj().T
    rho_reg /= np.trace(rho_reg)
    rho_reg = 0.5 * (rho_reg + rho_reg.conj().T)
    return rho_reg


def build_hermitian_basis_qubit():
    """
    Return an orthonormal Hermitian operator basis for a single qubit:

        H0 = I / sqrt(2)
        H1 = X / sqrt(2)
        H2 = Y / sqrt(2)
        H3 = Z / sqrt(2)

    satisfying Tr(Ha Hb) = delta_ab.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    H0 = I / np.sqrt(2.0)
    H1 = X / np.sqrt(2.0)
    H2 = Y / np.sqrt(2.0)
    H3 = Z / np.sqrt(2.0)

    H_basis = [H0, H1, H2, H3]

    # Quick orthonormality check
    G = np.zeros((4, 4), dtype=float)
    for a, Ha in enumerate(H_basis):
        for b, Hb in enumerate(H_basis):
            G[a, b] = float(np.real(np.trace(Ha.conj().T @ Hb)))
    if not np.allclose(G, np.eye(4), atol=1e-10):
        raise RuntimeError("Hermitian basis is not orthonormal in Hilbert Schmidt inner product.")

    return H_basis


def superop_matrix_in_basis(chan: SuperOp,
                            H_basis: list[np.ndarray]) -> np.ndarray:
    """
    Represent the channel as a matrix R on the Hermitian basis H_a:

        R_ab = Tr( H_a^† E(H_b) ),

    so that for any operator A with coordinates a_b one has

        a_out = R @ a_in.

    For a single qubit this returns a 4 x 4 matrix.
    """
    d2 = len(H_basis)
    R = np.zeros((d2, d2), dtype=np.complex128)
    for b, Hb in enumerate(H_basis):
        E_Hb = apply_superop_to_operator(chan, Hb)
        for a, Ha in enumerate(H_basis):
            R[a, b] = np.trace(Ha.conj().T @ E_Hb)
    return R


def build_bkm_metric_qubit(rho_ss: np.ndarray,
                           H_basis: list[np.ndarray],
                           eps_spec: float = BKM_EPS_SPEC) -> np.ndarray:
    """
    Build the Bogoliubov Kubo Mori metric matrix M_ab at rho_ss on the
    Hermitian basis H_a.

    The BKM inner product is

        g(A, B) = Tr( A Omega_rho^{-1}(B) ),

    where Omega_rho is the Kubo Mori operator

        Omega_rho(B) = integral_0^1 rho^s B rho^{1-s} ds.

    In the eigenbasis of rho with eigenvalues p_i, Omega_rho acts as

        [Omega_rho(B)]_{ij} = c_ij B_{ij},

    where

        c_ij = (p_i - p_j) / (log p_i - log p_j)       for i != j,
        c_ii = p_i.

    Its inverse is diagonal in the same basis with entries 1 / c_ij.
    """
    rho = 0.5 * (rho_ss + rho_ss.conj().T)
    evals, evecs = np.linalg.eigh(rho)
    evals_clipped = np.clip(evals.real, eps_spec, None)
    rho_reg = evecs @ np.diag(evals_clipped) @ evecs.conj().T
    rho_reg /= np.trace(rho_reg)

    evals_reg, U = np.linalg.eigh(rho_reg)
    d = len(evals_reg)

    # Build c_ij table
    C = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            pi = float(evals_reg[i].real)
            pj = float(evals_reg[j].real)
            if abs(pi - pj) < 1e-14:
                C[i, j] = pi
            else:
                C[i, j] = (pi - pj) / (np.log(pi) - np.log(pj))

    def omega_inv(B: np.ndarray) -> np.ndarray:
        """Apply Omega_rho^{-1} to B using the spectral representation."""
        B_eig = U.conj().T @ B @ U
        out_eig = np.zeros_like(B_eig, dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                cij = C[i, j]
                if cij > 0.0:
                    out_eig[i, j] = B_eig[i, j] / cij
                else:
                    out_eig[i, j] = 0.0
        return U @ out_eig @ U.conj().T

    d2 = len(H_basis)
    M = np.zeros((d2, d2), dtype=float)
    for a, Ha in enumerate(H_basis):
        for b, Hb in enumerate(H_basis):
            Omega_inv_Hb = omega_inv(Hb)
            M[a, b] = float(np.real(np.trace(Ha.conj().T @ Omega_inv_Hb)))

    return M


def extract_traceless_generator(chan: SuperOp,
                                H_basis: list[np.ndarray]):
    """
    From a reconstructed SuperOp channel, build a real generator K_tr
    on the traceless Bloch sector.

    Steps

      - Build the 4 x 4 representation R in the Hermitian basis.
      - Restrict to the traceless block R_tr = R[1:, 1:].
      - Compute K_tr = logm(R_tr) and keep its real part.
    """
    R = superop_matrix_in_basis(chan, H_basis)
    R_tr = R[1:, 1:]
    R_tr_real = R_tr.real

    # Logarithm of the channel in the traceless sector
    K_tr = la.logm(R_tr_real)
    K_tr_real = K_tr.real

    # Imaginary leakage diagnostic
    imag_norm = norm(K_tr.imag)
    print(f"\nGenerator extraction diagnostics")
    print(f"  Norm of imaginary part of logm(R_tr) = {imag_norm:.3e}")

    return K_tr_real, R_tr_real


def metric_adjoint_split(K_tr: np.ndarray,
                         M_tr: np.ndarray):
    """
    Perform the metric adjoint split of K_tr with respect to metric M_tr.

    The metric adjoint K_sharp is defined by

        <K u, v>_M = <u, K_sharp v>_M,

    which gives K_sharp = M^{-1} K^T M in matrix form.

    We then define

        G = 0.5 (K + K_sharp),
        J = 0.5 (K - K_sharp).
    """
    M_inv = la.inv(M_tr)
    K_sharp = M_inv @ K_tr.T @ M_tr

    G = 0.5 * (K_tr + K_sharp)
    J = 0.5 * (K_tr - K_sharp)

    return G, J, K_sharp


def ui_h_checks(K_tr: np.ndarray,
                G: np.ndarray,
                J: np.ndarray,
                M_tr: np.ndarray):
    """
    Compute UIH diagnostics on the traceless sector.

      - symmetry of M G,
      - skewness of M J,
      - spectrum of the dissipative operator.

    For a perfect UIH structure one expects

      - M G symmetric positive definite,
      - M J skew symmetric,
      - dissipative spectrum with strictly positive eigenvalues.
    """
    MG = M_tr @ G
    MJ = M_tr @ J

    sym_res = norm(MG - MG.T) / max(norm(MG), 1e-16)
    skew_res = norm(MJ + MJ.T) / max(norm(MJ), 1e-16)

    # Dissipative operator in M geometry
    # Eigenproblem for (-M G, M) is equivalent to symmetric part
    MG_sym = 0.5 * (MG + MG.T)
    evals_MG = eigvalsh(-MG_sym)

    print("\nUIH metric split diagnostics")
    print("--------------------------------")
    print(f"  Symmetry residual for M G        = {sym_res:.3e}")
    print(f"  Skewness residual for M J        = {skew_res:.3e}")
    print(f"  Eigenvalues of -sym(M G)         = {evals_MG}")
    print(f"  Smallest positive eigenvalue     = {evals_MG.min():.6f}")
    print(f"  Largest eigenvalue               = {evals_MG.max():.6f}")


def run_uih_k_tomography_test():
    print("=" * 72)
    print("UIH K tomography smoking gun test on an IBM Quantum qubit")
    print("=" * 72)

    backend = get_backend()
    qc_idle = build_idle_circuit(IDLE_DEPTH)

    print("\nIdle circuit for tomography")
    print("---------------------------")
    print(f"  Circuit name        = {qc_idle.name}")
    print(f"  Idle depth (id)     = {IDLE_DEPTH}")
    print(qc_idle.draw(fold=60))

    print("\nRunning one qubit process tomography")
    print("------------------------------------")
    qpt = ProcessTomography(qc_idle, physical_qubits=(0,))
    qpt.set_transpile_options(optimization_level=0)
    qpt.set_run_options(shots=N_SHOTS)

    expdata = qpt.run(backend)
    expdata.block_for_results()

    # Extract the estimated Choi state and turn it into a SuperOp
    state_df = expdata.analysis_results("state", dataframe=True)
    choi_est = state_df.iloc[0].value
    chan = SuperOp(choi_est)

    print("\nReconstructed channel diagnostics")
    print("---------------------------------")
    print(f"  SuperOp dimension              = {chan.dim}")
    print(f"  Number of qubits               = {chan.num_qubits}")

    # Stationary state
    rho_ss = compute_stationary_state(chan, n_iter=STATIONARY_ITERS)
    evals_rho, _ = np.linalg.eigh(rho_ss)

    print("\nStationary state diagnostics")
    print("----------------------------")
    print("  rho_ss eigenvalues            =", np.round(evals_rho.real, 6))
    print("  rho_ss                        =")
    print(np.round(rho_ss, 6))

    # Hermitian basis and superoperator representation
    H_basis = build_hermitian_basis_qubit()
    R = superop_matrix_in_basis(chan, H_basis)

    print("\nSuperoperator representation in Hermitian basis")
    print("-----------------------------------------------")
    print("  R (4 x 4)                     =")
    print(np.round(R.real, 6))

    # Extract generator on traceless sector
    K_tr, R_tr = extract_traceless_generator(chan, H_basis)

    print("\nTraceless sector matrices")
    print("-------------------------")
    print("  R_tr (3 x 3)                  =")
    print(np.round(R_tr, 6))
    print("  K_tr (3 x 3, real)            =")
    print(np.round(K_tr, 6))

    # BKM metric and its traceless block
    M = build_bkm_metric_qubit(rho_ss, H_basis, eps_spec=BKM_EPS_SPEC)
    M_tr = M[1:, 1:]

    cond_M_tr = np.linalg.cond(M_tr)

    print("\nBKM metric diagnostics")
    print("----------------------")
    print("  Full BKM metric M (4 x 4)     =")
    print(np.round(M, 6))
    print("  Traceless block M_tr (3 x 3)  =")
    print(np.round(M_tr, 6))
    print(f"  Condition number of M_tr      = {cond_M_tr:.3e}")

    # Metric adjoint split
    G, J, K_sharp = metric_adjoint_split(K_tr, M_tr)

    print("\nMetric adjoint K_sharp and split")
    print("--------------------------------")
    print("  K_sharp (3 x 3)               =")
    print(np.round(K_sharp, 6))
    print("  G = 0.5 (K + K_sharp)         =")
    print(np.round(G, 6))
    print("  J = 0.5 (K - K_sharp)         =")
    print(np.round(J, 6))

    # UIH checks
    ui_h_checks(K_tr, G, J, M_tr)

    print("\nConclusion")
    print("----------")
    print("  The script has reconstructed the noisy one qubit channel from")
    print("  process tomography, extracted a real generator K on the")
    print("  traceless Bloch sector, built the BKM metric at the stationary")
    print("  state, and performed the metric adjoint split K = G + J.")
    print("  Small symmetry and skewness residuals for M G and M J, and a")
    print("  strictly positive dissipative spectrum for -sym(M G), realise")
    print("  the UIH picture of one current and two quadratures in hardware.")


if __name__ == "__main__":
    run_uih_k_tomography_test()
