#!/usr/bin/env python3
"""
40_qapi_two_qubit_k_tomography_iqm_resonance.py

Two-qubit K-tomography experiment on an IQM backend using Qiskit on IQM.

This script:
  * builds a complete two-qubit quantum process tomography experiment
    for the "identity" channel (so all noise is hardware + compiler),
  * runs all circuits on an IQM backend via Qiskit-on-IQM,
  * reconstructs the 16x16 Pauli transfer matrix T in the {I,X,Y,Z}⊗2 basis,
  * restricts to the 15x15 traceless subspace (non-identity Paulis),
  * computes an effective generator K via matrix logarithm of the
    traceless block, and
  * saves diagnostics to .npz and .json for later UIH analysis.

Dependencies:
  - iqm-client with Qiskit extra:
        pip install "iqm-client[qiskit]"
  - qiskit >= 2.1
  - numpy

Authentication:
  Set the IQM_TOKEN environment variable to your IQM API token before running
  (or use any other supported authentication method for IQMProvider).

Backend selection:
  The IQM server URL is taken from the environment variable

      IQM_SERVER_URL

  with a fallback demo URL:
      https://cocos.resonance.meetiqm.com/garnet:mock

  The backend name is taken from the environment variable

      QAPI_IQM_BACKEND

  with fallback "facade_garnet" (a facade/noisy-sim backend on the mock URL).

  Example:
      set IQM_TOKEN=...
      set IQM_SERVER_URL=https://cocos.resonance.meetiqm.com/garnet
      set QAPI_IQM_BACKEND=garnet

  For quick local-style tests against the mock environment, you can omit
  IQM_SERVER_URL and QAPI_IQM_BACKEND and use the defaults above.
"""

import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider


# ----------------------------------------------------------------------
# Single-qubit state preparation and measurement bases
# ----------------------------------------------------------------------

PREP_LABELS_1Q = ["Zp", "Zm", "Xp", "Yp"]
MEAS_BASES_1Q = ["X", "Y", "Z"]
PAULIS_1Q = ["I", "X", "Y", "Z"]


def prepare_1q_state(circ: QuantumCircuit, qubit: int, label: str) -> None:
    """
    Prepare a standard single-qubit tomography state on `qubit`:

      Zp: |0>
      Zm: |1>
      Xp: |+>   = (|0> + |1>)/sqrt(2)
      Yp: |+i>  = (|0> + i|1>)/sqrt(2)
    """
    if label == "Zp":
        # |0>, nothing
        return
    if label == "Zm":
        circ.x(qubit)
        return
    if label == "Xp":
        circ.h(qubit)
        return
    if label == "Yp":
        circ.h(qubit)
        circ.s(qubit)
        return
    raise ValueError(f"Unknown prep label: {label}")


def apply_meas_basis(circ: QuantumCircuit, qubit: int, cbit: int, basis: str) -> None:
    """
    Apply basis-change gates then measure qubit -> classical bit.

    Basis choices:
      X: H then measure
      Y: Sdg, H then measure
      Z: direct measure
    """
    if basis == "X":
        circ.h(qubit)
    elif basis == "Y":
        circ.sdg(qubit)
        circ.h(qubit)
    elif basis == "Z":
        pass
    else:
        raise ValueError(f"Unknown measurement basis: {basis}")
    circ.measure(qubit, cbit)


# ----------------------------------------------------------------------
# Building two-qubit tomography circuits
# ----------------------------------------------------------------------

def build_two_qubit_tomo_circuits() -> Dict[Tuple[str, str, str, str], QuantumCircuit]:
    """
    Build the full set of two-qubit process tomography circuits for the
    identity channel.

    Input states: product states from PREP_LABELS_1Q ⊗ PREP_LABELS_1Q.
    Measurement bases: all 9 combinations from MEAS_BASES_1Q ⊗ MEAS_BASES_1Q.

    We label circuits by (prep1, prep2, mb1, mb2).
    """
    circuits: Dict[Tuple[str, str, str, str], QuantumCircuit] = {}

    for prep1 in PREP_LABELS_1Q:
        for prep2 in PREP_LABELS_1Q:
            for mb1 in MEAS_BASES_1Q:
                for mb2 in MEAS_BASES_1Q:
                    qc = QuantumCircuit(2, 2, name=f"tomo_{prep1}_{prep2}_{mb1}{mb2}")
                    # Prepare product state |psi(prep1)> ⊗ |psi(prep2)>
                    prepare_1q_state(qc, 0, prep1)
                    prepare_1q_state(qc, 1, prep2)

                    # Identity channel here: place for inserting a gate or delay if desired

                    # Measurement in chosen bases
                    apply_meas_basis(qc, 0, 0, mb1)
                    apply_meas_basis(qc, 1, 1, mb2)

                    circuits[(prep1, prep2, mb1, mb2)] = qc

    return circuits


# ----------------------------------------------------------------------
# Expectation value reconstruction from counts
# ----------------------------------------------------------------------

def _bitstring_to_bits(bitstring: str, num_bits: int) -> List[int]:
    """
    Map a Qiskit bitstring (msb on the left) to a list of bits indexed
    by classical bit index, i.e. bits[q] = 0 or 1 for cbit q.

    Qiskit uses little-endian ordering so we reverse the string.
    """
    s = bitstring[::-1]
    return [int(s[i]) for i in range(num_bits)]


def expectation_single_qubit_from_counts(
    counts: Dict[str, int],
    qubit: int,
) -> float:
    """
    Compute expectation of a single-qubit Pauli observable on the
    specified qubit, given counts from a measurement in that Pauli basis.

    Assumes measurement was already rotated into that basis and that
    '0' corresponds to +1 eigenvalue, '1' to -1.
    """
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    accum = 0.0
    for bitstring, cnt in counts.items():
        bits = _bitstring_to_bits(bitstring, 2)
        b = bits[qubit]
        eigen = 1.0 if b == 0 else -1.0
        accum += eigen * cnt
    return accum / shots


def expectation_two_qubit_from_counts(
    counts: Dict[str, int],
) -> float:
    """
    Compute expectation of a tensor product Pauli observable σ_a ⊗ σ_b
    along the measurement axes, from counts in that basis.

    Assumes:
      - qubit 0 measured in Pauli a,
      - qubit 1 measured in Pauli b,
      - '0' -> +1, '1' -> -1 for each measured qubit.
    """
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    accum = 0.0
    for bitstring, cnt in counts.items():
        bits = _bitstring_to_bits(bitstring, 2)
        b0, b1 = bits[0], bits[1]
        e0 = 1.0 if b0 == 0 else -1.0
        e1 = 1.0 if b1 == 0 else -1.0
        accum += e0 * e1 * cnt
    return accum / shots


def reconstruct_output_bloch_two_qubit(
    meas_counts_for_state: Dict[Tuple[str, str], Dict[str, int]]
) -> np.ndarray:
    """
    Given measurement counts for a single input state, reconstruct the
    full 16-component expectation vector

        v_out[j] = <P_j>,

    where P_j runs over {I,X,Y,Z}⊗{I,X,Y,Z} in row-major order:
        (I⊗I, I⊗X, I⊗Y, I⊗Z,
         X⊗I, X⊗X, ..., Z⊗Z).

    The dict meas_counts_for_state maps (mb1, mb2) -> counts, where
    mb1, mb2 in {X, Y, Z} are the local measurement bases.
    """
    vec = np.zeros(16, dtype=float)
    paulis = PAULIS_1Q

    for i, a in enumerate(paulis):
        for j, b in enumerate(paulis):
            idx = 4 * i + j

            if a == "I" and b == "I":
                # Always 1 for any density matrix
                vec[idx] = 1.0
                continue

            if a == "I" and b != "I":
                # Use measurement with (Z, b) for convenience
                counts = meas_counts_for_state[("Z", b)]
                vec[idx] = expectation_single_qubit_from_counts(counts, qubit=1)
                continue

            if a != "I" and b == "I":
                # Use measurement with (a, Z)
                counts = meas_counts_for_state[(a, "Z")]
                vec[idx] = expectation_single_qubit_from_counts(counts, qubit=0)
                continue

            # Both non-identity: use measurement with (a, b)
            counts = meas_counts_for_state[(a, b)]
            vec[idx] = expectation_two_qubit_from_counts(counts)

    return vec


def build_input_bloch_two_qubit() -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute the ideal 16-component Pauli expectation vectors for the
    16 product input states.

    Single-qubit Bloch / Pauli expectation vectors are:

      Zp: [1, 0, 0,  1]
      Zm: [1, 0, 0, -1]
      Xp: [1, 1, 0,  0]
      Yp: [1, 0, 1,  0]

    For two qubits, expectations factorise:

      <σ_a ⊗ σ_b> = <σ_a>_1 <σ_b>_2.
    """
    single_map: Dict[str, np.ndarray] = {
        "Zp": np.array([1.0, 0.0, 0.0, 1.0]),
        "Zm": np.array([1.0, 0.0, 0.0, -1.0]),
        "Xp": np.array([1.0, 1.0, 0.0, 0.0]),
        "Yp": np.array([1.0, 0.0, 1.0, 0.0]),
    }

    out: Dict[Tuple[str, str], np.ndarray] = {}
    for p1 in PREP_LABELS_1Q:
        for p2 in PREP_LABELS_1Q:
            v1 = single_map[p1]
            v2 = single_map[p2]
            v = np.zeros(16, dtype=float)
            for i, a in enumerate(PAULIS_1Q):
                for j, b in enumerate(PAULIS_1Q):
                    idx = 4 * i + j
                    v[idx] = v1[i] * v2[j]
            out[(p1, p2)] = v
    return out


# ----------------------------------------------------------------------
# Matrix logarithm for generator extraction
# ----------------------------------------------------------------------

def matrix_log_diag(A: np.ndarray) -> np.ndarray:
    """
    Matrix logarithm using eigen-decomposition.

    Assumes A is diagonalizable; for realistic noisy channels this should
    hold for the Pauli transfer block. Returns the real part of the log.
    """
    vals, vecs = np.linalg.eig(A)
    if np.any(np.isclose(vals, 0)):
        raise ValueError("Matrix has (near-)zero eigenvalues, cannot take log")

    L_vals = np.log(vals)
    V_inv = np.linalg.inv(vecs)
    L = vecs @ np.diag(L_vals) @ V_inv
    return L.real


# ----------------------------------------------------------------------
# IQM K-tomography runner
# ----------------------------------------------------------------------

@dataclass
class TwoQubitKTomographyIQMConfig:
    iqm_server_url: str
    backend_name: str
    shots: int = 8192
    tag: str = "iqm_two_qubit_k_tomography"
    out_dir: str = "iqm_results"
    optimization_level: int = 1


class TwoQubitKTomographyIQMRunner:
    def __init__(self, cfg: TwoQubitKTomographyIQMConfig):
        self.cfg = cfg
        print(f"[IQM] Initialising IQMProvider with server URL: {cfg.iqm_server_url}")
        self.provider = IQMProvider(cfg.iqm_server_url)

        try:
            self.backend = self.provider.get_backend(cfg.backend_name)
        except Exception as e:
            print(f"[IQM] Could not obtain backend '{cfg.backend_name}': {e}")
            try:
                print("[IQM] Available backends for this server / account:")
                for b in self.provider.backends():
                    try:
                        name = b.name
                    except AttributeError:
                        name = str(b)
                    print(f"   - {name}")
            except Exception as e2:
                print(f"[IQM] Additionally failed to list backends: {e2}")
            raise

        os.makedirs(cfg.out_dir, exist_ok=True)

    def run(self) -> None:
        print(f"[IQM] Using backend: {self.backend}")
        print(f"[IQM] Shots per circuit: {self.cfg.shots}")

        circuits = build_two_qubit_tomo_circuits()
        keys = list(circuits.keys())
        circ_list = [circuits[k] for k in keys]

        print(
            f"[IQM] Built {len(circ_list)} tomography circuits "
            f"for 16 input states and 9 measurement settings each."
        )

        # Transpile
        t0 = time.time()
        transpiled = transpile(
            circ_list,
            backend=self.backend,
            optimization_level=self.cfg.optimization_level,
        )
        t1 = time.time()
        print(f"[IQM] Transpilation done in {t1 - t0:.1f} seconds")

        # Run on backend
        print("[IQM] Submitting tomography circuits via IQMBackend.run...")
        t2 = time.time()
        job = self.backend.run(transpiled, shots=self.cfg.shots)
        job_id = getattr(job, "job_id", lambda: "unknown")()
        print(f"[IQM] Job submitted, id: {job_id}")
        print("[IQM] Waiting for job result...")
        result = job.result()
        t3 = time.time()
        print(f"[IQM] Job completed in {t3 - t2:.1f} seconds")

        # Extract counts for each circuit in order
        raw_counts = result.get_counts()
        if isinstance(raw_counts, dict):
            # This should not happen for >1 circuit, but handle defensively
            raw_counts_list: List[Dict[str, int]] = [raw_counts]
        else:
            raw_counts_list = list(raw_counts)

        if len(raw_counts_list) != len(keys):
            raise RuntimeError(
                f"[IQM] Mismatch: {len(raw_counts_list)} count-sets for "
                f"{len(keys)} circuits."
            )

        counts_map: Dict[Tuple[str, str, str, str], Dict[str, int]] = {}
        for key, cnts in zip(keys, raw_counts_list):
            counts_norm = {str(bitstr): int(val) for bitstr, val in cnts.items()}
            counts_map[key] = counts_norm

        print("[IQM] Reconstructing output Bloch vectors for all input states...")

        # Build input expectation vectors
        bloch_in_map = build_input_bloch_two_qubit()

        # Reconstruct output Bloch vectors per input state
        input_labels = [(p1, p2) for p1 in PREP_LABELS_1Q for p2 in PREP_LABELS_1Q]
        num_states = len(input_labels)
        assert num_states == 16

        V_in = np.zeros((16, num_states), dtype=float)
        V_out = np.zeros((16, num_states), dtype=float)

        for col, (p1, p2) in enumerate(input_labels):
            V_in[:, col] = bloch_in_map[(p1, p2)]

            meas_counts_for_state: Dict[Tuple[str, str], Dict[str, int]] = {}
            for mb1 in MEAS_BASES_1Q:
                for mb2 in MEAS_BASES_1Q:
                    key = (p1, p2, mb1, mb2)
                    meas_counts_for_state[(mb1, mb2)] = counts_map[key]

            v_out = reconstruct_output_bloch_two_qubit(meas_counts_for_state)
            V_out[:, col] = v_out

        print("[IQM] Solving for 16x16 Pauli transfer matrix T...")
        cond_num = np.linalg.cond(V_in)
        print(f"[IQM] Condition number of V_in: {cond_num:.2e}")
        if cond_num > 1e6:
            print("[IQM] Warning: V_in is ill-conditioned; T may be noisy.")

        T = V_out @ np.linalg.inv(V_in)

        idx_non_id = list(range(1, 16))
        T_tr = T[np.ix_(idx_non_id, idx_non_id)]

        print("[IQM] Computing effective generator K from T_tr via matrix log...")
        try:
            K = matrix_log_diag(T_tr)
            evals, _ = np.linalg.eig(K)
        except Exception as e:
            print(f"[IQM] Failed to compute matrix logarithm: {e}")
            K = np.full_like(T_tr, np.nan)
            evals = np.full(T_tr.shape[0], np.nan)

        print("[IQM] Eigenvalues of K (traceless 15x15 block):")
        for ev in evals:
            print(f"    {ev}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{self.cfg.tag}_{self.cfg.backend_name}_{ts}"
        npz_path = os.path.join(self.cfg.out_dir, base + ".npz")
        json_path = os.path.join(self.cfg.out_dir, base + ".json")

        np.savez_compressed(
            npz_path,
            T=T,
            T_tr=T_tr,
            K=K,
            K_evals=evals,
            V_in=V_in,
            V_out=V_out,
        )

        out_data = {
            "iqm_server_url": self.cfg.iqm_server_url,
            "backend_name": self.cfg.backend_name,
            "shots": int(self.cfg.shots),
            "job_id": job_id,
            "tag": self.cfg.tag,
            "timestamp": ts,
            "cond_V_in": float(cond_num),
            "K_evals_real": evals.real.tolist(),
            "K_evals_imag": evals.imag.tolist(),
        }

        with open(json_path, "w", encoding="utf8") as f:
            json.dump(out_data, f, indent=2)

        print(f"[IQM] Saved arrays to {npz_path}")
        print(f"[IQM] Saved metadata to {json_path}")
        print("[IQM] Two-qubit K-tomography run complete.")


def main():
    # IQM server URL and backend from environment, with sensible fallbacks
    iqm_server_url = os.getenv(
        "IQM_SERVER_URL",
        "https://cocos.resonance.meetiqm.com/garnet:mock",
    )
    backend_name = os.getenv("QAPI_IQM_BACKEND", "facade_garnet")
    shots = int(os.getenv("QAPI_KTOMO_SHOTS", "8192"))

    cfg = TwoQubitKTomographyIQMConfig(
        iqm_server_url=iqm_server_url,
        backend_name=backend_name,
        shots=shots,
        tag="iqm_two_qubit_k_tomography",
        out_dir="iqm_results",
        optimization_level=1,
    )
    runner = TwoQubitKTomographyIQMRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
