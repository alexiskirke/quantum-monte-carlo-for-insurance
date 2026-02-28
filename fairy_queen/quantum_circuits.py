"""Quantum circuit construction for catastrophe-insurance amplitude estimation.

Key components
--------------
1. **Custom state preparation** – Loads the discretised loss distribution
   into qubit amplitudes using a binary tree of controlled-Ry rotations.
   For n qubits this requires 2^n − 1 rotations, using only {Ry, CRy, MCRy,
   CX, X} gates — avoiding Qiskit's StatePreparation which decomposes into
   `cu` gates that crash Aer.

2. **Oracle A (amplitude encoding)** – Combines state preparation with
   payoff-only rotations on an ancilla qubit:
     |0⟩ → Σ √pᵢ |i⟩ (cos(θᵢ/2)|0⟩ + sin(θᵢ/2)|1⟩)
   where sin²(θᵢ/2) = f̃(xᵢ) = max(0, xᵢ − M) / max_excess.
   The probability P(|1⟩) = Σ pᵢ · f̃(xᵢ) = E[f̃(X)], which is genuinely
   small for tail events — enabling safe Grover amplification.

3. **Grover operator** – Standard Q = A · S₀ · A† · Sχ, built using
   A.to_gate() which Aer handles after transpilation.

4. **Backend helpers** – Thin wrappers around AerSimulator.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from fairy_queen.logging_config import get_logger


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def get_statevector_backend() -> AerSimulator:
    return AerSimulator(method="statevector")


def get_qasm_backend() -> AerSimulator:
    return AerSimulator()


def build_noise_model(
    p1q: float = 0.001,
    p2q: float = 0.01,
    p_readout: float = 0.005,
) -> NoiseModel:
    """Depolarising + readout noise."""
    nm = NoiseModel()
    _1q_gates = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z",
                 "s", "sdg", "t", "tdg", "sx", "id"]
    _2q_gates = ["cx", "cz", "cy", "swap"]

    if p1q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), _1q_gates)
    if p2q > 0:
        err2 = depolarizing_error(p2q, 2)
        nm.add_all_qubit_quantum_error(err2, _2q_gates)
    if p_readout > 0:
        ro = ReadoutError([[1 - p_readout, p_readout],
                           [p_readout, 1 - p_readout]])
        nm.add_all_qubit_readout_error(ro)
    return nm


NOISE_PRESETS = {
    "low":    {"p1q": 0.001, "p2q": 0.01,  "p_ro": 0.005},
    "medium": {"p1q": 0.005, "p2q": 0.05,  "p_ro": 0.02},
    "high":   {"p1q": 0.01,  "p2q": 0.10,  "p_ro": 0.05},
}


def get_noisy_backend(label: str = "medium") -> Tuple[AerSimulator, NoiseModel]:
    p = NOISE_PRESETS[label]
    nm = build_noise_model(p["p1q"], p["p2q"], p["p_ro"])
    return AerSimulator(noise_model=nm), nm


# ---------------------------------------------------------------------------
# Distribution discretisation
# ---------------------------------------------------------------------------

def discretise_distribution(
    shape: float, loc: float, scale: float,
    n_qubits: int = 3,
    low: Optional[float] = None,
    high: Optional[float] = None,
    binning: str = "equal_width",
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise a lognormal into 2^n_qubits bins, return (midpoints, probs).

    binning: 'equal_width' (default) or 'quantile'.
    Quantile binning places equal probability mass in each bin, avoiding
    pathological midpoint placement on heavy-tailed distributions.
    """
    from scipy.stats import lognorm
    n_bins = 2 ** n_qubits

    if binning == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        quantiles[0] = 0.001
        quantiles[-1] = 0.999
        edges = lognorm.ppf(quantiles, shape, loc, scale)
    else:
        if low is None:
            low = lognorm.ppf(0.001, shape, loc, scale)
        if high is None:
            high = lognorm.ppf(0.999, shape, loc, scale)
        edges = np.linspace(low, high, n_bins + 1)

    probs = np.diff(lognorm.cdf(edges, shape, loc, scale))
    probs = probs / probs.sum()
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    return midpoints, probs


# ---------------------------------------------------------------------------
# Low-level gate helpers
# ---------------------------------------------------------------------------

def _controlled_ry_on_state(
    qc: QuantumCircuit,
    ctrl_qubits: list,
    ctrl_state_str: str,
    target: int,
    theta: float,
) -> None:
    """Ry(theta) on *target*, controlled on *ctrl_qubits* matching *ctrl_state_str*.

    Uses X flips to select the desired computational-basis control state,
    then a bare mcry (which fires on all-ones), then un-flips.

    ctrl_state_str[i] corresponds to ctrl_qubits[i]:
      '1' → qubit must be |1⟩  (no flip needed)
      '0' → qubit must be |0⟩  (X before and after mcry)
    """
    if abs(theta) < 1e-14:
        return
    flips = []
    for i, bit in enumerate(ctrl_state_str):
        if bit == "0":
            qc.x(ctrl_qubits[i])
            flips.append(ctrl_qubits[i])
    qc.mcry(theta, ctrl_qubits, target)
    for q in flips:
        qc.x(q)


# ---------------------------------------------------------------------------
# Custom state preparation  (binary tree of controlled-Ry)
# ---------------------------------------------------------------------------

def build_state_prep(probs: np.ndarray, n_qubits: int) -> QuantumCircuit:
    """Load √pᵢ into computational-basis amplitudes using controlled rotations.

    Works for any n_qubits.  Iterates from MSB (qubit n-1) down to LSB
    (qubit 0), splitting probability mass at each level with a
    (multi-)controlled Ry rotation.  Total rotations = 2^n − 1.

    Gate palette: {Ry, CRy, MCRy, CX, X} — no StatePreparation / cu gates.
    """
    N = 2 ** n_qubits
    assert len(probs) == N
    qc = QuantumCircuit(n_qubits, name="DistLoad")

    def _angle(p_high: float, p_total: float) -> float:
        if p_total < 1e-15:
            return 0.0
        return 2 * np.arcsin(np.sqrt(np.clip(p_high / p_total, 0.0, 1.0)))

    for level in range(n_qubits - 1, -1, -1):
        block_size = 2 ** (level + 1)
        half = block_size // 2
        n_blocks = N // block_size

        for b in range(n_blocks):
            start = b * block_size
            p_low = float(np.sum(probs[start: start + half]))
            p_high = float(np.sum(probs[start + half: start + block_size]))
            p_total = p_low + p_high

            theta = _angle(p_high, p_total)
            if abs(theta) < 1e-14:
                continue

            target = level
            ctrl_qubits = list(range(level + 1, n_qubits))

            if len(ctrl_qubits) == 0:
                qc.ry(theta, target)
            elif len(ctrl_qubits) == 1:
                # Build ctrl state for the single control qubit
                bit_val = (b >> (ctrl_qubits[0] - level - 1)) & 1
                if bit_val == 1:
                    qc.cry(theta, ctrl_qubits[0], target)
                else:
                    qc.x(ctrl_qubits[0])
                    qc.cry(theta, ctrl_qubits[0], target)
                    qc.x(ctrl_qubits[0])
            else:
                ctrl_state = "".join(
                    str((b >> (cq - level - 1)) & 1) for cq in ctrl_qubits
                )
                _controlled_ry_on_state(qc, ctrl_qubits, ctrl_state, target, theta)

    return qc


# ---------------------------------------------------------------------------
# Oracle A  (amplitude encoding: state prep + payoff rotation)
# ---------------------------------------------------------------------------

def build_oracle_A(
    probs: np.ndarray,
    midpoints: np.ndarray,
    catastrophe_threshold: float,
) -> Tuple[QuantumCircuit, int, float]:
    """Build oracle A for amplitude estimation with amplitude encoding.

    Circuit layout (n_index + 1 qubits, ancilla is the last qubit):

    1. **State preparation** on the index register:
         |0…0⟩ → Σ √pᵢ |i⟩
       using a binary tree of controlled-Ry rotations.

    2. **Payoff rotation** on the ancilla for each basis state |i⟩:
         |i⟩|0⟩ → |i⟩ ( cos(θᵢ/2)|0⟩ + sin(θᵢ/2)|1⟩ )
       where sin²(θᵢ/2) = f̃(xᵢ) = max(0, xᵢ − M) / max_excess.

    Result:
      P(ancilla = |1⟩) = Σ pᵢ · f̃(xᵢ) = E[excess] / max_excess
      E[excess] = P(|1⟩) × max_excess

    Returns (circuit, objective_qubit_index, rescale_factor).
    """
    log = get_logger()
    n_index = int(np.log2(len(probs)))
    n_total = n_index + 1
    ancilla = n_total - 1

    excess = np.maximum(0.0, midpoints - catastrophe_threshold)
    max_excess = float(excess.max())
    if max_excess < 1e-12:
        max_excess = 1.0
    f_tilde = excess / max_excess

    qc = QuantumCircuit(n_total, name="Oracle_A")

    # Step 1: state preparation on index qubits
    sp = build_state_prep(probs, n_index)
    qc.compose(sp, qubits=list(range(n_index)), inplace=True)

    # Step 2: payoff rotation on ancilla, controlled on each basis state
    ctrl_qs = list(range(n_index))
    for idx in range(len(f_tilde)):
        ft = float(np.clip(f_tilde[idx], 0.0, 1.0))
        if ft < 1e-14:
            continue
        theta = 2.0 * np.arcsin(np.sqrt(ft))
        # LSB-first bit string to match ctrl_qs = [0, 1, ..., n_index-1]
        bits = format(idx, f"0{n_index}b")[::-1]
        _controlled_ry_on_state(qc, ctrl_qs, bits, ancilla, theta)

    rescale = max_excess
    log.debug("Oracle A (amplitude encoding): %d qubits, depth %d, rescale=%.2f",
              n_total, qc.depth(), rescale)
    return qc, ancilla, rescale


build_estimation_circuit = build_oracle_A


# ---------------------------------------------------------------------------
# Statevector exact readout  (noiseless "ideal QAE" proxy)
# ---------------------------------------------------------------------------

def exact_amplitude_readout(
    A_circuit: QuantumCircuit, objective_qubit: int
) -> float:
    """Statevector readout of Pr(objective = |1⟩) — the target quantity of QAE."""
    backend = get_statevector_backend()
    qc = A_circuit.copy()
    qc.save_statevector()
    tc = transpile(qc, backend)
    sv = np.array(backend.run(tc, shots=1).result().get_statevector())

    prob = 0.0
    for idx in range(len(sv)):
        if (idx >> objective_qubit) & 1:
            prob += np.abs(sv[idx]) ** 2
    return float(prob)


# ---------------------------------------------------------------------------
# Shot-based Grover-boosted estimate
# ---------------------------------------------------------------------------

def max_safe_k(prob: float) -> int:
    """Largest Grover iteration count k such that (2k+1)θ < π/2."""
    if prob <= 0 or prob >= 1:
        return 0
    theta = np.arcsin(np.sqrt(prob))
    if theta < 1e-10:
        return 100
    return max(0, int(np.pi / (2 * theta) - 1) // 2)


def grover_boosted_estimate(
    A_circuit: QuantumCircuit,
    objective_qubit: int,
    k_iters: int = 1,
    shots: int = 8192,
    backend: Optional[AerSimulator] = None,
) -> Tuple[float, dict]:
    """Run A then k Grover iterations, measure, de-amplify to recover P(|1⟩).

    The Grover operator is built using A.to_gate() and elementary gates
    (H, X, Z, mcx) that Aer handles natively after transpilation.

    Returns (estimated_prob, info_dict).
    """
    n = A_circuit.num_qubits
    A_gate = A_circuit.to_gate(label="A")

    qc = QuantumCircuit(n)
    qc.append(A_gate, range(n))

    for _ in range(k_iters):
        # S_χ : flip phase of |1⟩ on objective qubit
        qc.z(objective_qubit)
        # A†
        qc.append(A_gate.inverse(), range(n))
        # S₀ : flip phase of |0…0⟩
        for i in range(n):
            qc.x(i)
        if n >= 2:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        else:
            qc.z(0)
        for i in range(n):
            qc.x(i)
        # A
        qc.append(A_gate, range(n))

    cl = ClassicalRegister(1, "c")
    qc.add_register(cl)
    qc.measure(objective_qubit, cl[0])

    if backend is None:
        backend = get_qasm_backend()

    tc = transpile(qc, backend)
    result = backend.run(tc, shots=shots).result()
    counts = result.get_counts()

    ones = sum(v for key, v in counts.items() if key.strip()[-1] == "1")
    total = sum(counts.values())
    meas_prob = ones / total

    if k_iters > 0:
        amp_theta = np.arcsin(np.sqrt(np.clip(meas_prob, 0, 1)))
        true_theta = amp_theta / (2 * k_iters + 1)
        est_prob = float(np.sin(true_theta) ** 2)
    else:
        est_prob = float(meas_prob)

    info = {
        "k_iters": k_iters,
        "shots": shots,
        "circuit_depth": tc.depth(),
        "gate_count": tc.size(),
        "measured_prob_raw": meas_prob,
    }
    return est_prob, info
