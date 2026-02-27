"""Quantum circuit construction for catastrophe-insurance amplitude estimation.

Key components
--------------
1. **Oracle A** – Encodes the probability-weighted payoff into a quantum state
   using the same robust pattern as the proven v2 project:
     a) Put n index qubits into uniform superposition via Hadamard gates.
     b) For each basis state |i⟩, apply a controlled-Ry rotation on an
        ancilla qubit whose angle encodes the product  pᵢ · f̃(xᵢ),
        where pᵢ is the discretised loss probability and f̃ is the
        normalised payoff.
   The probability of measuring ancilla = |1⟩ equals a rescaled version
   of E[f(X)].

2. **Grover operator** – Standard Q = A · S₀ · A† · Sχ built inline to
   avoid Aer-incompatible gate decompositions.

3. **Backend helpers** – Thin wrappers around AerSimulator.
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
    """Depolarising + readout noise (mirrors the v2 project's noise_models)."""
    nm = NoiseModel()
    _1q_gates = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z",
                 "s", "sdg", "t", "tdg", "sx", "id"]
    _2q_gates = ["cx", "cz", "cy", "swap"]

    if p1q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), _1q_gates)
    if p2q > 0:
        err2 = depolarizing_error(p2q, 2)
        nm.add_all_qubit_quantum_error(err2, _2q_gates)
        for g in ["mcx", "mcry"]:
            try:
                nm.add_all_qubit_quantum_error(err2, [g])
            except Exception:
                pass
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise a lognormal into 2^n_qubits bins, return (midpoints, probs)."""
    from scipy.stats import lognorm
    n_bins = 2 ** n_qubits
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
# Oracle A  (uniform superposition + controlled-Ry reward encoding)
# ---------------------------------------------------------------------------

def _compute_rewards(
    probs: np.ndarray,
    midpoints: np.ndarray,
    catastrophe_threshold: float,
) -> Tuple[np.ndarray, float]:
    """Compute normalised reward values for each bin.

    reward_i encodes  N · p_i · f̃(x_i)  rescaled into [0, 1], where
    f̃(x_i) = max(0, x_i − M) / max_excess.

    Returns (rewards_array, total_rescale_factor) so that:
        E[max(0, X−M)] = P(ancilla=|1⟩) × total_rescale_factor
    """
    N = len(probs)
    excess = np.maximum(0.0, midpoints - catastrophe_threshold)
    max_excess = excess.max()
    if max_excess < 1e-12:
        max_excess = 1.0
    f_tilde = excess / max_excess

    # u_i = p_i · excess_i is the probability-weighted excess for bin i.
    # We want reward_i = u_i / u_max ∈ [0,1] so that:
    #   P(ancilla=|1⟩) = (1/N) Σ reward_i = E[excess] / (N · u_max)
    # ⟹ E[excess] = P(|1⟩) · N · u_max = P(|1⟩) · rescale
    u = probs * f_tilde  # prob-weighted normalised payoff per bin
    u_max = u.max()
    if u_max < 1e-15:
        u_max = 1.0

    rewards = u / u_max
    total_rescale = N * u_max * max_excess
    return rewards.astype(np.float64), float(total_rescale)


def _controlled_ry_on_index(
    qc: QuantumCircuit, idx: int, n_index: int, theta: float, anc: int
):
    """Ry(theta) on ancilla qubit, controlled on index register == |idx⟩."""
    bits = format(idx, f"0{n_index}b")
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)
    qc.mcry(theta, list(range(n_index)), anc)
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)


def build_oracle_A(
    probs: np.ndarray,
    midpoints: np.ndarray,
    catastrophe_threshold: float,
) -> Tuple[QuantumCircuit, int, float]:
    """Build oracle A for amplitude estimation.

    Circuit structure (n_index + 1 qubits):
      - Hadamard on all index qubits → uniform superposition
      - For each basis state |i⟩, controlled-Ry(θ_i) on ancilla where
        sin²(θ_i/2) = reward_i  (probability-weighted normalised payoff)

    The probability of measuring the ancilla in |1⟩ equals:
        P(|1⟩) = (1/N) Σ reward_i

    And the true expected excess loss is recovered via:
        E[max(0, X−M)] = P(|1⟩) × rescale_factor

    Returns (circuit, objective_qubit_index, rescale_factor).
    """
    log = get_logger()

    n_index = int(np.log2(len(probs)))
    n_total = n_index + 1
    ancilla = n_total - 1

    rewards, rescale = _compute_rewards(probs, midpoints, catastrophe_threshold)

    qc = QuantumCircuit(n_total, name="Oracle_A")

    # Uniform superposition over loss-severity buckets
    for i in range(n_index):
        qc.h(i)

    # Controlled rotations encoding probability-weighted payoff
    for idx in range(len(rewards)):
        r = float(np.clip(rewards[idx], 0.0, 1.0))
        if r < 1e-14:
            continue
        theta = 2 * np.arcsin(np.sqrt(r))
        _controlled_ry_on_index(qc, idx, n_index, theta, ancilla)

    log.debug("Oracle A: %d qubits, depth %d", n_total, qc.depth())
    return qc, ancilla, rescale


# Alias for backward compatibility with experiments
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
# Shot-based Grover-boosted estimate (inline, same pattern as v2 project)
# ---------------------------------------------------------------------------

def grover_boosted_estimate(
    A_circuit: QuantumCircuit,
    objective_qubit: int,
    k_iters: int = 1,
    shots: int = 8192,
    backend: Optional[AerSimulator] = None,
) -> Tuple[float, dict]:
    """Run A then k Grover iterations, measure, correct for amplification.

    Builds the Grover operator inline using only H, X, Z, mcx, and mcry
    gates that Aer handles natively.

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
