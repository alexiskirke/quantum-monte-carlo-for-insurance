"""Tests for quantum state preparation and payoff circuits."""

import numpy as np
import pytest

from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_oracle_A,
    exact_amplitude_readout,
    grover_boosted_estimate,
    _compute_rewards,
    get_statevector_backend,
)
from qiskit import transpile


class TestDiscretisation:
    def test_probs_sum_to_one(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_correct_bin_count(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=4)
        assert len(midpoints) == 16
        assert len(probs) == 16

    def test_midpoints_ordered(self):
        midpoints, _ = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        assert np.all(np.diff(midpoints) > 0)


class TestRewards:
    def test_rewards_in_unit_interval(self):
        _, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        midpoints = np.linspace(10_000, 200_000, 8)
        rewards, rescale = _compute_rewards(probs, midpoints, catastrophe_threshold=100_000)
        assert np.all(rewards >= 0.0)
        assert np.all(rewards <= 1.0 + 1e-10)

    def test_zero_excess_when_threshold_high(self):
        midpoints = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
        probs = np.ones(8) / 8
        rewards, rescale = _compute_rewards(probs, midpoints, catastrophe_threshold=100.0)
        assert np.allclose(rewards, 0.0)


class TestOracleA:
    def test_circuit_qubit_count(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        A, obj, _ = build_oracle_A(probs, midpoints, catastrophe_threshold=0.0)
        assert A.num_qubits == 4  # 3 index + 1 ancilla
        assert obj == 3

    def test_statevector_normalised(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        A, obj, _ = build_oracle_A(probs, midpoints, catastrophe_threshold=0.0)
        A.save_statevector()
        backend = get_statevector_backend()
        tc = transpile(A, backend)
        sv = np.array(backend.run(tc, shots=1).result().get_statevector())
        norm = np.sum(np.abs(sv) ** 2)
        assert abs(norm - 1.0) < 1e-10

    def test_readout_probability_in_range(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        threshold = float(np.percentile(midpoints, 50))
        A, obj, _ = build_oracle_A(probs, midpoints, threshold)
        prob = exact_amplitude_readout(A, obj)
        assert 0.0 <= prob <= 1.0, f"Readout probability {prob} out of [0,1]"


class TestGroverBoosted:
    def test_k0_runs_without_crash(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        A, obj, rescale = build_oracle_A(probs, midpoints, catastrophe_threshold=0.0)
        est, info = grover_boosted_estimate(A, obj, k_iters=0, shots=512)
        assert 0.0 <= est <= 1.0
        assert info["circuit_depth"] > 0

    def test_k1_runs_without_crash(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        A, obj, rescale = build_oracle_A(probs, midpoints, catastrophe_threshold=0.0)
        est, info = grover_boosted_estimate(A, obj, k_iters=1, shots=512)
        assert 0.0 <= est <= 1.0
        assert info["k_iters"] == 1
