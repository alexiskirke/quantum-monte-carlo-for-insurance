"""Tests for quantum state preparation, oracle, and Grover estimation."""

import numpy as np
import pytest

from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_state_prep,
    build_oracle_A,
    exact_amplitude_readout,
    grover_boosted_estimate,
    max_safe_k,
    get_statevector_backend,
)
from qiskit import transpile
from qiskit.quantum_info import Statevector


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


class TestStatePrep:
    """Verify the custom state preparation produces correct amplitudes."""

    def test_3qubit_amplitudes(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        sp = build_state_prep(probs, 3)
        sv = Statevector.from_instruction(sp)
        amp_sq = np.abs(sv.data) ** 2
        np.testing.assert_allclose(amp_sq, probs, atol=1e-12)

    def test_4qubit_amplitudes(self):
        midpoints, probs = discretise_distribution(0.667, 0.0, 97202.748, n_qubits=4)
        sp = build_state_prep(probs, 4)
        sv = Statevector.from_instruction(sp)
        amp_sq = np.abs(sv.data) ** 2
        np.testing.assert_allclose(amp_sq, probs, atol=1e-12)

    def test_uniform_distribution(self):
        probs = np.ones(8) / 8
        sp = build_state_prep(probs, 3)
        sv = Statevector.from_instruction(sp)
        amp_sq = np.abs(sv.data) ** 2
        np.testing.assert_allclose(amp_sq, probs, atol=1e-12)


class TestOracleA:
    def test_circuit_qubit_count(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        A, obj, _ = build_oracle_A(probs, midpoints, catastrophe_threshold=0.0)
        assert A.num_qubits == 4
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

    def test_readout_matches_analytical(self):
        """P(|1⟩) should equal Σ pᵢ · f̃(xᵢ)."""
        midpoints, probs = discretise_distribution(0.667, 0.0, 97202.748, n_qubits=3)
        threshold = float(np.percentile(midpoints, 70))
        excess = np.maximum(0.0, midpoints - threshold)
        max_excess = excess.max() if excess.max() > 0 else 1.0
        expected = float(np.dot(probs, excess / max_excess))

        A, obj, rescale = build_oracle_A(probs, midpoints, threshold)
        actual = exact_amplitude_readout(A, obj)
        assert abs(actual - expected) < 1e-5, \
            f"Readout {actual:.6f} != analytical {expected:.6f}"


class TestMaxSafeK:
    def test_small_probability(self):
        assert max_safe_k(0.001) >= 10

    def test_large_probability(self):
        assert max_safe_k(0.4) == 0

    def test_zero(self):
        assert max_safe_k(0.0) == 0


class TestGroverBoosted:
    def test_k0_runs_without_crash(self):
        midpoints, probs = discretise_distribution(1.0, 0.0, 50_000.0, n_qubits=3)
        threshold = float(np.percentile(midpoints, 80))
        A, obj, rescale = build_oracle_A(probs, midpoints, threshold)
        est, info = grover_boosted_estimate(A, obj, k_iters=0, shots=512)
        assert 0.0 <= est <= 1.0
        assert info["circuit_depth"] > 0

    def test_k1_runs_without_crash(self):
        midpoints, probs = discretise_distribution(0.667, 0.0, 97202.748, n_qubits=3)
        threshold = float(np.percentile(midpoints, 90))
        A, obj, rescale = build_oracle_A(probs, midpoints, threshold)
        prob = exact_amplitude_readout(A, obj)
        if max_safe_k(prob) >= 1:
            est, info = grover_boosted_estimate(A, obj, k_iters=1, shots=512)
            assert 0.0 <= est <= 1.0
            assert info["k_iters"] == 1

    def test_k3_produces_reasonable_estimate(self):
        """With k=3 Grover iterations, the estimate should be close to exact."""
        midpoints, probs = discretise_distribution(0.667, 0.0, 97202.748, n_qubits=3)
        threshold = float(np.percentile(midpoints, 90))
        A, obj, rescale = build_oracle_A(probs, midpoints, threshold)
        prob = exact_amplitude_readout(A, obj)
        k_safe = max_safe_k(prob)
        if k_safe >= 3:
            est, _ = grover_boosted_estimate(A, obj, k_iters=3, shots=4096)
            exact_excess = float(np.dot(probs, np.maximum(0, midpoints - threshold)))
            est_loss = est * rescale
            rel_err = abs(est_loss - exact_excess) / exact_excess if exact_excess > 0 else 0
            assert rel_err < 0.15, \
                f"k=3 estimate ${est_loss:.2f} too far from exact ${exact_excess:.2f}"
