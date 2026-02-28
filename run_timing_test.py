"""Minimal timing test: just build the oracle at n=8 and time each step."""
import time
import numpy as np

from fairy_queen.data_pipeline import get_synthetic_loss_data
from fairy_queen.quantum_circuits import (
    discretise_distribution, build_oracle_A, exact_amplitude_readout,
    grover_boosted_estimate, max_safe_k,
)

losses, dp = get_synthetic_loss_data()
shape, loc, scale = dp
threshold = float(np.percentile(losses, 95))

for n in [3, 5, 7, 8]:
    print(f"\nn={n} ({2**n} bins):")
    t0 = time.time()
    mp, pr = discretise_distribution(shape, loc, scale, n)
    print(f"  discretise: {time.time()-t0:.1f}s")

    t1 = time.time()
    A, oq, rsc = build_oracle_A(pr, mp, threshold)
    print(f"  build_oracle_A: {time.time()-t1:.1f}s  ({A.num_qubits} qubits, {A.size()} gates)")

    t2 = time.time()
    ep = exact_amplitude_readout(A, oq)
    ks = max_safe_k(ep)
    print(f"  exact_readout: {time.time()-t2:.1f}s  P(|1>)={ep:.6f} k_max={ks}")

    t3 = time.time()
    est, info = grover_boosted_estimate(A, oq, k_iters=1, shots=10)
    print(f"  1 Grover run (k=1, 10 shots): {time.time()-t3:.1f}s")
    print(f"  circuit depth={info.get('circuit_depth','?')}, gates={info.get('gate_count','?')}")
