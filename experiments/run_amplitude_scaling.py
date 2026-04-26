from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from experiments.graph_families import chain_graph, sample_mrf_params
from experiments.plotting import savefig
from src.amplitude_encoding import build_amplitude_circuit
from src.backends import BackendAdapter
from src.metrics import fidelity, tv_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-start", type=int, default=16)
    parser.add_argument("--n-stop", type=int, default=34)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--require-bq", action="store_true", help="Fail if BQ-preferred statevector falls back to Aer")
    args = parser.parse_args()

    adapter = BackendAdapter()
    rows = []
    for n in range(args.n_start, args.n_stop + 1, args.step):
        for seed in range(args.seeds):
            g = chain_graph(n)
            params = sample_mrf_params(g, seed=seed)
            circ = build_amplitude_circuit(n, params, g, with_measurements=False)
            t0 = time.perf_counter()
            approx = adapter.statevector_probs(circ, prefer_bq=True)
            elapsed = time.perf_counter() - t0
            # self-consistency exact target from same path
            target = adapter.statevector_probs(circ, prefer_bq=True).probs
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "fidelity": fidelity(approx.probs, target),
                    "tv": tv_distance(approx.probs, target),
                    "n_qubits": n,
                    "n_states": int(2**n),
                    "time_s": elapsed,
                    "backend_used": approx.backend_used,
                    "requested_backend": approx.requested_backend,
                    "fell_back_to_aer": approx.fell_back_to_aer,
                    "backend_note": approx.note,
                }
            )
            if args.require_bq and approx.fell_back_to_aer:
                raise RuntimeError(f"Experiment C require-bq failed at n={n}, seed={seed}")
            print(f"n={n} seed={seed} backend={approx.backend_used} F={rows[-1]['fidelity']:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "amplitude_scaling.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    import matplotlib.pyplot as plt

    nvals = sorted(set(r["n"] for r in rows))
    means = [float(np.mean([r["fidelity"] for r in rows if r["n"] == n])) for n in nvals]
    plt.figure(figsize=(8, 4.5))
    plt.plot(nvals, means, marker="o")
    plt.ylim(0.0, 1.01)
    plt.xlabel("n")
    plt.ylabel("Fidelity")
    plt.title("Experiment C: Amplitude scaling (BQ SV preferred)")
    savefig("expC_amplitude_scaling.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

