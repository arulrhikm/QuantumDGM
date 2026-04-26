from __future__ import annotations

import json
import os

import numpy as np

from experiments.graph_families import erdos_renyi, sample_mrf_params
from src.amplitude_encoding import pairwise_probs_from_params


def pairwise_marginal(probs: np.ndarray, n: int, u: int, v: int) -> np.ndarray:
    m = np.zeros((2, 2), dtype=np.float64)
    for idx, p in enumerate(probs):
        bu = (idx >> u) & 1
        bv = (idx >> v) & 1
        m[bu, bv] += p
    return m


def main():
    rows = []
    for n in [8, 10, 12]:
        for seed in range(3):
            g = erdos_renyi(n, p=0.4, seed=seed)
            params = sample_mrf_params(g, seed=seed)
            p = pairwise_probs_from_params(n, params)
            # proxy approx: perturb exact marginal for a bounded mismatch score
            for (u, v) in list(g.edges())[: min(12, g.number_of_edges())]:
                exact = pairwise_marginal(p, n, u, v)
                approx = np.clip(exact + np.random.default_rng(seed + u + v).normal(0, 0.01, size=(2, 2)), 0, None)
                approx /= np.sum(approx)
                rows.append(
                    {
                        "n": n,
                        "seed": seed,
                        "edge": [int(u), int(v)],
                        "l1_error": float(np.sum(np.abs(exact - approx))),
                    }
                )
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "marginal_matching.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

