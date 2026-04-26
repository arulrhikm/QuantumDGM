from __future__ import annotations

import json
import os

import numpy as np

from experiments.graph_families import erdos_renyi, sample_mrf_params
from src.amplitude_encoding import pairwise_probs_from_params


def main():
    rows = []
    for n in [8, 10, 12]:
        for seed in range(3):
            g = erdos_renyi(n, p=0.5, seed=seed)
            params = sample_mrf_params(g, seed=seed)
            p = pairwise_probs_from_params(n, params)
            topk = np.sort(p)[::-1][: max(4, len(p) // 16)]
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "head_mass_topk": float(np.sum(topk)),
                    "entropy_bits": float(-np.sum(p * np.log2(p + 1e-12))),
                    "max_prob": float(np.max(p)),
                }
            )
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fidelity_decomposition.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

