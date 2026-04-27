"""
experiments/run_inverse_cdf_baseline.py
=======================================
E1 — Exact Inverse-CDF Baseline  (Experiment E1 from revised_plan_v4.tex)

This script uses the IDENTICAL instance-generation pipeline as run_ess_sweep.py:
  - graph_families.family_registry() + sample_mrf_params()
  - src.amplitude_encoding.pairwise_probs_from_params()
  - src.metrics.ess_ratio_from_series()   (returns ESS/N ratio in (0,1])

The scalar observable is Hamming weight (bin(x).count("1")), exactly matching
the ESS sweep, so ESS ratios are directly comparable.

For the quantum sampler row, we additionally record t_preprocess_s (the time
to enumerate P_theta with pairwise_probs_from_params) and report an amortized
ESS/s that includes this cost — which is the fair apples-to-apples comparison
since the inverse-CDF builder incurs the same O(2^n) preprocessing.

Output
------
  experiments/results/inverse_cdf_baseline.json

Usage
-----
  python -m experiments.run_inverse_cdf_baseline
  python -m experiments.run_inverse_cdf_baseline --n-values 8,10 --seeds 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import numpy as np

from experiments.graph_families import family_registry, sample_mrf_params
from src.amplitude_encoding import pairwise_probs_from_params
from src.metrics import ess_ratio_from_series


N_SAMPLES = 3000
BURN_IN   = 0          # inverse-CDF draws i.i.d. samples; no burn-in needed
FAMILIES_DEFAULT = "chain,erdos_renyi,two_clique,barbell,barbell_path"


def _icdf_samples(probs: np.ndarray, n_samples: int, rng: np.random.Generator):
    """
    Draw n_samples i.i.d. integer indices from `probs` using NumPy vectorised
    search on the pre-built CDF.  Returns (samples: np.ndarray[int], t_cdf_build_s, t_sample_s).
    """
    # --- build CDF (timed separately = O(2^n) preprocessing) ---
    t_pre0 = time.perf_counter()
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0          # guarantee exact 1.0 at tail
    t_cdf_build_s = time.perf_counter() - t_pre0

    # --- draw samples via searchsorted (each sample O(log 2^n) = O(n)) ---
    t_samp0 = time.perf_counter()
    u = rng.random(n_samples)
    samples = np.searchsorted(cdf, u, side="right")
    samples = np.clip(samples, 0, len(probs) - 1).astype(int)
    t_sample_s = time.perf_counter() - t_samp0

    return samples, t_cdf_build_s, t_sample_s


def _scalar_series(idx: np.ndarray) -> np.ndarray:
    """
    Convert integer configuration indices to Hamming-weight scalars.
    Matches the observable used in run_ess_sweep.py.
    """
    return np.array([bin(int(x)).count("1") for x in idx], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="E1: Exact inverse-CDF baseline")
    parser.add_argument("--n-values",  default="8,10,12")
    parser.add_argument("--seeds",     type=int, default=4,
                        help="Number of seeds per (family, n) pair (default 4, matching ess_sweep default)")
    parser.add_argument("--families",  default=FAMILIES_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    n_values  = [int(x) for x in args.n_values.split(",")]
    fam_order = [x.strip() for x in args.families.split(",") if x.strip()]
    n_samples = args.n_samples
    n_seeds   = args.seeds

    families  = family_registry()
    rng_global = np.random.default_rng(1234)   # reproducible sub-rngs per instance

    rows = []
    for fam_name in fam_order:
        if fam_name not in families:
            raise ValueError(f"Unknown family '{fam_name}'. Available: {list(families)}")
        for n in n_values:
            for seed in range(n_seeds):
                label = f"{fam_name}_n{n}_seed{seed}"
                print(f"  {label} ...", end=" ", flush=True)

                # ── Reconstruct instance exactly as in run_ess_sweep.py ───────
                g      = families[fam_name](n, seed)
                params = sample_mrf_params(g, seed=seed)

                # ── Time the P_theta computation (shared preprocessing cost) ──
                t_pre0 = time.perf_counter()
                probs  = pairwise_probs_from_params(n, params)
                t_preprocess_s = time.perf_counter() - t_pre0

                # ── Quantum i.i.d. samples (np.random.choice on probs) ────────
                # This matches run_ess_sweep.py exactly.
                rng_inst = np.random.default_rng(rng_global.integers(2**31))
                t_q0 = time.perf_counter()
                q_idx    = rng_inst.choice(len(probs), size=n_samples, p=probs)
                t_quantum_sample_s = time.perf_counter() - t_q0

                q_series = _scalar_series(q_idx)
                ess_ratio_q = ess_ratio_from_series(q_series)    # ratio in (0,1]
                ess_q       = ess_ratio_q * n_samples            # absolute ESS
                ess_per_s_q_sample_only = ess_q / max(1e-12, t_quantum_sample_s)
                # Amortized = include the O(2^n) Hamiltonian-diagonal preprocessing
                ess_per_s_q_amortized   = ess_q / max(1e-12, t_preprocess_s + t_quantum_sample_s)

                # ── Exact inverse-CDF samples ─────────────────────────────────
                icdf_idx, t_cdf_build_s, t_sample_s = _icdf_samples(
                    probs, n_samples, rng_inst)
                icdf_series = _scalar_series(icdf_idx)

                ess_ratio_icdf = ess_ratio_from_series(icdf_series)
                ess_icdf       = ess_ratio_icdf * n_samples
                t_total_icdf   = t_cdf_build_s + t_sample_s     # same O(2^n) cost

                ess_per_s_icdf_sample_only = ess_icdf / max(1e-12, t_sample_s)
                ess_per_s_icdf_amortized   = ess_icdf / max(1e-12, t_total_icdf)

                print(
                    f"Q ESS={ess_q:.1f} ({ess_per_s_q_amortized:.0f}/s amort)  |  "
                    f"CDF ESS={ess_icdf:.1f} ({ess_per_s_icdf_amortized:.0f}/s amort)  "
                    f"(t_pre={t_preprocess_s:.4f}s  t_cdf_build={t_cdf_build_s:.5f}s  t_samp={t_sample_s:.4f}s)"
                )

                rows.append({
                    "instance_id":                   label,
                    "family":                        fam_name,
                    "n":                             n,
                    "seed":                          seed,
                    # Preprocessing (shared cost)
                    "t_preprocess_ptable_s":         round(t_preprocess_s, 6),
                    # Quantum sampler
                    "t_quantum_sample_s":            round(t_quantum_sample_s, 6),
                    "ess_quantum":                   round(float(ess_q), 4),
                    "ess_ratio_quantum":             round(float(ess_ratio_q), 6),
                    "ess_per_s_quantum_sample_only": round(ess_per_s_q_sample_only, 4),
                    "ess_per_s_quantum_amortized":   round(ess_per_s_q_amortized, 4),
                    # Inverse-CDF sampler
                    "t_cdf_build_s":                 round(t_cdf_build_s, 6),
                    "t_icdf_sample_s":               round(t_sample_s, 6),
                    "t_icdf_total_s":                round(t_total_icdf, 6),
                    "ess_icdf":                      round(float(ess_icdf), 4),
                    "ess_ratio_icdf":                round(float(ess_ratio_icdf), 6),
                    "ess_per_s_icdf_sample_only":    round(ess_per_s_icdf_sample_only, 4),
                    "ess_per_s_icdf_amortized":      round(ess_per_s_icdf_amortized, 4),
                    # Ratio (icdf amortized vs quantum amortized)
                    "ratio_icdf_over_quantum_amortized": round(
                        ess_per_s_icdf_amortized / max(1e-12, ess_per_s_q_amortized), 4),
                })

    # ── Summary ───────────────────────────────────────────────────────────────
    def _mean(key): return float(np.mean([r[key] for r in rows]))
    def _med(key):  return float(np.median([r[key] for r in rows]))
    def _min(key):  return float(np.min([r[key] for r in rows]))
    def _max(key):  return float(np.max([r[key] for r in rows]))

    summary = {
        "n_instances":                        len(rows),
        # Quantum amortized ESS/s
        "mean_ess_per_s_quantum_amortized":   round(_mean("ess_per_s_quantum_amortized"), 4),
        "median_ess_per_s_quantum_amortized": round(_med ("ess_per_s_quantum_amortized"), 4),
        # Inverse-CDF sample-only ESS/s
        "mean_ess_per_s_icdf_sample_only":    round(_mean("ess_per_s_icdf_sample_only"), 4),
        "median_ess_per_s_icdf_sample_only":  round(_med ("ess_per_s_icdf_sample_only"), 4),
        # Inverse-CDF amortized ESS/s
        "mean_ess_per_s_icdf_amortized":      round(_mean("ess_per_s_icdf_amortized"),   4),
        "median_ess_per_s_icdf_amortized":    round(_med ("ess_per_s_icdf_amortized"),   4),
        # Ratio: how many times faster is inverse-CDF (amortized) than quantum (amortized)
        "mean_ratio_icdf_over_quantum_amortized":   round(_mean("ratio_icdf_over_quantum_amortized"), 4),
        "median_ratio_icdf_over_quantum_amortized": round(_med ("ratio_icdf_over_quantum_amortized"), 4),
        "min_ratio_icdf_over_quantum_amortized":    round(_min ("ratio_icdf_over_quantum_amortized"), 4),
        "max_ratio_icdf_over_quantum_amortized":    round(_max ("ratio_icdf_over_quantum_amortized"), 4),
    }

    out = {
        "generated_utc":            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_samples_per_instance":   n_samples,
        "burn_in":                  BURN_IN,
        "scalar_observable":        "hamming_weight",
        "ess_metric":               "ess_ratio_from_series * n_samples  (matches run_ess_sweep.py)",
        "per_instance":             rows,
        "summary":                  summary,
    }

    out_dir  = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inverse_cdf_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[E1] Done. Wrote {out_path}")
    print(f"     mean quantum amortized ESS/s = {summary['mean_ess_per_s_quantum_amortized']:.1f}")
    print(f"     mean iCDF   amortized ESS/s  = {summary['mean_ess_per_s_icdf_amortized']:.1f}")
    print(f"     mean iCDF/quantum ratio (amortized) = {summary['mean_ratio_icdf_over_quantum_amortized']:.2f}x")


if __name__ == "__main__":
    main()
