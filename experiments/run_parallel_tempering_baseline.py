"""
experiments/run_parallel_tempering_baseline.py
===============================================
E2 — Parallel Tempering Baseline  (Experiment E2 from revised_plan_v4.tex)

This script uses the IDENTICAL instance-generation pipeline as run_ess_sweep.py:
  - graph_families.family_registry() + sample_mrf_params()
  - src.amplitude_encoding.pairwise_probs_from_params()
  - src.metrics.ess_ratio_from_series()   (returns ESS/N ratio in (0,1])

The Gibbs sweep inside each replica operates on integer state indices using
the precomputed `probs` array directly (the same approach as run_ess_sweep.py),
giving O(2^n) per site update — consistent with the single-site Gibbs baseline.

The scalar observable is Hamming weight (bin(x).count("1")), matching ess_sweep.

Protocol (from revised_plan_v4.tex §2.2)
-----------------------------------------
  K = 8 replicas, geometric beta ladder in [beta_min=0.1, beta_max=1.0]
  Replica swap proposals every L_swap = 10 within-replica sweeps (Metropolis)
  Burn-in: 1000 within-replica sweeps per replica
  3000 retained samples from the beta=1.0 replica
  Total wall-clock = all replica work combined

Output
------
  experiments/results/parallel_tempering_baseline.json

Usage
-----
  python -m experiments.run_parallel_tempering_baseline
  python -m experiments.run_parallel_tempering_baseline --n-replicas 4 --seeds 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time

import numpy as np

from experiments.graph_families import family_registry, sample_mrf_params
from src.amplitude_encoding import pairwise_probs_from_params
from src.metrics import ess_ratio_from_series


N_SAMPLES     = 3000
BURN_IN       = 1000
K_REPLICAS    = 8
BETA_MIN      = 0.1
BETA_MAX      = 1.0
L_SWAP        = 10
FAMILIES_DEFAULT = "chain,erdos_renyi,two_clique,barbell,barbell_path"


def _tempered_probs(probs: np.ndarray, beta: float) -> np.ndarray:
    """
    Return probs^beta / Z(beta).
    Uses log-space for numerical stability.
    """
    log_p = np.log(np.maximum(probs, 1e-300))
    log_pb = beta * log_p
    log_pb -= log_pb.max()
    pb = np.exp(log_pb)
    return pb / pb.sum()


def _single_site_gibbs_step(
    state: int, n: int, probs: np.ndarray, beta: float, rng: np.random.Generator
) -> int:
    """
    One random-site Gibbs update using the precomputed prob table.
    Matches the logic in run_ess_sweep.py:gibbs_sample_from_exact but for
    an arbitrary beta (via tempered_probs).
    """
    # Pick a random variable
    i = int(rng.integers(n))
    mask = 1 << i
    s0 = state & ~mask           # bit i = 0
    s1 = s0 | mask               # bit i = 1
    # Tempered conditional (proportional to probs^beta at these two states)
    p0 = float(probs[s0]) ** beta
    p1 = float(probs[s1]) ** beta
    z  = p0 + p1 + 1e-300
    return s1 if rng.random() < p1 / z else s0


def _gibbs_sweep(
    state: int, n: int, probs: np.ndarray, beta: float, rng: np.random.Generator
) -> int:
    """One full random-scan single-site sweep (n individual updates)."""
    for _i in rng.integers(n, size=n):   # random scan
        mask = 1 << int(_i)
        s0 = state & ~mask
        s1 = s0 | mask
        p0 = float(probs[s0]) ** beta
        p1 = float(probs[s1]) ** beta
        z  = p0 + p1 + 1e-300
        state = s1 if rng.random() < p1 / z else s0
    return state


def _swap_log_accept(probs: np.ndarray, s1: int, s2: int,
                     beta1: float, beta2: float) -> float:
    """
    Log Metropolis acceptance ratio for swapping replicas at beta1, beta2.
    log alpha = (beta2 - beta1) * (log p(s1) - log p(s2))
    """
    log_p1 = math.log(max(float(probs[s1]), 1e-300))
    log_p2 = math.log(max(float(probs[s2]), 1e-300))
    return (beta2 - beta1) * (log_p1 - log_p2)


def run_pt(probs: np.ndarray, n: int, n_replicas: int,
           beta_min: float, beta_max: float, l_swap: int,
           burn_in: int, n_samples: int, seed: int):
    """
    Full parallel-tempering run on a single instance.

    Returns
    -------
    samples        : np.ndarray of int, shape (n_samples,)  — from beta=1 replica
    swap_acc_rate  : float — mean swap acceptance rate
    t_total_s      : float — total wall-clock (all replicas)
    """
    rng = np.random.default_rng(seed)

    # Geometric beta ladder; last entry is beta_max = 1.0 (target)
    if n_replicas == 1:
        betas = np.array([beta_max])
    else:
        betas = np.array([
            beta_min * (beta_max / beta_min) ** (k / (n_replicas - 1))
            for k in range(n_replicas)
        ])
    target_idx = n_replicas - 1    # betas[-1] = beta_max = 1.0

    # Initialise each replica to a random configuration
    states = [int(rng.integers(2**n)) for _ in range(n_replicas)]

    n_swap_prop = 0
    n_swap_acc  = 0

    t0 = time.perf_counter()

    # ── Burn-in ───────────────────────────────────────────────────────────────
    for sweep in range(burn_in):
        for k in range(n_replicas):
            states[k] = _gibbs_sweep(states[k], n, probs, betas[k], rng)
        if (sweep + 1) % l_swap == 0:
            # Propose adjacent swaps in random order
            adj = rng.permutation(n_replicas - 1)
            for k in adj:
                k, k2 = int(k), int(k) + 1
                log_a = _swap_log_accept(probs, states[k], states[k2],
                                          betas[k], betas[k2])
                n_swap_prop += 1
                if math.log(max(float(rng.random()), 1e-300)) < log_a:
                    states[k], states[k2] = states[k2], states[k]
                    n_swap_acc += 1

    # ── Sampling ──────────────────────────────────────────────────────────────
    samples = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        for k in range(n_replicas):
            states[k] = _gibbs_sweep(states[k], n, probs, betas[k], rng)
        # Swap every l_swap sweeps during sampling too
        if (i + 1) % l_swap == 0:
            adj = rng.permutation(n_replicas - 1)
            for k in adj:
                k, k2 = int(k), int(k) + 1
                log_a = _swap_log_accept(probs, states[k], states[k2],
                                          betas[k], betas[k2])
                n_swap_prop += 1
                if math.log(max(float(rng.random()), 1e-300)) < log_a:
                    states[k], states[k2] = states[k2], states[k]
                    n_swap_acc += 1
        samples[i] = states[target_idx]

    t_total_s    = time.perf_counter() - t0
    swap_acc_rate = float(n_swap_acc / max(1, n_swap_prop))

    return samples, swap_acc_rate, t_total_s


def main():
    parser = argparse.ArgumentParser(description="E2: Parallel Tempering Baseline")
    parser.add_argument("--n-values",    default="8,10,12")
    parser.add_argument("--seeds",       type=int, default=4,
                        help="Number of seeds per (family, n) pair (default 4, matching ess_sweep)")
    parser.add_argument("--families",    default=FAMILIES_DEFAULT)
    parser.add_argument("--n-replicas",  type=int, default=K_REPLICAS)
    parser.add_argument("--beta-min",    type=float, default=BETA_MIN)
    parser.add_argument("--beta-max",    type=float, default=BETA_MAX)
    parser.add_argument("--l-swap",      type=int, default=L_SWAP)
    parser.add_argument("--burn-in",     type=int, default=BURN_IN)
    parser.add_argument("--n-samples",   type=int, default=N_SAMPLES)
    args = parser.parse_args()

    n_values  = [int(x) for x in args.n_values.split(",")]
    fam_order = [x.strip() for x in args.families.split(",") if x.strip()]
    families  = family_registry()

    rows = []
    for fam_name in fam_order:
        if fam_name not in families:
            raise ValueError(f"Unknown family '{fam_name}'. Available: {list(families)}")
        for n in n_values:
            for seed in range(args.seeds):
                label = f"{fam_name}_n{n}_seed{seed}"
                print(f"  {label} ...", end=" ", flush=True)

                g      = families[fam_name](n, seed)
                params = sample_mrf_params(g, seed=seed)
                probs  = pairwise_probs_from_params(n, params)

                # Also compute the quantum ESS for this instance (for ratio)
                rng_q = np.random.default_rng(seed * 100 + 99)
                t_q0 = time.perf_counter()
                q_idx    = rng_q.choice(len(probs), size=args.n_samples, p=probs)
                t_quantum_s = time.perf_counter() - t_q0
                q_series    = np.array([bin(int(x)).count("1") for x in q_idx], dtype=float)
                ess_ratio_q = ess_ratio_from_series(q_series)
                ess_q       = ess_ratio_q * args.n_samples

                # Run parallel tempering
                pt_samples, swap_acc, t_pt_s = run_pt(
                    probs, n,
                    n_replicas=args.n_replicas,
                    beta_min=args.beta_min,
                    beta_max=args.beta_max,
                    l_swap=args.l_swap,
                    burn_in=args.burn_in,
                    n_samples=args.n_samples,
                    seed=seed + 77777,
                )

                pt_series    = np.array([bin(int(x)).count("1") for x in pt_samples], dtype=float)
                ess_ratio_pt = ess_ratio_from_series(pt_series)
                ess_pt       = ess_ratio_pt * args.n_samples
                ess_per_s_pt = ess_pt / max(1e-12, t_pt_s)

                # Quantum/PT ESS ratio (in sample-count space, matching ess_sweep convention)
                ratio_q_over_pt = float(ess_q / max(1e-12, ess_pt))

                print(
                    f"Q ESS={ess_q:.1f}  PT ESS={ess_pt:.1f}  "
                    f"Q/PT={ratio_q_over_pt:.2f}  swap_acc={swap_acc:.1%}  t_pt={t_pt_s:.1f}s"
                )

                rows.append({
                    "instance_id":          label,
                    "family":               fam_name,
                    "n":                    n,
                    "seed":                 seed,
                    # Quantum reference (sampled fresh here for per-instance ratio)
                    "ess_quantum":          round(float(ess_q), 4),
                    "ess_ratio_quantum":    round(float(ess_ratio_q), 6),
                    "t_quantum_sample_s":   round(float(t_quantum_s), 6),
                    # Parallel tempering
                    "ess_pt":               round(float(ess_pt), 4),
                    "ess_ratio_pt":         round(float(ess_ratio_pt), 6),
                    "t_total_s":            round(float(t_pt_s), 4),
                    "ess_per_s_pt":         round(float(ess_per_s_pt), 4),
                    "swap_acceptance_mean": round(float(swap_acc), 4),
                    # Q/PT ratio
                    "ratio_quantum_over_pt": round(float(ratio_q_over_pt), 4),
                })

    # ── Summary ───────────────────────────────────────────────────────────────
    ratios = [r["ratio_quantum_over_pt"] for r in rows]
    summary = {
        "n_instances":                     len(rows),
        "ratio_quantum_over_pt_mean":      round(float(np.mean(ratios)), 4),
        "ratio_quantum_over_pt_median":    round(float(np.median(ratios)), 4),
        "ratio_quantum_over_pt_min":       round(float(np.min(ratios)), 4),
        "ratio_quantum_over_pt_max":       round(float(np.max(ratios)), 4),
        "mean_ess_per_s_pt":               round(float(np.mean([r["ess_per_s_pt"] for r in rows])), 4),
        "mean_swap_acceptance":            round(float(np.mean([r["swap_acceptance_mean"] for r in rows])), 4),
        "pre_registered_expected_range":   "[0.8, 1.5] (per revised_plan_v4.tex)",
    }

    out = {
        "generated_utc":               datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_replicas":                  args.n_replicas,
        "beta_schedule":               "geometric",
        "beta_min":                    args.beta_min,
        "beta_max":                    args.beta_max,
        "swap_interval_sweeps":        args.l_swap,
        "burn_in_per_replica_sweeps":  args.burn_in,
        "n_samples_per_instance":      args.n_samples,
        "scalar_observable":           "hamming_weight",
        "ess_metric":                  "ess_ratio_from_series * n_samples  (matches run_ess_sweep.py)",
        "per_instance":                rows,
        "summary":                     summary,
    }

    out_dir  = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "parallel_tempering_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[E2] Done. Wrote {out_path}")
    print(f"     Quantum/PT ratio: mean={summary['ratio_quantum_over_pt_mean']:.3f}  "
          f"median={summary['ratio_quantum_over_pt_median']:.3f}  "
          f"range=[{summary['ratio_quantum_over_pt_min']:.3f}, {summary['ratio_quantum_over_pt_max']:.3f}]")


if __name__ == "__main__":
    main()
