"""
experiments/run_parallel_tempering_baseline.py
===============================================
E2 — Parallel Tempering Baseline (Experiment E2 from revised_plan_v4.tex)

Motivation
----------
The current strongest classical baseline is tuned-block Gibbs (mean ratio
1.82×).  Reviewer concern C4 notes this is a near-tie; modern MCMC for
peaked discrete distributions typically uses tempering.  Adding a parallel-
tempering (PT) baseline tests whether the residual quantum advantage survives
against a state-of-the-art discrete sampler.

Protocol (from revised_plan_v4.tex §2.2)
-----------------------------------------
For each of the 60 instances:
  - K = 8 replicas at geometric inverse-temperature ladder
    beta_k = beta_min * (beta_max/beta_min)^{k/(K-1)},  k=0..K-1
    beta_min = 0.1, beta_max = 1.0  (target is beta = 1.0)
  - Each replica runs internal single-site Gibbs sweeps
  - Replica swap proposals every L_swap = 10 within-replica sweeps
    using standard Metropolis swap acceptance
  - Burn-in: 1000 within-replica sweeps per replica (matching other baselines)
  - Retain 3000 samples from the beta=1 replica only
  - Total wall-clock = sum over all replicas

Output
------
  experiments/results/parallel_tempering_baseline.json  (schema per plan)

Usage
-----
  python experiments/run_parallel_tempering_baseline.py
  python experiments/run_parallel_tempering_baseline.py --n_max 12
  python experiments/run_parallel_tempering_baseline.py --n_replicas 4   # faster test
"""

import json
import time
import argparse
import datetime
import os
import math
import numpy as np

# ── Reuse the same MRF builder from E1 ──────────────────────────────────────
# (copy inline so this script is standalone; users can refactor if src/ allows)

def _make_mrf_graph(family: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    nodes = list(range(n))

    if family == "chain":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif family == "barbell":
        half = n // 2
        edges = []
        for i in range(half):
            for j in range(i + 1, half):
                edges.append((i, j))
        for i in range(half, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        edges.append((half - 1, half))
    elif family == "barbell_path":
        half = n // 2
        edges = []
        for i in range(half):
            for j in range(i + 1, half):
                edges.append((i, j))
        for i in range(half, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        for k in range(half - 1, half):
            edges.append((k, k + 1))
    elif family == "erdos_renyi":
        p_edge = 0.5
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p_edge:
                    edges.append((i, j))
        if not edges:
            edges = [(0, 1)]
    elif family == "two_clique":
        half = n // 2
        edges = []
        for i in range(half):
            for j in range(i + 1, half):
                edges.append((i, j))
        for i in range(half, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        edges.append((0, half))
    else:
        raise ValueError(f"Unknown family: {family}")

    rng2 = np.random.default_rng(seed + 1000)
    edge_theta = {}
    for e in edges:
        edge_theta[e] = rng2.uniform(-5.0, 0.0, size=(2, 2))

    singleton_theta = {}
    for i in nodes:
        singleton_theta[i] = rng2.uniform(-5.0, 0.0, size=(2,))

    return n, edges, edge_theta, singleton_theta


def _log_unnorm(state: list, n: int, edges, edge_theta, singleton_theta,
                beta: float) -> float:
    lp = 0.0
    for i, s in enumerate(state):
        lp += singleton_theta[i][s]
    for e in edges:
        lp += edge_theta[e][state[e[0]], state[e[1]]]
    return beta * lp


def gibbs_sweep(state: list, n: int, edges, edge_theta, singleton_theta,
                beta: float, rng: np.random.Generator) -> list:
    """One full random-scan single-site Gibbs sweep at inverse temperature beta."""
    state = list(state)
    for i in rng.permutation(n):
        # compute conditional log-probs for x_i = 0, 1
        log_probs = []
        for val in [0, 1]:
            s = state[:]
            s[i] = val
            log_probs.append(_log_unnorm(s, n, edges, edge_theta, singleton_theta, beta))
        # numerically stable softmax → sample
        lp_max = max(log_probs)
        probs = [math.exp(lp - lp_max) for lp in log_probs]
        z = sum(probs)
        p0 = probs[0] / z
        state[i] = 0 if rng.random() < p0 else 1
    return state


def compute_ess_from_series(series: list) -> float:
    n = len(series)
    if n < 4:
        return float(n)
    arr = np.array(series, dtype=float)
    arr -= arr.mean()
    var = np.var(arr)
    if var < 1e-12:
        return float(n)
    gamma0 = np.dot(arr, arr) / n
    iat = 1.0
    for lag in range(1, n // 2):
        gamma = np.dot(arr[:-lag], arr[lag:]) / n
        rho = gamma / gamma0
        if rho <= 0.0:
            break
        iat += 2.0 * rho
    return float(n) / max(iat, 1.0)


FAMILIES = ["barbell", "barbell_path", "chain", "erdos_renyi", "two_clique"]
N_INSTANCES_PER_FAMILY = 12
N_PER_FAMILY = 10
N_SAMPLES = 3000
BURN_IN = 1000
K_REPLICAS = 8
BETA_MIN = 0.1
BETA_MAX = 1.0
L_SWAP = 10          # swap proposal every L_swap within-replica sweeps


def make_instance_list(n_max: int = 99):
    instances = []
    for family in FAMILIES:
        for inst_idx in range(N_INSTANCES_PER_FAMILY):
            seed = 42 + inst_idx * 7 + FAMILIES.index(family) * 100
            n = N_PER_FAMILY
            if n > n_max:
                continue
            instances.append({
                "instance_id": f"{family}_n{n}_seed{seed}",
                "family": family,
                "n": n,
                "seed": seed,
                "inst_idx": inst_idx,
            })
    return instances


def run_parallel_tempering(n: int, edges, edge_theta, singleton_theta,
                            n_replicas: int, beta_min: float, beta_max: float,
                            l_swap: int, burn_in: int, n_samples: int,
                            seed: int):
    """
    Run parallel tempering and return (samples_from_target, swap_acc_rate, t_total_s).
    samples_from_target: list of length n_samples from the beta=1 replica.
    """
    rng = np.random.default_rng(seed + 9999)

    # Geometric beta ladder
    if n_replicas == 1:
        betas = [beta_max]
    else:
        betas = [beta_min * (beta_max / beta_min) ** (k / (n_replicas - 1))
                 for k in range(n_replicas)]

    # Initialize replicas to random states
    replicas = [[int(rng.random() < 0.5) for _ in range(n)]
                for _ in range(n_replicas)]

    n_swap_attempts = 0
    n_swap_accepted = 0

    t_start = time.perf_counter()

    # ── Burn-in ───────────────────────────────────────────────────────────────
    for sweep in range(burn_in):
        for k in range(n_replicas):
            replicas[k] = gibbs_sweep(
                replicas[k], n, edges, edge_theta, singleton_theta, betas[k], rng)
        # Swap proposals every l_swap sweeps
        if (sweep + 1) % l_swap == 0:
            # Propose swaps between adjacent replicas (random order)
            for adj in rng.permutation(n_replicas - 1):
                k1, k2 = int(adj), int(adj) + 1
                lp1 = _log_unnorm(replicas[k1], n, edges, edge_theta,
                                   singleton_theta, betas[k1])
                lp2 = _log_unnorm(replicas[k2], n, edges, edge_theta,
                                   singleton_theta, betas[k2])
                lp1x = _log_unnorm(replicas[k1], n, edges, edge_theta,
                                    singleton_theta, betas[k2])
                lp2x = _log_unnorm(replicas[k2], n, edges, edge_theta,
                                    singleton_theta, betas[k1])
                log_acc = (lp1x + lp2x) - (lp1 + lp2)
                n_swap_attempts += 1
                if math.log(max(rng.random(), 1e-300)) < log_acc:
                    replicas[k1], replicas[k2] = replicas[k2], replicas[k1]
                    n_swap_accepted += 1

    # ── Sampling ──────────────────────────────────────────────────────────────
    target_idx = n_replicas - 1   # beta=1 is the last (hottest = 1.0)
    samples = []
    n_collected = 0
    sweep = 0
    while n_collected < n_samples:
        for k in range(n_replicas):
            replicas[k] = gibbs_sweep(
                replicas[k], n, edges, edge_theta, singleton_theta, betas[k], rng)
        sweep += 1
        if sweep % l_swap == 0:
            for adj in rng.permutation(n_replicas - 1):
                k1, k2 = int(adj), int(adj) + 1
                lp1 = _log_unnorm(replicas[k1], n, edges, edge_theta,
                                   singleton_theta, betas[k1])
                lp2 = _log_unnorm(replicas[k2], n, edges, edge_theta,
                                   singleton_theta, betas[k2])
                lp1x = _log_unnorm(replicas[k1], n, edges, edge_theta,
                                    singleton_theta, betas[k2])
                lp2x = _log_unnorm(replicas[k2], n, edges, edge_theta,
                                    singleton_theta, betas[k1])
                log_acc = (lp1x + lp2x) - (lp1 + lp2)
                n_swap_attempts += 1
                if math.log(max(rng.random(), 1e-300)) < log_acc:
                    replicas[k1], replicas[k2] = replicas[k2], replicas[k1]
                    n_swap_accepted += 1
        # record one sample per sweep from target replica
        samples.append(replicas[target_idx][0])   # scalar = x_0
        n_collected += 1

    t_total = time.perf_counter() - t_start
    swap_acc = n_swap_accepted / max(n_swap_attempts, 1)
    return samples, swap_acc, t_total


def run_e2(n_max: int = 99, n_replicas: int = K_REPLICAS):
    instances = make_instance_list(n_max)
    results = []

    # These will be filled after quantum ESS values are loaded
    # (for now we just record PT ESS; ratios need quantum ESS from ess_sweep)
    try:
        with open("paper_table_metrics.json") as f:
            ptm_data = json.load(f)
        quantum_ess_map = {
            r["instance_id"]: r["ess_quantum"]
            for r in ptm_data.get("per_instance", [])
            if "ess_quantum" in r and "instance_id" in r
        }
    except Exception:
        quantum_ess_map = {}

    print(f"Running E2 (Parallel Tempering, K={n_replicas}) on {len(instances)} instances ...")

    for inst in instances:
        iid = inst["instance_id"]
        family = inst["family"]
        n = inst["n"]
        seed = inst["seed"]
        print(f"  {iid} ...", end=" ", flush=True)

        n_, edges, edge_theta, singleton_theta = _make_mrf_graph(family, n, seed)
        samples, swap_acc, t_total = run_parallel_tempering(
            n_, edges, edge_theta, singleton_theta,
            n_replicas=n_replicas,
            beta_min=BETA_MIN, beta_max=BETA_MAX,
            l_swap=L_SWAP, burn_in=BURN_IN, n_samples=N_SAMPLES,
            seed=seed)

        ess = compute_ess_from_series(samples)
        ess_per_s = ess / t_total if t_total > 0 else float("inf")

        # Compute quantum/PT ratio if quantum ESS available for this instance
        q_ess = quantum_ess_map.get(iid, None)
        ratio = (q_ess / ess) if (q_ess is not None and ess > 0) else None

        results.append({
            "instance_id":        iid,
            "family":             family,
            "n":                  n,
            "ess":                round(ess, 4),
            "t_total_s":          round(t_total, 4),
            "ess_per_s":          round(ess_per_s, 4),
            "swap_acceptance_mean": round(swap_acc, 4),
            "ratio_quantum_over_pt": round(ratio, 4) if ratio is not None else None,
        })
        print(f"ESS={ess:.1f}  swap_acc={swap_acc:.2%}  t={t_total:.1f}s"
              + (f"  ratio={ratio:.2f}" if ratio else ""))

    # ── Summary ───────────────────────────────────────────────────────────────
    ess_vals   = [r["ess"] for r in results]
    ratio_vals = [r["ratio_quantum_over_pt"] for r in results
                  if r["ratio_quantum_over_pt"] is not None]

    summary: dict = {
        "mean_ess_per_s":     round(float(np.mean([r["ess_per_s"] for r in results])), 4),
        "mean_swap_acceptance": round(float(np.mean([r["swap_acceptance_mean"] for r in results])), 4),
    }
    if ratio_vals:
        summary.update({
            "ratio_quantum_over_pt_mean":   round(float(np.mean(ratio_vals)), 4),
            "ratio_quantum_over_pt_median": round(float(np.median(ratio_vals)), 4),
            "ratio_quantum_over_pt_min":    round(float(np.min(ratio_vals)), 4),
            "ratio_quantum_over_pt_max":    round(float(np.max(ratio_vals)), 4),
        })
    else:
        summary.update({
            "ratio_quantum_over_pt_mean":   None,
            "ratio_quantum_over_pt_median": None,
            "ratio_quantum_over_pt_min":    None,
            "ratio_quantum_over_pt_max":    None,
            "note": "Quantum ESS not found in paper_table_metrics.json; "
                    "rerun after generating quantum ESS data.",
        })

    out = {
        "generated_utc":               datetime.datetime.utcnow().isoformat(),
        "n_instances":                 len(results),
        "n_replicas":                  n_replicas,
        "beta_schedule":               "geometric",
        "beta_min":                    BETA_MIN,
        "beta_max":                    BETA_MAX,
        "swap_interval_sweeps":        L_SWAP,
        "burn_in_per_replica_sweeps":  BURN_IN,
        "samples_per_instance":        N_SAMPLES,
        "per_instance":                results,
        "summary":                     summary,
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/parallel_tempering_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[E2] Done. Wrote {out_path}")
    if ratio_vals:
        print(f"     Quantum/PT ratio: mean={summary['ratio_quantum_over_pt_mean']:.3f}  "
              f"median={summary['ratio_quantum_over_pt_median']:.3f}  "
              f"range=[{summary['ratio_quantum_over_pt_min']:.3f}, "
              f"{summary['ratio_quantum_over_pt_max']:.3f}]")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2: Parallel Tempering Baseline")
    parser.add_argument("--n_max", type=int, default=99,
                        help="Skip instances with n > n_max (for quick test)")
    parser.add_argument("--n_replicas", type=int, default=K_REPLICAS,
                        help=f"Number of PT replicas (default {K_REPLICAS})")
    args = parser.parse_args()
    run_e2(n_max=args.n_max, n_replicas=args.n_replicas)
