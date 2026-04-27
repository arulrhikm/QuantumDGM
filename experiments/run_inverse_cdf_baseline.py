"""
experiments/run_inverse_cdf_baseline.py
=======================================
E1 — Exact Inverse-CDF Baseline (Experiment E1 from revised_plan_v4.tex)

Motivation
----------
The amplitude-encoding pipeline already computes the full diagonal of H_theta
in O(2^n) time.  Once that array is in memory, the apples-to-apples classical
baseline is exact inverse-CDF sampling — both methods amortize the SAME
preprocessing cost and both produce i.i.d. samples with tau ≈ 1.

Protocol (from revised_plan_v4.tex §2.1)
-----------------------------------------
For each of the 60 ESS instances (5 families × 12 instances):
  1. Rebuild P_theta from the same seed/graph used in the ESS sweep.
  2. Build the cumulative array F(j) = sum_{i<=j} P_theta(x_i).  [O(2^n), timed]
  3. Draw 3000 i.i.d. samples via binary search on F.  [O(n) per sample, timed]
  4. Compute ESS from the scalar sample series using the same IAT estimator
     used everywhere else in the codebase.
  5. Record t_preprocess_s, t_sample_s, t_total_s, ess,
     ess_per_s_sample_only, ess_per_s_amortized.

Output
------
  experiments/results/inverse_cdf_baseline.json  (schema per plan)

Usage
-----
  python experiments/run_inverse_cdf_baseline.py
  python experiments/run_inverse_cdf_baseline.py --n_max 12  # smaller test
"""

import json
import time
import bisect
import argparse
import datetime
import math
import os
import numpy as np

# ── MRF / graph helpers (reuse existing codebase utilities) ──────────────────
# We try to import from src/.  If not available, we inline the minimal logic.
try:
    from src.mrf import build_mrf, sample_exact   # type: ignore
    _HAS_SRC = True
except ImportError:
    _HAS_SRC = False

# ── ESS estimator (autocorrelation time, same as rest of codebase) ──────────
def compute_ess_from_series(series: list) -> float:
    """
    Estimate ESS via integrated autocorrelation time.
    For i.i.d. samples, IAT = 1 and ESS = len(series).
    """
    n = len(series)
    if n < 4:
        return float(n)
    arr = np.array(series, dtype=float)
    arr -= arr.mean()
    var = np.var(arr)
    if var < 1e-12:
        return float(n)
    # sum of normalized auto-covariances, truncated at first negative lag
    gamma0 = np.dot(arr, arr) / n
    iat = 1.0
    for lag in range(1, n // 2):
        gamma = np.dot(arr[:-lag], arr[lag:]) / n
        rho = gamma / gamma0
        if rho <= 0.0:
            break
        iat += 2.0 * rho
    return float(n) / max(iat, 1.0)

# ── Minimal MRF builder (inline fallback if src/ not importable) ─────────────
def _make_mrf_graph(family: str, n: int, seed: int):
    """
    Reconstruct the same MRF instance used in the ESS sweep.
    Returns (edges, theta) where theta is a dict (edge) -> (2,2) array.
    Mirrors the MRF construction in the existing experiment runners.
    """
    rng = np.random.default_rng(seed)
    nodes = list(range(n))

    if family == "chain":
        edges = [(i, i + 1) for i in range(n - 1)]

    elif family == "barbell":
        # two cliques of size n//2 connected by a bridge
        half = n // 2
        edges = []
        for i in range(half):
            for j in range(i + 1, half):
                edges.append((i, j))
        for i in range(half, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        edges.append((half - 1, half))   # bridge

    elif family == "barbell_path":
        half = n // 2
        edges = []
        for i in range(half):
            for j in range(i + 1, half):
                edges.append((i, j))
        for i in range(half, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        # path connector
        for k in range(half - 1, half):
            edges.append((k, k + 1))

    elif family == "erdos_renyi":
        p = 0.5
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
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
        edges.append((0, half))          # single bridge

    else:
        raise ValueError(f"Unknown family: {family}")

    # Edge potentials: theta_{ij}(x_i, x_j) ~ U(-5, 0) per plan
    rng2 = np.random.default_rng(seed + 1000)
    theta = {}
    for e in edges:
        theta[e] = rng2.uniform(-5.0, 0.0, size=(2, 2))

    # Also singleton potentials (theta_i(x_i) ~ U(-5,0))
    singleton = {}
    for i in nodes:
        singleton[i] = rng2.uniform(-5.0, 0.0, size=(2,))

    return n, edges, theta, singleton


def _compute_log_prob_table(n: int, edges, edge_theta, singleton_theta) -> np.ndarray:
    """
    Enumerate all 2^n configurations and compute unnormalized log-probabilities.
    Returns normalized probability array P of shape (2^n,).
    """
    N = 2**n
    log_p = np.zeros(N)
    for state_int in range(N):
        bits = [(state_int >> i) & 1 for i in range(n)]
        lp = 0.0
        for i, s in enumerate(bits):
            lp += singleton_theta[i][s]
        for (i, j), th in zip(edges, [edge_theta[e] for e in edges]):
            lp += th[bits[i], bits[j]]
        log_p[state_int] = lp
    # normalize
    log_p -= np.max(log_p)
    p = np.exp(log_p)
    p /= p.sum()
    return p


def exact_icdf_sample(cdf: np.ndarray, n_samples: int) -> list:
    """Draw n_samples from the distribution defined by CDF via binary search."""
    u = np.random.default_rng().random(n_samples)
    samples = []
    for ui in u:
        idx = bisect.bisect_right(cdf, ui)
        idx = min(idx, len(cdf) - 1)
        samples.append(idx)
    return samples


# ── Instance specification: 5 families × 12 instances ───────────────────────
# Seeds and sizes matching the ESS sweep in experiments/results/ess_sweep.json
FAMILIES = ["barbell", "barbell_path", "chain", "erdos_renyi", "two_clique"]
N_INSTANCES_PER_FAMILY = 12
N_PER_FAMILY = 10           # qubit count for each instance
N_SAMPLES = 3000


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


def run_e1(n_max: int = 99):
    instances = make_instance_list(n_max)
    results = []

    print(f"Running E1 (Exact Inverse-CDF Baseline) on {len(instances)} instances ...")

    for inst in instances:
        iid = inst["instance_id"]
        family = inst["family"]
        n = inst["n"]
        seed = inst["seed"]
        print(f"  {iid} ...", end=" ", flush=True)

        # ── Build P_theta ─────────────────────────────────────────────────
        n_, edges, edge_theta, singleton_theta = _make_mrf_graph(family, n, seed)
        p = _compute_log_prob_table(n_, edges, edge_theta, singleton_theta)

        # ── E1 step 2: build CDF  (timed) ─────────────────────────────────
        t0 = time.perf_counter()
        cdf = np.cumsum(p)
        cdf[-1] = 1.0                # ensure exactly 1.0 at end
        t_preprocess = time.perf_counter() - t0

        # ── E1 step 3: draw 3000 i.i.d. samples  (timed) ──────────────────
        t1 = time.perf_counter()
        sample_indices = exact_icdf_sample(cdf, N_SAMPLES)
        t_sample = time.perf_counter() - t1
        t_total = t_preprocess + t_sample

        # ── E1 step 4: compute ESS from scalar (x_0 marginal) series ──────
        scalar_series = [float((idx >> 0) & 1) for idx in sample_indices]
        ess = compute_ess_from_series(scalar_series)

        ess_per_s_sample_only = ess / t_sample if t_sample > 0 else float("inf")
        ess_per_s_amortized   = ess / t_total  if t_total  > 0 else float("inf")

        results.append({
            "instance_id":            iid,
            "family":                 family,
            "n":                      n,
            "t_preprocess_s":         round(t_preprocess, 6),
            "t_sample_s":             round(t_sample, 6),
            "t_total_s":              round(t_total, 6),
            "ess":                    round(ess, 4),
            "ess_per_s_sample_only":  round(ess_per_s_sample_only, 4),
            "ess_per_s_amortized":    round(ess_per_s_amortized, 4),
        })
        print(f"ESS={ess:.1f}  t_sample={t_sample:.3f}s  t_pre={t_preprocess:.4f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_ess       = float(np.mean([r["ess"] for r in results]))
    median_ess     = float(np.median([r["ess"] for r in results]))
    mean_samp_only = float(np.mean([r["ess_per_s_sample_only"] for r in results]))
    mean_amort     = float(np.mean([r["ess_per_s_amortized"] for r in results]))

    out = {
        "generated_utc":           datetime.datetime.utcnow().isoformat(),
        "n_instances":             len(results),
        "families":                FAMILIES,
        "burn_in":                 0,
        "samples_per_instance":    N_SAMPLES,
        "per_instance":            results,
        "summary": {
            "mean_ess":                    round(mean_ess, 4),
            "median_ess":                  round(median_ess, 4),
            "mean_ess_per_s_sample_only":  round(mean_samp_only, 4),
            "mean_ess_per_s_amortized":    round(mean_amort, 4),
        },
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/inverse_cdf_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[E1] Done. Wrote {out_path}")
    print(f"     mean ESS={mean_ess:.1f}  mean ESS/s (sample-only)={mean_samp_only:.1f}"
          f"  mean ESS/s (amortized)={mean_amort:.1f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1: Exact Inverse-CDF Baseline")
    parser.add_argument("--n_max", type=int, default=99,
                        help="Skip instances with n > n_max (for quick test)")
    args = parser.parse_args()
    run_e1(n_max=args.n_max)
