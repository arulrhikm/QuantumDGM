from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from experiments.graph_families import family_registry, sample_mrf_params
from experiments.plotting import savefig, style_ax
from src.amplitude_encoding import pairwise_probs_from_params
from src.metrics import ess_ratio_from_series


def gibbs_sample_from_exact(probs: np.ndarray, n: int, n_samples: int = 2000, burn_in: int = 1000):
    states = np.arange(2**n)
    state = int(np.random.randint(2**n))
    out = []
    for t in range(n_samples + burn_in):
        i = int(np.random.randint(n))
        s0 = state & ~(1 << i)
        s1 = s0 | (1 << i)
        p1 = float(probs[s1])
        p0 = float(probs[s0])
        q1 = p1 / (p0 + p1 + 1e-12)
        state = s1 if float(np.random.random()) < q1 else s0
        if t >= burn_in:
            out.append(state)
    return np.array(out, dtype=int)


def block_gibbs_sample_from_exact(
    probs: np.ndarray, graph, n: int, n_samples: int = 2000, burn_in: int = 1000
):
    import networkx as nx

    cliques = [tuple(sorted(c)) for c in nx.find_cliques(graph)]
    state = int(np.random.randint(2**n))
    out = []
    all_states = np.arange(2**n, dtype=int)
    for t in range(n_samples + burn_in):
        block = cliques[int(np.random.randint(len(cliques)))]
        block_set = set(block)
        candidates = []
        weights = []
        for s in all_states:
            ok = True
            for bit in range(n):
                if bit not in block_set:
                    if ((s >> bit) & 1) != ((state >> bit) & 1):
                        ok = False
                        break
            if ok:
                candidates.append(int(s))
                weights.append(float(probs[s]))
        w = np.array(weights, dtype=float)
        s = float(w.sum())
        if s <= 0.0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / s
        # Enforce exact unit-sum for numpy.choice robustness.
        w = np.clip(w, 0.0, None)
        s2 = float(w.sum())
        if s2 <= 0.0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / s2
        state = int(np.random.choice(np.array(candidates, dtype=int), p=w))
        if t >= burn_in:
            out.append(state)
    return np.array(out, dtype=int)


def block_gibbs_tuned_sample_from_exact(
    probs: np.ndarray,
    graph,
    n: int,
    n_samples: int = 2000,
    burn_in: int = 1000,
    top_k_cliques: int = 2,
):
    """
    Stronger baseline than random-clique block Gibbs:
    cycles deterministically through the largest cliques.
    """
    import networkx as nx

    all_cliques = [tuple(sorted(c)) for c in nx.find_cliques(graph)]
    if not all_cliques:
        return gibbs_sample_from_exact(probs, n, n_samples=n_samples, burn_in=burn_in)

    all_cliques = sorted(all_cliques, key=lambda c: len(c), reverse=True)
    cliques = all_cliques[: max(1, min(top_k_cliques, len(all_cliques)))]

    state = int(np.random.randint(2**n))
    out = []
    all_states = np.arange(2**n, dtype=int)
    for t in range(n_samples + burn_in):
        block = cliques[t % len(cliques)]
        block_set = set(block)
        candidates = []
        weights = []
        for s in all_states:
            ok = True
            for bit in range(n):
                if bit not in block_set and ((s >> bit) & 1) != ((state >> bit) & 1):
                    ok = False
                    break
            if ok:
                candidates.append(int(s))
                weights.append(float(probs[s]))
        w = np.array(weights, dtype=float)
        ssum = float(w.sum())
        if ssum <= 0.0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / ssum
        w = np.clip(w, 0.0, None)
        ssum2 = float(w.sum())
        w = (w / ssum2) if ssum2 > 0.0 else (np.ones_like(w) / float(len(w)))
        state = int(np.random.choice(np.array(candidates, dtype=int), p=w))
        if t >= burn_in:
            out.append(state)
    return np.array(out, dtype=int)


def spectral_gap_proxy(probs: np.ndarray) -> float:
    p = np.sort(probs)[::-1]
    if len(p) < 2:
        return 1.0
    return float(max(1e-8, p[0] - p[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="8,10,12")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--families", default="chain,erdos_renyi,two_clique,barbell,barbell_path")
    parser.add_argument("--tuned-n-values", default="10,12")
    parser.add_argument("--target-ess", type=float, default=1000.0)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]
    tuned_n_values = {int(x) for x in args.tuned_n_values.split(",") if x.strip()}
    target_ess = float(args.target_ess)
    families = family_registry()
    fam_order = [x.strip() for x in args.families.split(",") if x.strip()]
    rows = []
    for fam_name in fam_order:
        if fam_name not in families:
            raise ValueError(f"Unknown family: {fam_name}")
        for n in n_values:
            for seed in range(args.seeds):
                g = families[fam_name](n, seed)
                params = sample_mrf_params(g, seed=seed)
                probs = pairwise_probs_from_params(n, params)
                # Quantum i.i.d series from exact distribution
                t0 = time.perf_counter()
                q_idx = np.random.choice(2**n, size=3000, p=probs)
                q_series = np.array([bin(x).count("1") for x in q_idx], dtype=float)
                t_quantum_s = time.perf_counter() - t0

                t1 = time.perf_counter()
                g_idx = gibbs_sample_from_exact(probs, n, n_samples=3000, burn_in=1000)
                g_series = np.array([bin(x).count("1") for x in g_idx], dtype=float)
                t_single_gibbs_s = time.perf_counter() - t1

                t2 = time.perf_counter()
                b_idx = block_gibbs_sample_from_exact(probs, g, n, n_samples=3000, burn_in=1000)
                b_series = np.array([bin(x).count("1") for x in b_idx], dtype=float)
                t_block_gibbs_s = time.perf_counter() - t2

                ess_q = ess_ratio_from_series(q_series)
                ess_g = ess_ratio_from_series(g_series)
                ess_b = ess_ratio_from_series(b_series)

                ess_t = None
                t_tuned_gibbs_s = None
                eps_t = None
                ratio_tuned = None
                if n in tuned_n_values:
                    t3 = time.perf_counter()
                    t_idx = block_gibbs_tuned_sample_from_exact(probs, g, n, n_samples=3000, burn_in=1000)
                    t_series = np.array([bin(x).count("1") for x in t_idx], dtype=float)
                    t_tuned_gibbs_s = time.perf_counter() - t3
                    ess_t = ess_ratio_from_series(t_series)
                    eps_t = float(ess_t / max(1e-8, t_tuned_gibbs_s))
                    ratio_tuned = float(ess_q / max(1e-8, ess_t))

                gap = spectral_gap_proxy(probs)
                rows.append(
                    {
                        "graph_family": fam_name,
                        "n": n,
                        "seed": seed,
                        "ess_quantum": ess_q,
                        "ess_gibbs_single": ess_g,
                        "ess_gibbs_block": ess_b,
                        "ess_ratio_single": float(ess_q / max(1e-8, ess_g)),
                        "ess_ratio_block": float(ess_q / max(1e-8, ess_b)),
                        "ess_ratio_tuned_block": ratio_tuned,
                        "t_quantum_s": float(t_quantum_s),
                        "t_single_gibbs_s": float(t_single_gibbs_s),
                        "t_block_gibbs_s": float(t_block_gibbs_s),
                        "t_tuned_gibbs_s": float(t_tuned_gibbs_s) if t_tuned_gibbs_s is not None else None,
                        "ess_per_sec_quantum": float(ess_q / max(1e-8, t_quantum_s)),
                        "ess_per_sec_single_gibbs": float(ess_g / max(1e-8, t_single_gibbs_s)),
                        "ess_per_sec_block_gibbs": float(ess_b / max(1e-8, t_block_gibbs_s)),
                        "ess_per_sec_tuned_block_gibbs": eps_t,
                        "ess_gibbs_tuned_block": float(ess_t) if ess_t is not None else None,
                        "time_to_target_ess_quantum_s": float(target_ess / max(1e-8, (ess_q / max(1e-8, t_quantum_s)))),
                        "time_to_target_ess_single_gibbs_s": float(
                            target_ess / max(1e-8, (ess_g / max(1e-8, t_single_gibbs_s)))
                        ),
                        "time_to_target_ess_block_gibbs_s": float(
                            target_ess / max(1e-8, (ess_b / max(1e-8, t_block_gibbs_s)))
                        ),
                        "time_to_target_ess_tuned_block_gibbs_s": float(target_ess / max(1e-8, eps_t))
                        if eps_t is not None
                        else None,
                        "spectral_gap_proxy": gap,
                    }
                )

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ess_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    import matplotlib.pyplot as plt

    fams = sorted(set(r["graph_family"] for r in rows))
    markers = {"barbell": "s", "chain": "o", "erdos_renyi": "^", "two_clique": "D", "barbell_path": "P"}
    plt.figure(figsize=(8.5, 5.5))
    all_x = []
    all_y = []
    for fam in fams:
        xf = np.array([r["spectral_gap_proxy"] for r in rows if r["graph_family"] == fam], dtype=float)
        yf = np.array([r["ess_ratio_single"] for r in rows if r["graph_family"] == fam], dtype=float)
        all_x.extend(list(xf))
        all_y.extend(list(yf))
        plt.scatter(xf, yf, alpha=0.75, marker=markers.get(fam, "o"), label=fam)
    if len(all_x) > 1:
        xlog = np.log10(np.array(all_x))
        yarr = np.array(all_y)
        coef = np.polyfit(xlog, yarr, 1)
        xx = np.linspace(min(xlog), max(xlog), 120)
        yy = coef[0] * xx + coef[1]
        plt.plot(10**xx, yy, "--", color="#333333", linewidth=2, label="global log-fit")
        plt.text(0.02, 0.96, f"slope={coef[0]:.2f}", transform=plt.gca().transAxes, va="top", fontsize=9)
    plt.xscale("log")
    style_ax(
        plt.gca(),
        "Spectral gap proxy (log scale)",
        "ESS ratio (Quantum / Single-site Gibbs)",
        "Experiment D: ESS advantage vs mixing difficulty",
    )
    plt.legend(fontsize=8, ncol=2)
    savefig("expD_ess_vs_gap.png")

    plt.figure(figsize=(8.5, 5.5))
    single = np.array([r["ess_ratio_single"] for r in rows], dtype=float)
    block = np.array([r["ess_ratio_block"] for r in rows], dtype=float)
    tuned_vals = np.array(
        [float(r["ess_ratio_tuned_block"]) for r in rows if r.get("ess_ratio_tuned_block") is not None],
        dtype=float,
    )
    data = [single, block]
    labels = ["Q/Single-site", "Q/Block"]
    if len(tuned_vals) > 0:
        data.append(tuned_vals)
        labels.append("Q/Tuned-block")
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    style_ax(plt.gca(), "Comparator baseline", "ESS ratio (Quantum / baseline)", "Experiment D: baseline-strength comparison")
    savefig("expD_ess_baseline_comparison.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

