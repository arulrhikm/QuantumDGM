from __future__ import annotations

import argparse
import json
import os
import time

import networkx as nx
import numpy as np

from experiments.bq_oracle import evaluate_trained_circuit_bq, get_exact_probs_bq
from experiments.graph_families import family_registry, sample_mrf_params
from experiments.plotting import PALETTE, mean_std, savefig, style_ax
from src.entanglement import build_variational_circuit, get_entanglement_pairs
from src.train import train_parameter_shift


def run_one(graph_name: str, n: int, seed: int, depth: int, n_iters: int):
    families = family_registry()
    g = families[graph_name](n, seed)
    if g.number_of_nodes() != n:
        # Some synthetic families can return a graph with a different node count for small n.
        return []
    params = sample_mrf_params(g, seed=seed)
    target_probs, target_backend = get_exact_probs_bq(n, params, g)

    rows = []
    for strategy in ["linear", "clique", "full"]:
        pairs = get_entanglement_pairs(strategy, n, g)
        circuit, pvec = build_variational_circuit(n, depth, pairs)
        t0 = time.perf_counter()
        trained, history = train_parameter_shift(
            circuit, pvec, target_probs, n_iters=n_iters, lr=0.04, seed=seed, prefer_bq_eval=False
        )
        elapsed = time.perf_counter() - t0
        bound = circuit.assign_parameters(dict(zip(pvec, trained)))
        metrics = evaluate_trained_circuit_bq(bound, target_probs)
        rows.append(
            {
                "graph_family": graph_name,
                "n": n,
                "seed": seed,
                "strategy": strategy,
                "depth": depth,
                "density": float(nx.density(g)),
                "n_cx": len(pairs) * max(0, depth - 1),
                "target_backend": target_backend,
                "eval_backend": metrics["backend_used"],
                "fidelity": metrics["fidelity"],
                "kl": metrics["kl"],
                "tv": metrics["tv"],
                "train_seconds": elapsed,
                "loss_history": history,
            }
        )
    return rows


def make_plots(records: list[dict]):
    import matplotlib.pyplot as plt

    # A1: strategy curves with 95% CI and sample counts
    plt.figure(figsize=(10, 5.5))
    for strategy in ["linear", "clique", "full"]:
        xs = np.array([r["n"] for r in records if r["strategy"] == strategy], dtype=int)
        ys = np.array([r["fidelity"] for r in records if r["strategy"] == strategy], dtype=float)
        ux, m, s = mean_std(xs, ys)
        n_rep = np.array([np.sum(xs == u) for u in ux], dtype=float)
        ci95 = 1.96 * s / np.sqrt(np.maximum(n_rep, 1.0))
        color = PALETTE[strategy]
        plt.plot(ux, m, marker="o", linewidth=2, color=color, label=f"{strategy} (mean)")
        plt.fill_between(ux, m - ci95, m + ci95, alpha=0.18, color=color, label=f"{strategy} 95% CI")
        for x_u, y_u in zip(ux, m):
            plt.text(x_u, y_u + 0.01, f"{y_u:.2f}", fontsize=8, ha="center")
    style_ax(plt.gca(), "n", "Fidelity", "Experiment A1: Fidelity vs n (with 95% CI)")
    plt.ylim(0.0, 1.0)
    plt.legend(ncol=2, fontsize=8)
    savefig("expA_fidelity_vs_n.png")

    # A2: delta-fidelity vs density with fit stats
    dens = {}
    for r in records:
        key = (r["graph_family"], r["n"], r["seed"])
        dens.setdefault(key, {"density": r["density"]})
        dens[key][r["strategy"]] = r["fidelity"]
    x = []
    y = []
    for _, d in dens.items():
        if "clique" in d and "linear" in d:
            x.append(d["density"])
            y.append(d["clique"] - d["linear"])
    plt.figure(figsize=(8.5, 5.5))
    plt.scatter(x, y, alpha=0.75, s=35, edgecolor="black", linewidth=0.3)
    r2_text = "R^2 = n/a"
    if len(x) > 1:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(min(x), max(x), 80)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy, "--", linewidth=2, color="#444444")
        yhat = coef[0] * np.array(x) + coef[1]
        ss_res = float(np.sum((np.array(y) - yhat) ** 2))
        ss_tot = float(np.sum((np.array(y) - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
        r2_text = f"R^2 = {r2:.3f}, slope = {coef[0]:.3f}"
    plt.axhline(0.0, linestyle=":", color="gray")
    style_ax(plt.gca(), "Graph density", "Delta Fidelity (clique - linear)", "Experiment A2: Topology benefit vs density")
    plt.text(0.02, 0.95, r2_text, transform=plt.gca().transAxes, fontsize=9, va="top")
    savefig("expA_deltaF_vs_density.png")

    # A3: fidelity efficiency with CX cost overlay
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    for strategy in ["linear", "clique", "full"]:
        xs = np.array([r["n"] for r in records if r["strategy"] == strategy], dtype=int)
        eff = np.array(
            [r["fidelity"] / max(1, r["n_cx"]) for r in records if r["strategy"] == strategy], dtype=float
        )
        cx = np.array([r["n_cx"] for r in records if r["strategy"] == strategy], dtype=float)
        ux, m_eff, _ = mean_std(xs, eff)
        _, m_cx, _ = mean_std(xs, cx)
        color = PALETTE[strategy]
        ax1.plot(ux, m_eff, marker="o", linewidth=2, color=color, label=f"{strategy}: F/CX")
        ax2.plot(ux, m_cx, marker="x", linestyle="--", color=color, alpha=0.6, label=f"{strategy}: CX")
    style_ax(ax1, "n", "Fidelity per CX", "Experiment A3: Information efficiency vs entanglement cost")
    ax2.set_ylabel("Average CX gates")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=2, fontsize=8)
    savefig("expA_fidelity_per_cx.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="10,12,14")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--require-bq", action="store_true", help="Fail if any BQ-preferred path falls back to Aer")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]
    graph_names = ["chain", "erdos_renyi", "two_clique", "barbell"]
    records = []
    n_fallbacks = 0
    for graph_name in graph_names:
        for n in n_values:
            for seed in range(args.seeds):
                rows = run_one(graph_name, n, seed, args.depth, args.iters)
                for r in rows:
                    if r.get("target_backend", "").startswith("aer") or r.get("eval_backend", "").startswith("aer"):
                        n_fallbacks += 1
                records.extend(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "entanglement_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    make_plots(records)
    if n_fallbacks > 0:
        print(f"[ExperimentA] WARNING: {n_fallbacks} records used Aer fallback on BQ-preferred paths.")
        if args.require_bq:
            raise RuntimeError("Experiment A require-bq failed due to Aer fallback.")
    print(f"Wrote {out_path} with {len(records)} rows")


if __name__ == "__main__":
    main()

