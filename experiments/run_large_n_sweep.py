from __future__ import annotations

import argparse
import json
import os
import time

import networkx as nx
import numpy as np

from experiments.bq_mps_baseline import fidelity_from_probs, find_chi_matching_vqc_params, mps_approximate_probs
from experiments.bq_oracle import evaluate_trained_circuit_bq, get_exact_probs_bq
from experiments.graph_families import erdos_renyi, sample_mrf_params
from experiments.plotting import PALETTE, mean_std, savefig, style_ax
from src.entanglement import build_variational_circuit, get_entanglement_pairs
from src.train import train_parameter_shift


def run(n_values: list[int], seeds: int, depth: int, train_iters: int, require_bq: bool):
    records = []
    for n in n_values:
        for seed in range(seeds):
            g = erdos_renyi(n, p=0.5, seed=seed)
            params = sample_mrf_params(g, seed=seed)
            target_probs, target_backend = get_exact_probs_bq(n, params, g)

            n_vqc_params = n * depth
            pairs = get_entanglement_pairs("clique", n, g)
            circ, pvec = build_variational_circuit(n, depth, pairs)
            t0 = time.perf_counter()
            trained, loss_hist = train_parameter_shift(
                circ, pvec, target_probs, n_iters=train_iters if n < 28 else max(20, train_iters // 2), lr=0.03, seed=seed
            )
            t_vqc = time.perf_counter() - t0
            bound = circ.assign_parameters(dict(zip(pvec, trained)))
            vqc = evaluate_trained_circuit_bq(bound, target_probs)

            chi = find_chi_matching_vqc_params(n_vqc_params, n)
            t1 = time.perf_counter()
            mps = mps_approximate_probs(n, params, g, chi_max=chi, n_shots=40_000)
            t_mps = time.perf_counter() - t1
            f_mps = fidelity_from_probs(mps["probs"], target_probs)
            fell_back = (
                target_backend.startswith("aer")
                or vqc.get("fell_back_to_aer", False)
                or mps.get("fell_back_to_aer", False)
            )
            records.append(
                {
                    "n": n,
                    "seed": seed,
                    "graph": "er_05",
                    "density": float(nx.density(g)),
                    "n_vqc_params": n_vqc_params,
                    "n_mps_params": mps["n_mps_params"],
                    "chi_max": chi,
                    "F_vqc": vqc["fidelity"],
                    "F_mps": f_mps,
                    "kl_vqc": vqc["kl"],
                    "compression_vqc": float((2**n) / n_vqc_params),
                    "compression_mps": float((2**n) / mps["n_mps_params"]),
                    "target_backend": target_backend,
                    "vqc_eval_backend": vqc["backend_used"],
                    "vqc_requested_backend": vqc.get("requested_backend", ""),
                    "vqc_fell_back_to_aer": vqc.get("fell_back_to_aer", False),
                    "mps_backend": mps["backend_used"],
                    "mps_requested_backend": mps.get("requested_backend", ""),
                    "mps_fell_back_to_aer": mps.get("fell_back_to_aer", False),
                    "any_bq_fallback": fell_back,
                    "t_vqc_s": t_vqc,
                    "t_mps_s": t_mps,
                    "loss_history": loss_hist,
                }
            )
            if require_bq and fell_back:
                raise RuntimeError(f"Experiment B require-bq failed at n={n}, seed={seed}")
            print(f"n={n} seed={seed} F_vqc={vqc['fidelity']:.3f} F_mps={f_mps:.3f}")
    return records


def plot(records):
    import matplotlib.pyplot as plt

    n = np.array([r["n"] for r in records], dtype=int)
    fv = np.array([r["F_vqc"] for r in records], dtype=float)
    fm = np.array([r["F_mps"] for r in records], dtype=float)
    ux, mv, sv = mean_std(n, fv)
    _, mm, sm = mean_std(n, fm)

    plt.figure(figsize=(10, 5.5))
    plt.plot(ux, mv, marker="o", linewidth=2, color=PALETTE["vqc"], label="VQC mean")
    plt.fill_between(ux, mv - sv, mv + sv, alpha=0.18, color=PALETTE["vqc"], label="VQC ±1 std")
    plt.plot(ux, mm, marker="s", linewidth=2, color=PALETTE["mps"], label="MPS mean")
    plt.fill_between(ux, mm - sm, mm + sm, alpha=0.18, color=PALETTE["mps"], label="MPS ±1 std")
    if np.any(mv > mm):
        idx = int(np.argmax(mv > mm))
        plt.axvline(ux[idx], linestyle="--", color="gray")
        plt.text(ux[idx], 0.08, f"Estimated crossover n*={ux[idx]}", rotation=90, va="bottom", ha="right")
    delta = mv - mm
    for i, xv in enumerate(ux):
        plt.text(xv, max(mv[i], mm[i]) + 0.015, f"Δ={delta[i]:+.2f}", fontsize=8, ha="center")
    style_ax(plt.gca(), "n", "Fidelity", "Experiment B1: VQC vs MPS (matched budget)")
    plt.ylim(0.0, 1.0)
    plt.legend(ncol=2, fontsize=8)
    savefig("expB_headline_crossover.png")

    cv = np.array([r["compression_vqc"] for r in records], dtype=float)
    cm = np.array([r["compression_mps"] for r in records], dtype=float)
    ux, mcv, _ = mean_std(n, cv)
    _, mcm, _ = mean_std(n, cm)
    pv = np.array([r["n_vqc_params"] for r in records], dtype=float)
    pm = np.array([r["n_mps_params"] for r in records], dtype=float)
    _, mpv, _ = mean_std(n, pv)
    _, mpm, _ = mean_std(n, pm)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(ux, mcv, marker="o", linewidth=2, color=PALETTE["vqc"], label="VQC compression")
    ax1.plot(ux, mcm, marker="s", linewidth=2, color=PALETTE["mps"], label="MPS compression")
    ax1.set_yscale("log")
    ax2.plot(ux, mpv, marker="x", linestyle="--", color=PALETTE["vqc"], alpha=0.6, label="VQC params")
    ax2.plot(ux, mpm, marker="+", linestyle="--", color=PALETTE["mps"], alpha=0.6, label="MPS params")
    style_ax(ax1, "n", "Compression (2^n / n_params, log scale)", "Experiment B2: Compression and absolute parameter counts")
    ax2.set_ylabel("Absolute parameter count")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=2, fontsize=8)
    savefig("expB_compression_curve.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="16,20,24,28,32")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--require-bq", action="store_true", help="Fail if any BQ-preferred path falls back to Aer")
    args = parser.parse_args()

    records = run([int(x) for x in args.n_values.split(",")], args.seeds, args.depth, args.iters, args.require_bq)
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "large_n_comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    plot(records)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

