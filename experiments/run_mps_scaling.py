from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python experiments/run_mps_scaling.py` or running from inside experiments/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from experiments.bq_mps_baseline import fidelity_from_counts, mps_approximate_probs
from experiments.graph_families import erdos_renyi, sample_mrf_params
from experiments.plotting import savefig, style_ax


def plot(rows):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    for n in sorted(set(r["n"] for r in rows)):
        filt = [r for r in rows if r["n"] == n]
        xs = sorted(set(r["chi"] for r in filt))
        y_mean = []
        y_std = []
        c_mean = []
        for x in xs:
            vals = [r["F_mps"] for r in filt if r["chi"] == x]
            comps = [r["compression"] for r in filt if r["chi"] == x]
            y_mean.append(float(np.mean(vals)))
            y_std.append(float(np.std(vals)))
            c_mean.append(float(np.mean(comps)))
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)
        ax1.plot(xs, y_mean, marker="o", linewidth=2, label=f"Fidelity n={n}")
        ax1.fill_between(xs, y_mean - y_std, y_mean + y_std, alpha=0.15)
        ax2.plot(xs, c_mean, linestyle="--", alpha=0.55, label=f"Compression n={n}")
    ax1.set_xscale("log")
    style_ax(ax1, "chi_max (log scale)", "MPS Fidelity", "Experiment B3: MPS fidelity and compression vs chi")
    ax2.set_ylabel("Compression ratio")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2)
    savefig("expB_mps_vs_chi.png")


def main():
    parser = argparse.ArgumentParser()
    # Budget-aware defaults: keep B3 runnable in regular sessions.
    parser.add_argument("--profile", choices=["quick", "balanced", "full"], default="balanced")
    parser.add_argument("--n-values", default=None)
    parser.add_argument("--chi-values", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=None, help="Shots for each evaluated chi point")
    parser.add_argument(
        "--reference-shots",
        type=int,
        default=None,
        help="Shots for high-chi reference used in count-based fidelity",
    )
    parser.add_argument("--require-bq", action="store_true", help="Fail if any BQ-preferred path falls back to Aer")
    parser.add_argument("--no-resume", action="store_true", help="Disable default resume behavior and rerun all requested trials")
    args = parser.parse_args()

    presets = {
        "quick": {
            "n_values": [24, 32],
            "chi_values": [8, 16],
            "seeds": 1,
            "n_shots": 15_000,
            "reference_shots": 20_000,
        },
        "balanced": {
            "n_values": [24, 32, 40],
            "chi_values": [8, 16, 32],
            "seeds": 1,
            "n_shots": 20_000,
            "reference_shots": 30_000,
        },
        "full": {
            "n_values": [32, 40, 48, 56, 64],
            "chi_values": [8, 16, 32, 64, 128],
            "seeds": 2,
            "n_shots": 50_000,
            "reference_shots": 50_000,
        },
    }
    cfg = presets[args.profile]

    n_values = [int(x) for x in args.n_values.split(",")] if args.n_values else cfg["n_values"]
    chi_values = [int(x) for x in args.chi_values.split(",")] if args.chi_values else cfg["chi_values"]
    seeds = int(args.seeds) if args.seeds is not None else int(cfg["seeds"])
    n_shots = int(args.n_shots) if args.n_shots is not None else int(cfg["n_shots"])
    ref_shots = int(args.reference_shots) if args.reference_shots is not None else int(cfg["reference_shots"])

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "mps_scaling.json")

    existing_rows: list[dict] = []
    resume = not args.no_resume
    if resume and os.path.exists(out):
        with open(out, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                existing_rows = loaded

    done_keys = {
        (int(r.get("n", -1)), int(r.get("chi", -1)), int(r.get("seed", -1)))
        for r in existing_rows
    }

    rows = list(existing_rows)
    skipped = 0

    def _persist_checkpoint() -> None:
        rows.sort(key=lambda r: (int(r.get("n", 0)), int(r.get("chi", 0)), int(r.get("seed", 0))))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    print(
        f"[B3] profile={args.profile} n={n_values} chi={chi_values} "
        f"seeds={seeds} shots={n_shots} ref_shots={ref_shots} resume={resume}"
    )

    for n in n_values:
        chi_ref = max(chi_values)
        for chi in chi_values:
            for seed in range(seeds):
                key = (n, chi, seed)
                if key in done_keys:
                    skipped += 1
                    print(f"skip n={n} chi={chi} seed={seed} (already present)")
                    continue
                g = erdos_renyi(n, 0.5, seed=seed)
                params = sample_mrf_params(g, seed=seed)
                mps = mps_approximate_probs(n, params, g, chi_max=chi, n_shots=n_shots, dense_output=False)
                ref = mps_approximate_probs(n, params, g, chi_max=chi_ref, n_shots=ref_shots, dense_output=False)
                if mps["counts"] is None or ref["counts"] is None:
                    raise RuntimeError("Expected count dictionaries for sparse MPS fidelity computation.")
                rows.append(
                    {
                        "n": n,
                        "chi": chi,
                        "seed": seed,
                        "F_mps": fidelity_from_counts(mps["counts"], ref["counts"]),
                        "n_params": mps["n_mps_params"],
                        "compression": float((2**n) / mps["n_mps_params"]),
                        "backend_used": mps["backend_used"],
                        "requested_backend": mps.get("requested_backend", ""),
                        "fell_back_to_aer": mps.get("fell_back_to_aer", False),
                        "target_backend": ref["backend_used"],
                        "reference_chi": chi_ref,
                    }
                )
                # Checkpoint after every completed trial so resume survives interrupts/network failures.
                _persist_checkpoint()
                if args.require_bq and (mps.get("fell_back_to_aer", False) or ref.get("fell_back_to_aer", False)):
                    raise RuntimeError(f"Experiment B3 require-bq failed at n={n}, chi={chi}, seed={seed}")
                print(f"n={n} chi={chi} seed={seed} shots={n_shots}/{ref_shots}")

    _persist_checkpoint()

    plot(rows)
    print(f"Wrote {out} ({len(rows)} rows, skipped {skipped} existing trials)")


if __name__ == "__main__":
    main()

