from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
FIGS = os.path.join(ROOT, "figures")


def _read_json(name: str, default):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_amplitude(rows):
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    out = []
    for n in sorted(by_n):
        vals = by_n[n]
        out.append(
            {
                "n": n,
                "states": int(2**n),
                "fidelity_mean": float(np.mean([v["fidelity"] for v in vals])),
                "time_mean_s": float(np.mean([v["time_s"] for v in vals])),
            }
        )
    return out


def _aggregate_fair(ent_rows):
    # Fair-table proxy from smallest n in Experiment A
    if not ent_rows:
        return []
    min_n = min(r["n"] for r in ent_rows)
    subset = [r for r in ent_rows if r["n"] == min_n]
    out = []
    for s in ["linear", "clique", "full"]:
        vals = [r["fidelity"] for r in subset if r["strategy"] == s]
        if vals:
            out.append({"method": f"Quantum VQC ({s})", "fidelity": float(np.mean(vals)), "notes": f"n={min_n}"})
    return out


def _aggregate_timing(b_rows):
    if not b_rows:
        return []
    return [
        {"method": "VQC train+eval", "time_s": float(np.mean([r["t_vqc_s"] for r in b_rows])), "notes": "Experiment B"},
        {"method": "MPS sampling", "time_s": float(np.mean([r["t_mps_s"] for r in b_rows])), "notes": "Experiment B"},
    ]


def _compression_table(b_rows):
    by_n = {}
    for r in b_rows:
        by_n.setdefault(r["n"], []).append(r)
    out = []
    for n in sorted(by_n):
        vals = by_n[n]
        out.append(
            {
                "n": n,
                "vqc_params": int(round(np.mean([v["n_vqc_params"] for v in vals]))),
                "mps_params": int(round(np.mean([v["n_mps_params"] for v in vals]))),
                "f_vqc": float(np.mean([v["F_vqc"] for v in vals])),
                "f_mps": float(np.mean([v["F_mps"] for v in vals])),
                "compression_vqc": float(np.mean([v["compression_vqc"] for v in vals])),
            }
        )
    return out


def _delta_density_summary(ent_rows):
    grouped: dict[tuple[str, int, int], dict] = {}
    for r in ent_rows:
        key = (str(r["graph_family"]), int(r["n"]), int(r["seed"]))
        grouped.setdefault(key, {"density": float(r["density"])})
        grouped[key][str(r["strategy"])] = float(r["fidelity"])

    xs = []
    ys = []
    for rec in grouped.values():
        if "clique" in rec and "linear" in rec:
            xs.append(float(rec["density"]))
            ys.append(float(rec["clique"] - rec["linear"]))

    if len(xs) < 2:
        return {"n_pairs": len(xs), "delta_mean": float(np.mean(ys)) if ys else 0.0, "slope": 0.0, "r2": 0.0}

    coef = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)
    yhat = coef[0] * np.array(xs, dtype=float) + coef[1]
    ss_res = float(np.sum((np.array(ys, dtype=float) - yhat) ** 2))
    ss_tot = float(np.sum((np.array(ys, dtype=float) - np.mean(ys)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return {
        "n_pairs": len(xs),
        "delta_mean": float(np.mean(ys)),
        "slope": float(coef[0]),
        "r2": float(r2),
    }


def _amplitude_detail_table(amp_rows):
    by_n = {}
    for r in amp_rows:
        by_n.setdefault(int(r["n"]), []).append(r)
    out = []
    for n in sorted(by_n):
        vals = by_n[n]
        out.append(
            {
                "n": n,
                "fidelity_mean": float(np.mean([float(v["fidelity"]) for v in vals])),
                "tv_mean": float(np.mean([float(v.get("tv", 0.0)) for v in vals])),
                "time_mean_s": float(np.mean([float(v["time_s"]) for v in vals])),
                "backend": str(vals[0].get("backend_used", "unknown")),
                "any_fallback": any(bool(v.get("fell_back_to_aer", False)) for v in vals),
            }
        )
    return out


def _mps_scaling_table(rows):
    grouped: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        key = (int(r["n"]), int(r["chi"]))
        grouped.setdefault(key, []).append(r)
    out = []
    for (n, chi), vals in sorted(grouped.items()):
        fvals = np.array([float(v["F_mps"]) for v in vals], dtype=float)
        cvals = np.array([float(v["compression"]) for v in vals], dtype=float)
        n_trials = len(vals)
        f_std = float(np.std(fvals))
        c_std = float(np.std(cvals))
        ci_mult = 1.96 / np.sqrt(max(1, n_trials))
        out.append(
            {
                "n": n,
                "chi": chi,
                "n_trials": int(n_trials),
                "f_mps_mean": float(np.mean(fvals)),
                "f_mps_std": f_std,
                "f_mps_ci95": float(ci_mult * f_std),
                "compression_mean": float(np.mean(cvals)),
                "compression_std": c_std,
                "compression_ci95": float(ci_mult * c_std),
                "backend": str(vals[0].get("backend_used", "unknown")),
                "any_fallback": any(bool(v.get("fell_back_to_aer", False)) for v in vals),
            }
        )
    return out


def _ess_summary(rows):
    if not rows:
        return {
            "n_instances": 0,
            "n_families": 0,
            "mean_ratio": 0.0,
            "median_ratio": 0.0,
            "min_ratio": 0.0,
            "max_ratio": 0.0,
            "slope": 0.0,
            "r2": 0.0,
        }

    ratio_key = "ess_ratio_single" if rows and "ess_ratio_single" in rows[0] else "ess_ratio"
    ratios = np.array([float(r[ratio_key]) for r in rows], dtype=float)
    gaps = np.array([float(r["spectral_gap_proxy"]) for r in rows], dtype=float)
    families = {str(r.get("graph_family", "unknown")) for r in rows}
    block_ratios = None
    if rows and "ess_ratio_block" in rows[0]:
        block_ratios = np.array([float(r["ess_ratio_block"]) for r in rows], dtype=float)
    eps_quantum = eps_single = eps_block = None
    if rows and "ess_per_sec_quantum" in rows[0]:
        eps_quantum = np.array([float(r["ess_per_sec_quantum"]) for r in rows], dtype=float)
        eps_single = np.array([float(r["ess_per_sec_single_gibbs"]) for r in rows], dtype=float)
        eps_block = np.array([float(r["ess_per_sec_block_gibbs"]) for r in rows], dtype=float)
    tuned_ratios = tuned_eps = None
    if rows and "ess_ratio_tuned_block" in rows[0]:
        vals = [r["ess_ratio_tuned_block"] for r in rows if r.get("ess_ratio_tuned_block") is not None]
        if vals:
            tuned_ratios = np.array([float(v) for v in vals], dtype=float)
    if rows and "ess_per_sec_tuned_block_gibbs" in rows[0]:
        vals = [r["ess_per_sec_tuned_block_gibbs"] for r in rows if r.get("ess_per_sec_tuned_block_gibbs") is not None]
        if vals:
            tuned_eps = np.array([float(v) for v in vals], dtype=float)

    slope = 0.0
    r2 = 0.0
    if len(rows) >= 2:
        coef = np.polyfit(gaps, ratios, 1)
        yhat = coef[0] * gaps + coef[1]
        ss_res = float(np.sum((ratios - yhat) ** 2))
        ss_tot = float(np.sum((ratios - np.mean(ratios)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
        slope = float(coef[0])

    return {
        "n_instances": int(len(rows)),
        "n_families": int(len(families)),
        # Backward-compatible single-baseline keys
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        # Explicit single-site vs block-gibbs keys
        "mean_ratio_single": float(np.mean(ratios)),
        "median_ratio_single": float(np.median(ratios)),
        "min_ratio_single": float(np.min(ratios)),
        "max_ratio_single": float(np.max(ratios)),
        "mean_ratio_block": float(np.mean(block_ratios)) if block_ratios is not None else None,
        "median_ratio_block": float(np.median(block_ratios)) if block_ratios is not None else None,
        "min_ratio_block": float(np.min(block_ratios)) if block_ratios is not None else None,
        "max_ratio_block": float(np.max(block_ratios)) if block_ratios is not None else None,
        "mean_ratio_tuned_block": float(np.mean(tuned_ratios)) if tuned_ratios is not None else None,
        "median_ratio_tuned_block": float(np.median(tuned_ratios)) if tuned_ratios is not None else None,
        "min_ratio_tuned_block": float(np.min(tuned_ratios)) if tuned_ratios is not None else None,
        "max_ratio_tuned_block": float(np.max(tuned_ratios)) if tuned_ratios is not None else None,
        "ess_per_sec_quantum_mean": float(np.mean(eps_quantum)) if eps_quantum is not None else None,
        "ess_per_sec_single_gibbs_mean": float(np.mean(eps_single)) if eps_single is not None else None,
        "ess_per_sec_block_gibbs_mean": float(np.mean(eps_block)) if eps_block is not None else None,
        "ess_per_sec_tuned_block_gibbs_mean": float(np.mean(tuned_eps)) if tuned_eps is not None else None,
        "slope": slope,
        "r2": float(r2),
    }


def generate():
    os.makedirs(FIGS, exist_ok=True)
    amp_raw = _read_json("amplitude_scaling.json", [])
    amp = _aggregate_amplitude(amp_raw)
    ent = _read_json("entanglement_sweep.json", [])
    b = _read_json("large_n_comparison.json", [])
    b3 = _read_json("mps_scaling.json", [])
    ess = _read_json("ess_sweep.json", [])
    fair = _aggregate_fair(ent)
    timing = _aggregate_timing(b)
    compression = _compression_table(b)
    delta_summary = _delta_density_summary(ent)
    amp_detail = _amplitude_detail_table(amp_raw)
    mps_scaling = _mps_scaling_table(b3)
    ess_summary = _ess_summary(ess)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    metrics = {
        "generated_on_utc": ts,
        "amplitude": amp,
        "amplitude_detail": amp_detail,
        "delta_density_summary": delta_summary,
        "fair": fair,
        "timing": timing,
        "compression": compression,
        "mps_scaling": mps_scaling,
        "ess_summary": ess_summary,
    }

    with open(os.path.join(FIGS, "paper_table_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    def row_amp(r):
        return f'{r["n"]} & {r["states"]} & {r["fidelity_mean"]:.3f} & $O(1)$ & {r["n"]} & {r["time_mean_s"]:.2f} \\\\'

    def row_fair(r):
        return f'{r["method"]} & {r["fidelity"]:.3f} & {r["notes"]} \\\\'

    def row_time(r):
        return f'{r["method"]} & {r["time_s"]:.2f} & {r["notes"]} \\\\'

    def row_comp(r):
        return (
            f'{r["n"]} & {r["vqc_params"]} & {r["mps_params"]} & '
            f'{r["f_vqc"]:.3f} & {r["f_mps"]:.3f} & {r["compression_vqc"]:.1f} \\\\'
        )

    def row_delta_summary(r):
        return (
            f'{r["n_pairs"]} & {r["delta_mean"]:.4f} & {r["slope"]:.4f} & {r["r2"]:.4f} \\\\'
        )

    def row_amp_detail(r):
        fb = "yes" if r["any_fallback"] else "no"
        backend = str(r["backend"]).replace("_", "\\_")
        return (
            f'{r["n"]} & {r["fidelity_mean"]:.6f} & {r["tv_mean"]:.6f} & '
            f'{r["time_mean_s"]:.2f} & {backend} & {fb} \\\\'
        )

    def row_mps_scaling(r):
        fb = "yes" if r["any_fallback"] else "no"
        backend = str(r["backend"]).replace("_", "\\_")
        return (
            f'{r["n"]} & {r["chi"]} & {r["f_mps_mean"]:.4f} $\\pm$ {r["f_mps_ci95"]:.4f} & '
            f'{r["compression_mean"]:.2f} $\\pm$ {r["compression_ci95"]:.2f} & {r["n_trials"]} & {backend} & {fb} \\\\'
        )

    with open(os.path.join(FIGS, "paper_generated_rows.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated. Do not edit.\n")
        f.write("\\newcommand{\\AmplitudeTableRows}{%\n" + "\n".join(row_amp(r) for r in amp) + "\n}\n\n")
        f.write("\\newcommand{\\FairTableRows}{%\n" + "\n".join(row_fair(r) for r in fair) + "\n}\n\n")
        f.write("\\newcommand{\\TimingTableRows}{%\n" + "\n".join(row_time(r) for r in timing) + "\n}\n\n")
        f.write("\\newcommand{\\CompressionTableRows}{%\n" + "\n".join(row_comp(r) for r in compression) + "\n}\n\n")
        f.write("\\newcommand{\\DeltaDensityTableRows}{%\n" + row_delta_summary(delta_summary) + "\n}\n\n")
        f.write("\\newcommand{\\AmplitudeScalingDetailRows}{%\n" + "\n".join(row_amp_detail(r) for r in amp_detail) + "\n}\n\n")
        f.write("\\newcommand{\\MPSScalingRows}{%\n" + "\n".join(row_mps_scaling(r) for r in mps_scaling) + "\n}\n")

    with open(os.path.join(FIGS, "paper_generated_metadata.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated. Do not edit.\n")
        f.write(f"\\newcommand{{\\PaperTablesGeneratedOn}}{{{ts} UTC}}\n")
        f.write(f"\\newcommand{{\\EssNInstances}}{{{ess_summary['n_instances']}}}\n")
        f.write(f"\\newcommand{{\\EssNFamilies}}{{{ess_summary['n_families']}}}\n")
        f.write(f"\\newcommand{{\\EssMeanRatio}}{{{ess_summary['mean_ratio']:.2f}}}\n")
        f.write(f"\\newcommand{{\\EssMedianRatio}}{{{ess_summary['median_ratio']:.2f}}}\n")
        f.write(f"\\newcommand{{\\EssMinRatio}}{{{ess_summary['min_ratio']:.2f}}}\n")
        f.write(f"\\newcommand{{\\EssMaxRatio}}{{{ess_summary['max_ratio']:.2f}}}\n")
        f.write(f"\\newcommand{{\\EssSlope}}{{{ess_summary['slope']:.2f}}}\n")
        f.write(f"\\newcommand{{\\EssRSquared}}{{{ess_summary['r2']:.3f}}}\n")

    print("Generated paper artifacts in figures/")


if __name__ == "__main__":
    generate()

