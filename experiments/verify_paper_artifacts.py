from __future__ import annotations

import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "experiments", "results")


def must_exist(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def main():
    required_results = [
        "entanglement_sweep.json",
        "large_n_comparison.json",
        "mps_scaling.json",
        "amplitude_scaling.json",
        "ess_sweep.json",
        "fidelity_decomposition.json",
        "marginal_matching.json",
    ]
    required_figures = [
        "expA_fidelity_vs_n.png",
        "expA_deltaF_vs_density.png",
        "expA_fidelity_per_cx.png",
        "expB_headline_crossover.png",
        "expB_compression_curve.png",
        "expB_mps_vs_chi.png",
        "expC_amplitude_scaling.png",
        "expD_ess_vs_gap.png",
        "paper_generated_rows.tex",
        "paper_generated_metadata.tex",
        "paper_table_metrics.json",
    ]

    for f in required_results:
        must_exist(os.path.join(RESULTS, f))
    for f in required_figures:
        must_exist(os.path.join(FIGS, f))

    with open(os.path.join(FIGS, "paper_generated_rows.tex"), "r", encoding="utf-8") as f:
        txt = f.read()
    for macro in [
        "\\AmplitudeTableRows",
        "\\FairTableRows",
        "\\TimingTableRows",
        "\\CompressionTableRows",
        "\\DeltaDensityTableRows",
        "\\AmplitudeScalingDetailRows",
        "\\MPSScalingRows",
    ]:
        if macro not in txt:
            raise ValueError(f"Missing macro {macro}")

    with open(os.path.join(FIGS, "paper_generated_metadata.tex"), "r", encoding="utf-8") as f:
        meta = f.read()
    for macro in [
        "\\PaperTablesGeneratedOn",
        "\\EssNInstances",
        "\\EssNFamilies",
        "\\EssMeanRatio",
        "\\EssMedianRatio",
        "\\EssMinRatio",
        "\\EssMaxRatio",
        "\\EssSlope",
        "\\EssRSquared",
    ]:
        if macro not in meta:
            raise ValueError(f"Missing metadata macro {macro}")

    with open(os.path.join(FIGS, "paper_table_metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in [
        "amplitude",
        "amplitude_detail",
        "delta_density_summary",
        "fair",
        "timing",
        "compression",
        "mps_scaling",
        "ess_summary",
        "generated_on_utc",
    ]:
        if key not in metrics:
            raise ValueError(f"Missing metrics key: {key}")

    print("Verification OK: paper artifacts and experiment outputs are present.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)

