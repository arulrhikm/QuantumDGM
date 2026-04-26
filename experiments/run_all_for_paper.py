from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")


def _run(cmd: list[str]):
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run reduced settings for quick verification")
    parser.add_argument("--require-bq", action="store_true", help="Fail if BQ-preferred paths fall back to Aer")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    bq_flag = ["--require-bq"] if args.require_bq else []
    if args.fast:
        _run(["python", "-m", "experiments.run_entanglement_sweep", "--n-values", "8,10", "--seeds", "1", "--iters", "10", *bq_flag])
        _run(["python", "-m", "experiments.run_large_n_sweep", "--n-values", "16,20", "--seeds", "1", "--iters", "10", *bq_flag])
        _run(["python", "-m", "experiments.run_mps_scaling", "--n-values", "32,40", "--chi-values", "8,16", "--seeds", "1", *bq_flag])
        _run(["python", "-m", "experiments.run_amplitude_scaling", "--n-start", "16", "--n-stop", "20", "--seeds", "1", *bq_flag])
        _run(["python", "-m", "experiments.run_ess_sweep", "--n-values", "8,10", "--seeds", "1"])
    else:
        _run(["python", "-m", "experiments.run_entanglement_sweep", *bq_flag])
        _run(["python", "-m", "experiments.run_large_n_sweep", *bq_flag])
        _run(["python", "-m", "experiments.run_mps_scaling", *bq_flag])
        _run(["python", "-m", "experiments.run_amplitude_scaling", *bq_flag])
        _run(["python", "-m", "experiments.run_ess_sweep"])

    _run(["python", "-m", "experiments.run_fidelity_decomposition"])
    _run(["python", "-m", "experiments.run_marginal_matching"])
    _run(["python", "-m", "experiments.generate_paper_artifacts"])

    manifest = {
        "generated_on_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "fast" if args.fast else "full",
        "require_bq": args.require_bq,
        "outputs_dir": "experiments/results",
        "artifact_generator": "experiments/generate_paper_artifacts.py",
    }
    with open(os.path.join(RESULTS, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote experiments/results/run_manifest.json")


if __name__ == "__main__":
    main()

