# Revised Plan Traceability

This map ties each revised-plan requirement to executable scripts, artifacts, and manuscript sections.

## Tier 1

- **Generated tables/macros**
  - Script: `experiments/generate_paper_artifacts.py`
  - Artifacts:
    - `figures/paper_generated_rows.tex`
    - `figures/paper_generated_metadata.tex`
    - `figures/paper_table_metrics.json`
  - Paper sections: amplitude/fair/timing/compression tables
- **Document class**
  - File: `paper.tex` (`\\documentclass[conference,10pt]{IEEEtran}`)
- **Depth-table reproducibility**
  - Replaced with generated summary statement from artifacts

## Experiment A: Clique Entanglement Sweep

- Runner: `experiments/run_entanglement_sweep.py`
- Input helpers:
  - `experiments/graph_families.py`
  - `experiments/bq_oracle.py`
  - `src/entanglement.py`
  - `src/train.py`
- Artifacts:
  - `experiments/results/entanglement_sweep.json`
  - `figures/expA_fidelity_vs_n.png`
  - `figures/expA_deltaF_vs_density.png`
  - `figures/expA_fidelity_per_cx.png`
- Paper linkage:
  - Results section IV-B (entanglement sweep)

## Experiment B: VQC vs MPS Crossover

- Runners:
  - `experiments/run_large_n_sweep.py`
  - `experiments/run_mps_scaling.py`
- Helpers:
  - `experiments/bq_mps_baseline.py`
- Artifacts:
  - `experiments/results/large_n_comparison.json`
  - `experiments/results/mps_scaling.json`
  - `figures/expB_headline_crossover.png`
  - `figures/expB_compression_curve.png`
  - `figures/expB_mps_vs_chi.png`
- Paper linkage:
  - Results section IV-C, replacement compression table

## Experiment C: Amplitude Scaling

- Runner: `experiments/run_amplitude_scaling.py`
- Artifacts:
  - `experiments/results/amplitude_scaling.json`
  - `figures/expC_amplitude_scaling.png`
- Paper linkage:
  - Results section IV-A, amplitude table extension

## Experiment D: ESS vs Mixing Difficulty

- Runner: `experiments/run_ess_sweep.py`
- Artifact:
  - `experiments/results/ess_sweep.json`
  - `figures/expD_ess_vs_gap.png`
- Paper linkage:
  - Results section IV-D

## Experiments E/F

- Runner: `experiments/run_fidelity_decomposition.py`
- Runner: `experiments/run_marginal_matching.py`
- Artifacts:
  - `experiments/results/fidelity_decomposition.json`
  - `experiments/results/marginal_matching.json`
- Paper linkage:
  - Results section IV-E and appendix notes

## Unified Orchestration

- Orchestrator: `experiments/run_all_for_paper.py`
- Responsibilities:
  - execute experiments A-F (configurable),
  - regenerate figures,
  - regenerate paper macros/metadata,
  - emit a single run manifest.
