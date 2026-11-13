# QuantumDGM

Code for the paper

**An End-to-End Hybrid Quantum–Classical Sampling Workflow for Discrete Markov Random Fields: A Reproducible Case Study**
Arul Rhik Mazumder, Elias Towe. IEEE Quantum Week (QCE) 2026. [arXiv:2607.09893](https://arxiv.org/abs/2607.09893)

The paper studies amplitude-encoded i.i.d. sampling from small discrete Markov random fields (MRFs), with the
target distribution computed classically, and compares it with classical MCMC (Gibbs, block Gibbs and parallel
tempering), exact inverse-CDF sampling, matrix product state (MPS) approximations and variational circuits. The
amplitude encoding follows Piatkowski and Zoufal, [arXiv:2206.00398](https://arxiv.org/abs/2206.00398).

## Setup

Python 3.10 or later.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Circuits run on Qiskit Aer by default. Some experiments prefer BlueQubit's simulators: install `bluequbit` and
set `BLUEQUBIT_API_TOKEN` to use them, otherwise they fall back to Aer. On Aer the large-n sweeps (n = 16 and
above) take hours even with `--fast`.

## Running

From the repository root:

```
python -m experiments.run_all_for_paper --fast   # reduced settings
python -m experiments.run_all_for_paper          # the paper's settings
```

This runs the experiments, writes raw results to `experiments/results/` and regenerates the figures and table
macros in `figures/`. Add `--require-bq` to fail instead of falling back to Aer. The two classical baselines
and the VQC ablation run separately:

```
python -m experiments.run_inverse_cdf_baseline
python -m experiments.run_parallel_tempering_baseline
python -m experiments.run_vqc_diagnosis_ablation
```

## Layout

| Path | Contents |
|---|---|
| `src/` | Amplitude encoding, variational circuits and entanglement layouts, training, backends and metrics |
| `experiments/graph_families.py` | The five MRF graph families and parameter sampling |
| `experiments/run_*.py` | One script per experiment |
| `experiments/generate_paper_artifacts.py` | Builds the table macros in `figures/` from the results |
| `figures/` | Recorded table metrics and generated LaTeX macros |

## Citing

```bibtex
@inproceedings{mazumder2026mrf,
  author    = {Mazumder, Arul Rhik and Towe, Elias},
  title     = {An End-to-End Hybrid Quantum--Classical Sampling Workflow for Discrete {Markov} Random Fields:
               A Reproducible Case Study},
  booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year      = {2026},
  eprint    = {2607.09893},
  archivePrefix = {arXiv}
}
```

## Licence

MIT (`LICENSE`).
