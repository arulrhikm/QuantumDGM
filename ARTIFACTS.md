# Artifact Organization

The project is organized for paper reproducibility as follows:

- `experiments/results/` holds raw experiment outputs (`*.json`).
- `figures/` holds paper-facing assets:
  - generated plots (`*.png`)
  - `paper_generated_rows.tex`
  - `paper_generated_metadata.tex`
  - `paper_table_metrics.json`

Use:

```bash
python -m experiments.run_all_for_paper --fast
python -m experiments.generate_paper_artifacts
```

Then compile `paper.tex`, which consumes assets from `figures/`.
