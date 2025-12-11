# Examples Guide

This document describes the example scripts and demos included with QuantumDGM.

---

## Quick Reference

| File | Purpose | Run Time |
|------|---------|----------|
| `demo_script.py` | Basic functionality | ~10 sec |
| `quantum_vs_classical_demo.py` | Benchmarking comparison | ~30 sec |
| `variational_training_demo.py` | Large model training | ~1 min |
| `demo_notebook.ipynb` | Interactive tutorial | varies |

---

## Demo Scripts

### `demo_script.py` - Basic Functionality

Demonstrates core library features:
- Creating graphical models
- Quantum sampling
- Distribution comparison

```bash
cd examples
python demo_script.py
```

**Output:**
- Model structure visualization
- Sample distribution vs exact
- Fidelity metrics

---

### `quantum_vs_classical_demo.py` - Honest Comparison

Research-focused comparison between quantum and classical sampling methods.

```bash
python quantum_vs_classical_demo.py
```

**Topics Covered:**
1. **Fair Comparison**: When both methods know P(x)
2. **Quantum Properties**: Independence, no burn-in
3. **Unfair Comparison Explained**: Why Gibbs comparison is misleading
4. **When Quantum Matters**: Real use cases

---

### `variational_training_demo.py` - Large Models

Demonstrates variational circuit training for n > 10 variables.

```bash
python variational_training_demo.py
```

**Output:**
- Training progress with loss/fidelity
- Parameter reduction statistics
- Final circuit performance

---

### `demo_notebook.ipynb` - Interactive Tutorial

Jupyter notebook with step-by-step explanations.

```bash
jupyter notebook demo_notebook.ipynb
```

**Sections:**
1. Introduction to graphical models
2. Building quantum circuits
3. Sampling and analysis
4. Visualization examples

---

## Test Suite

Located in `examples/tests/`:

### `test_variational_training.py`

Validates variational training implementation.

```bash
python examples/tests/test_variational_training.py
```

**Tests (7 total):**
- Circuit construction
- Parameterization
- Optimization convergence
- Fidelity improvement
- Entanglement strategies
- Edge cases

---

### `test_optimizations.py`

Tests performance optimizations.

```bash
python examples/tests/test_optimizations.py
```

**Tests:**
- Sparse Hamiltonian correctness
- Caching effectiveness
- Memory usage

---

### Running All Tests

```bash
# From project root
python examples/tests/test_variational_training.py
python examples/tests/test_optimizations.py
```

**Expected Output:**
```
✅ All tests passing (14/14)
✅ No warnings
✅ Production ready
```

---

## Example Code Snippets

### Basic Quantum Sampling

```python
from qcgm import DiscreteGraphicalModel, QCGMSampler

# Create model
model = DiscreteGraphicalModel(4, [{0,1}, {1,2}, {2,3}])
model.set_random_parameters()

# Sample
sampler = QCGMSampler(model)
samples, rate = sampler.sample(n_samples=1000)

print(f"Success rate: {rate:.2%}")
```

---

### Compare with Exact Distribution

```python
from qcgm import estimate_distribution, compute_fidelity

exact = model.compute_probabilities()
empirical = estimate_distribution(samples, model.n_vars)

fidelity = compute_fidelity(exact, empirical)
print(f"Fidelity: {fidelity:.4f}")
```

---

### Variational Training

```python
from qcgm import ApproximateCircuitBuilder

model = DiscreteGraphicalModel(12, [{i, i+1} for i in range(11)])
model.set_random_parameters()

builder = ApproximateCircuitBuilder(depth=3)
circuit, params, info = builder.build_circuit_with_target(
    model,
    n_optimization_steps=50,
    verbose=True
)
```

---

### Visualization

```python
from qcgm import visualize_graphical_model, plot_distribution_comparison

# Model structure
fig1 = visualize_graphical_model(model)

# Distribution comparison  
fig2 = plot_distribution_comparison(
    exact, empirical, 
    ['000', '001', '010', '011', '100', '101', '110', '111'],
    fidelity
)
```

---

## Figures Directory

Example outputs are saved to `examples/figures/`:
- `model_structure.png` - Graph visualization
- `distribution_comparison.png` - Bar chart comparison
- `training_progress.png` - Optimization curve

---

## See Also

- [Getting Started](getting_started.md) - Installation guide
- [API Reference](api_reference.md) - Function documentation
- [IMPLEMENTATION_SUMMARY.md](../examples/tests/IMPLEMENTATION_SUMMARY.md) - Test details
