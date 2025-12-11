# Getting Started

This guide covers installation, setup, and basic usage of QuantumDGM.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantumDGM.git
cd QuantumDGM

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Core runtime |
| numpy | ≥ 1.20.0 | Numerical computations |
| qiskit | ≥ 0.39.0 | Quantum circuit framework |
| qiskit-aer | ≥ 0.11.0 | Quantum simulator |
| scipy | ≥ 1.7.0 | Optimization |
| matplotlib | ≥ 3.3.0 | Visualization (optional) |
| networkx | ≥ 2.5 | Graph visualization (optional) |

### Verify Installation

```python
import qcgm
qcgm.print_dependency_status()
```

---

## Quick Start

### Basic Sampling (n ≤ 10 variables)

```python
from qcgm import DiscreteGraphicalModel, QCGMSampler

# Create a graphical model (chain structure)
model = DiscreteGraphicalModel(n_vars=4, cliques=[{0,1}, {1,2}, {2,3}])
model.set_random_parameters(low=-2.0, high=-0.5)

# Sample using quantum circuit
sampler = QCGMSampler(model)
samples, success_rate = sampler.sample(n_samples=1000)

print(f"Generated {len(samples)} samples")
print(f"Success rate: {success_rate:.4f}")
```

### Large Models (n > 10) - Variational Training

```python
from qcgm import DiscreteGraphicalModel, ApproximateCircuitBuilder

# Create a larger model
model = DiscreteGraphicalModel(12, [{i, i+1} for i in range(11)])
model.set_random_parameters()

# Train variational circuit
builder = ApproximateCircuitBuilder(depth=3, entanglement='linear')
circuit, params, info = builder.build_circuit_with_target(
    model,
    n_optimization_steps=100,
    verbose=True
)

print(f"Final fidelity: {info['final_fidelity']:.4f}")
```

### Convenience Functions

```python
from qcgm import create_chain_model, create_star_model

# Quick model creation
chain = create_chain_model(5, low=-2.0, high=-0.5)
star = create_star_model(5, center=0)
```

---

## Common Workflows

### 1. Compare Quantum vs Exact Sampling

```python
from qcgm import DiscreteGraphicalModel, QCGMSampler
from qcgm import estimate_distribution, compute_fidelity

model = DiscreteGraphicalModel(4, [{0,1}, {1,2}, {2,3}])
model.set_random_parameters()

# Get exact probabilities
exact_probs = model.compute_probabilities()

# Get quantum samples
sampler = QCGMSampler(model)
samples, _ = sampler.sample(n_samples=5000)
quantum_probs = estimate_distribution(samples, model.n_vars)

# Compare
fidelity = compute_fidelity(exact_probs, quantum_probs)
print(f"Fidelity: {fidelity:.4f}")
```

### 2. Visualize Model Structure

```python
from qcgm import DiscreteGraphicalModel, visualize_graphical_model

model = DiscreteGraphicalModel(5, [{0,1}, {1,2}, {2,3}, {3,4}, {0,4}])
model.set_random_parameters()

fig = visualize_graphical_model(model, title="5-Variable Ring Model")
fig.savefig("model_structure.png")
```

---

## Next Steps

- [Architecture](architecture.md) - Understand the codebase structure
- [API Reference](api_reference.md) - Detailed class and function documentation
- [Examples Guide](examples.md) - Run demonstration scripts
