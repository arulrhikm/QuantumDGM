# API Reference

Complete documentation for all public classes and functions in QuantumDGM.

---

## Core Classes

### `DiscreteGraphicalModel`

Represents a discrete graphical model over binary variables.

```python
from QuantumDGM import DiscreteGraphicalModel

model = DiscreteGraphicalModel(n_vars=4, cliques=[{0,1}, {1,2}, {2,3}])
```

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_vars` | `int` | Number of binary variables |
| `cliques` | `List[Set[int]]` | List of maximal cliques (sets of variable indices) |

#### Methods

| Method | Description |
|--------|-------------|
| `set_parameters(theta_dict)` | Set parameters from dictionary |
| `set_random_parameters(low, high, seed)` | Initialize with random parameters |
| `compute_probabilities()` | Compute exact probability distribution |
| `compute_hamiltonian_diagonal()` | Get sparse Hamiltonian diagonal |
| `compute_partition_function()` | Compute Z(θ) |
| `compute_entropy()` | Compute H(P_θ) |
| `sample_exact(n_samples)` | Generate ground-truth samples |

#### Example

```python
model = DiscreteGraphicalModel(3, [{0,1}, {1,2}])
model.set_random_parameters(low=-2.0, high=-0.5, seed=42)
probs = model.compute_probabilities()
print(f"Distribution: {probs}")
```

---

### `QuantumCircuitBuilder`

Builds exact quantum circuits via amplitude encoding (for n ≤ 10).

```python
from QuantumDGM import QuantumCircuitBuilder

circuit = QuantumCircuitBuilder.build_circuit(model)
```

#### Static Methods

| Method | Description |
|--------|-------------|
| `build_circuit(model)` | Build circuit using direct initialization |
| `build_circuit_simplified(model)` | Alias for direct initialization |
| `circuit_depth_estimate(model)` | Estimate circuit depth |
| `required_qubits(model)` | Get required qubit count |

---

### `ApproximateCircuitBuilder`

Variational circuit builder for large models (n > 10).

```python
from QuantumDGM import ApproximateCircuitBuilder

builder = ApproximateCircuitBuilder(depth=3, entanglement='clique')
circuit, params, info = builder.build_circuit_with_target(model)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | `int` | 3 | Number of variational layers |
| `entanglement` | `str` | `'clique'` | Strategy: `'clique'`, `'linear'`, or `'full'` |

#### Methods

| Method | Description |
|--------|-------------|
| `build_variational_circuit(model, initial_params, seed)` | Create parameterized circuit |
| `build_circuit_with_target(model, ...)` | Build and optimize circuit |
| `estimate_optimal_depth(model)` | Suggest depth based on structure |

#### `build_circuit_with_target` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `DiscreteGraphicalModel` | required | Target model |
| `n_optimization_steps` | `int` | 100 | Optimizer iterations |
| `loss` | `str` | `'kl'` | Loss function: `'kl'`, `'fidelity'`, `'l2'` |
| `verbose` | `bool` | False | Print progress |

#### Returns

| Name | Type | Description |
|------|------|-------------|
| `circuit` | `QuantumCircuit` | Optimized circuit |
| `params` | `np.ndarray` | Trained parameters |
| `info` | `dict` | Training info including `final_fidelity` |

---

### `QCGMSampler`

Main interface for quantum sampling.

```python
from QuantumDGM import QCGMSampler

sampler = QCGMSampler(model)
samples, success_rate = sampler.sample(n_samples=1000)
```

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `DiscreteGraphicalModel` | Graphical model to sample from |

#### Methods

| Method | Description |
|--------|-------------|
| `sample(n_samples, backend, simplified)` | Draw samples from circuit |
| `sample_with_retry(target_samples, max_shots)` | Sample until target reached |
| `build_circuit(simplified)` | Build internal circuit |
| `get_circuit_stats()` | Get circuit statistics |

---

### `smart_circuit_builder`

Automatically selects exact or approximate method based on model size.

```python
from QuantumDGM import smart_circuit_builder

circuit, info = smart_circuit_builder(model, optimize_approx=True, verbose=True)
print(f"Method used: {info['method']}")  # 'exact' or 'approximate'
```

---

## Convenience Functions

### Model Creation

```python
from QuantumDGM import create_chain_model, create_star_model, create_complete_model, create_tree_model

# Chain: v0 - v1 - v2 - v3 - v4
chain = create_chain_model(5, low=-2.0, high=-0.5)

# Star: center connected to all
star = create_star_model(5, center=0)

# Complete: all pairs connected
complete = create_complete_model(4)

# Tree: custom edges
edges = [(0,1), (0,2), (1,3), (1,4)]
tree = create_tree_model(edges)
```

---

## Utility Functions

### Distribution Comparison

```python
from QuantumDGM import compute_fidelity, kl_divergence, hellinger_distance, total_variation_distance

f = compute_fidelity(p, q)      # Fidelity F(P,Q) ∈ [0,1]
kl = kl_divergence(p, q)        # KL(P || Q)
h = hellinger_distance(p, q)    # Hellinger H(P,Q) ∈ [0,1]
tv = total_variation_distance(p, q)  # TV(P,Q) ∈ [0,1]
```

### Sample Analysis

```python
from QuantumDGM import estimate_distribution, compare_distributions, sample_statistics

# Convert samples to distribution
dist = estimate_distribution(samples, n_vars=4)

# Comprehensive comparison
metrics = compare_distributions(exact_probs, quantum_probs)
print(metrics['fidelity'], metrics['kl_divergence'])

# Sample statistics
stats = sample_statistics(samples)
```

### Other Utilities

```python
from QuantumDGM import generate_state_labels, print_comparison

labels = generate_state_labels(3)  # ['000', '001', ..., '111']
print_comparison(metrics)          # Pretty-print comparison
```

---

## Visualization Functions

Available when matplotlib/networkx are installed.

```python
from QuantumDGM import (
    visualize_graphical_model,
    visualize_circuit_diagram,
    plot_distribution_comparison,
    analyze_circuit_complexity,
    show_qiskit_circuit
)
```

| Function | Description |
|----------|-------------|
| `visualize_graphical_model(model)` | Plot model graph structure |
| `visualize_circuit_diagram(model, sampler)` | Simplified circuit view |
| `plot_distribution_comparison(exact, quantum, labels, fidelity)` | Distribution bar charts |
| `analyze_circuit_complexity(max_vars)` | Complexity scaling plots |
| `show_qiskit_circuit(model)` | Display Qiskit circuit |

---

## Configuration

```python
import QuantumDGM

# Set default backend
from qiskit_aer import AerSimulator
QuantumDGM.set_default_backend(AerSimulator())

# Check dependencies
QuantumDGM.print_dependency_status()

# Package info
QuantumDGM.info()
```
