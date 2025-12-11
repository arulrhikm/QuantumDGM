# Utility Functions

This document covers utility functions for analysis, comparison, and visualization.

---

## Distribution Comparison

### `compute_fidelity(p, q)`

Compute fidelity between two probability distributions.

```python
from QuantumDGM import compute_fidelity

fidelity = compute_fidelity(exact_probs, quantum_probs)
# Returns value in [0, 1], where 1 = identical
```

**Formula:** F(P, Q) = (∑_x √(P(x)Q(x)))²

---

### `kl_divergence(p, q, epsilon=1e-10)`

Compute KL divergence from Q to P.

```python
from QuantumDGM import kl_divergence

kl = kl_divergence(target, approximation)
# KL(P || Q) - not symmetric!
```

> [!NOTE]
> Returns infinity if P(x) > 0 but Q(x) = 0 for any x.

---

### `hellinger_distance(p, q)`

Compute Hellinger distance.

```python
from QuantumDGM import hellinger_distance

h = hellinger_distance(p, q)  # Returns value in [0, 1]
```

**Formula:** H(P, Q) = √(1 - √F(P, Q))

---

### `total_variation_distance(p, q)`

Compute total variation distance.

```python
from QuantumDGM import total_variation_distance

tv = total_variation_distance(p, q)  # Returns value in [0, 1]
```

**Formula:** TV(P, Q) = (1/2) ∑_x |P(x) - Q(x)|

---

### `compare_distributions(p, q, labels)`

Comprehensive comparison returning all metrics.

```python
from QuantumDGM import compare_distributions, print_comparison

metrics = compare_distributions(exact_probs, quantum_probs, 
                                 labels=('Exact', 'Quantum'))

print(metrics['fidelity'])       # 0.95
print(metrics['kl_divergence'])  # 0.08
print(metrics['hellinger'])      # 0.15
print(metrics['tv_distance'])    # 0.12

# Pretty print all metrics
print_comparison(metrics)
```

---

## Sample Analysis

### `estimate_distribution(samples, n_vars)`

Convert samples to probability distribution.

```python
from QuantumDGM import estimate_distribution

# samples: array of shape (n_samples, n_vars)
dist = estimate_distribution(samples, n_vars=4)
# dist: array of shape (2^n_vars,) with empirical frequencies
```

---

### `sample_statistics(samples)`

Compute statistics about a sample set.

```python
from QuantumDGM import sample_statistics

stats = sample_statistics(samples)

print(stats['n_samples'])      # Total samples
print(stats['n_unique'])       # Unique configurations
print(stats['mode'])           # Most common configuration
print(stats['mode_count'])     # Count of mode
print(stats['entropy_estimate'])  # Empirical entropy
```

---

### `generate_state_labels(n_vars)`

Generate binary state labels for plotting.

```python
from QuantumDGM import generate_state_labels

labels = generate_state_labels(3)
# ['000', '001', '010', '011', '100', '101', '110', '111']
```

---

## Visualization Functions

> [!NOTE]
> Visualization requires `matplotlib` and `networkx`:
> ```bash
> pip install matplotlib networkx
> ```

---

### `visualize_graphical_model(model, title, figsize, save_path)`

Plot the structure of a graphical model.

```python
from QuantumDGM import visualize_graphical_model

fig = visualize_graphical_model(
    model, 
    title="4-Variable Chain Model",
    figsize=(10, 6),
    save_path="model.png"  # Optional
)
```

---

### `visualize_circuit_diagram(model, sampler, title)`

Create a simplified circuit visualization.

```python
from QuantumDGM import visualize_circuit_diagram, QCGMSampler

sampler = QCGMSampler(model)
sampler.build_circuit()

fig = visualize_circuit_diagram(
    model,
    sampler=sampler,
    title="Quantum Circuit"
)
```

---

### `plot_distribution_comparison(exact, quantum, labels, fidelity)`

Create distribution comparison bar charts.

```python
from QuantumDGM import plot_distribution_comparison, generate_state_labels, compute_fidelity

labels = generate_state_labels(model.n_vars)
fidelity = compute_fidelity(exact_probs, quantum_probs)

fig = plot_distribution_comparison(
    exact_probs,
    quantum_probs,
    labels,
    fidelity,
    save_path="comparison.png"
)
```

---

### `analyze_circuit_complexity(max_vars)`

Plot circuit complexity scaling.

```python
from QuantumDGM import analyze_circuit_complexity

fig = analyze_circuit_complexity(
    max_vars=8,
    figsize=(14, 5)
)
```

---

### `show_qiskit_circuit(model, max_gates)`

Display the actual Qiskit circuit.

```python
from QuantumDGM import show_qiskit_circuit

circuit = show_qiskit_circuit(model, max_gates=50)
# Returns the QuantumCircuit object
```

---

## Configuration Functions

### `set_default_backend(backend)`

Set global default backend for all samplers.

```python
import QuantumDGM
from qiskit_aer import AerSimulator

qcgm.set_default_backend(AerSimulator())
```

---

### `check_dependencies()` / `print_dependency_status()`

Check installed packages.

```python
import QuantumDGM

# Returns dict with version info
status = qcgm.check_dependencies()

# Pretty print
qcgm.print_dependency_status()
```

---

### `info()`

Print package information.

```python
import QuantumDGM
qcgm.info()
```

---

## See Also

- [API Reference](api_reference.md) - Complete function signatures
- [Examples Guide](examples.md) - Using utilities in demos
