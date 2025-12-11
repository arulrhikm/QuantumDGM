# Quantum Circuits

This document covers quantum circuit construction for discrete graphical models.

---

## Overview

QuantumDGM uses quantum circuits to prepare states whose measurement outcomes are distributed according to P_θ(x). Two approaches are available:

| Approach | Model Size | Method |
|----------|------------|--------|
| **Exact** | n ≤ 10 | Amplitude encoding |
| **Approximate** | n > 10 | Variational training |

---

## Exact Circuit (Amplitude Encoding)

### Principle

Given target probabilities P_θ(x), prepare state |ψ⟩ such that:

```
|⟨x|ψ⟩|² = P_θ(x)
```

The amplitudes are set to √P_θ(x) for each basis state |x⟩.

### Usage

```python
from qcgm import QuantumCircuitBuilder, DiscreteGraphicalModel

model = DiscreteGraphicalModel(4, [{0,1}, {1,2}, {2,3}])
model.set_random_parameters()

circuit = QuantumCircuitBuilder.build_circuit(model)
```

### Qubit Ordering

> [!IMPORTANT]
> Qiskit uses **little-endian** bit ordering. QuantumDGM handles the reordering internally.

- Model state (v0, v1, v2) = (0, 1, 1)
- Qiskit measures bitstring "110" (reversed)
- The library converts back automatically

### Limitations

For n > 10, amplitudes become exponentially small and circuit compilation becomes expensive.

---

## Variational Circuits

### Why Approximate?

Exact amplitude encoding scales as O(2^n), making it impractical for large models. Variational circuits use O(n × depth) parameters.

### Hardware-Efficient Ansatz

Each layer consists of:
1. **Rotation layer**: Ry and Rz gates on each qubit
2. **Entanglement layer**: CX gates connecting qubits

```
Layer 1:    [Ry--Rz]--[Ry--Rz]--[Ry--Rz]--[Ry--Rz]
                 ├──────CX──────┤
                           ├────────CX────────┤

Layer 2:    [Ry--Rz]--[Ry--Rz]--[Ry--Rz]--[Ry--Rz]
                 ├──────CX──────┤
                           ├────────CX────────┤
```

### Usage

```python
from qcgm import ApproximateCircuitBuilder, DiscreteGraphicalModel

model = DiscreteGraphicalModel(12, [{i, i+1} for i in range(11)])
model.set_random_parameters()

builder = ApproximateCircuitBuilder(depth=3, entanglement='clique')
circuit, params, info = builder.build_circuit_with_target(
    model,
    n_optimization_steps=100,
    loss='kl',
    verbose=True
)

print(f"Final fidelity: {info['final_fidelity']:.4f}")
```

---

## Entanglement Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `'clique'` | Entangle based on model cliques | Matches graph structure |
| `'linear'` | Sequential: q0-q1, q1-q2, ... | Chain models |
| `'full'` | All-to-all entanglement | Dense graphs (expensive) |

### Clique-Based Entanglement

For model with cliques {0,1}, {1,2}, the CX gates connect:
- Qubit 0 ↔ Qubit 1
- Qubit 1 ↔ Qubit 2

```python
builder = ApproximateCircuitBuilder(depth=3, entanglement='clique')
```

---

## Training Parameters

### Loss Functions

| Loss | Formula | Properties |
|------|---------|------------|
| `'kl'` | KL(P_target \|\| P_circuit) | Standard divergence |
| `'fidelity'` | 1 - F(P_target, P_circuit) | Symmetric similarity |
| `'l2'` | \|\|P_target - P_circuit\|\|² | L2 distance |

### Optimization

- **Optimizer**: COBYLA (gradient-free)
- **Steps**: 50-200 typical (more = better but slower)
- **Depth**: 2-5 layers (more = more expressive)

### Expected Fidelity

| Model Complexity | Fidelity Range |
|------------------|----------------|
| Simple (chain) | 0.85 - 0.99 |
| Moderate | 0.65 - 0.85 |
| Complex (dense) | 0.45 - 0.77 |

---

## Automatic Method Selection

```python
from qcgm import smart_circuit_builder

circuit, info = smart_circuit_builder(
    model,
    optimize_approx=True,  # Train variational circuit if n > 10
    verbose=True
)

if info['method'] == 'exact':
    print("Used amplitude encoding")
else:
    print(f"Used variational with fidelity {info['fidelity']:.4f}")
```

---

## Performance Tips

### Depth Selection

```python
from qcgm import ApproximateCircuitBuilder

# Use heuristic
depth = ApproximateCircuitBuilder.estimate_optimal_depth(model)
builder = ApproximateCircuitBuilder(depth=depth)
```

### Parameter Reduction

| n (vars) | Exact Params | Variational (d=3) | Reduction |
|----------|--------------|-------------------|-----------|
| 10 | 1,024 | 60 | 17× |
| 15 | 32,768 | 90 | 364× |
| 20 | 1,048,576 | 120 | 8,738× |

---

## See Also

- [API Reference](api_reference.md) - Circuit builder classes
- [Graphical Models](graphical_models.md) - Theory background
- [Examples Guide](examples.md) - Demo scripts
