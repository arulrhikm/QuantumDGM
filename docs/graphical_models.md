# Discrete Graphical Models

This document explains the theory behind discrete graphical models as implemented in QuantumDGM.

---

## Overview

A **discrete graphical model** (DGM) defines a probability distribution over binary random variables using a graph structure. The distribution factorizes according to the graph's cliques.

---

## Mathematical Formulation

### Exponential Family Form

The probability of a configuration x is given by:

```
P_θ(X = x) = (1/Z(θ)) exp(∑_{C∈𝒞} ∑_{y∈𝒳_C} θ_{C,y} φ_{C,y}(x))
```

Where:
- **𝒞** = Set of maximal cliques in the graph
- **θ_{C,y}** = Canonical parameters for clique C and assignment y
- **φ_{C,y}(x)** = Sufficient statistics (indicator functions)
- **Z(θ)** = Partition function (normalization constant)

### Sufficient Statistics

The sufficient statistic φ_{C,y}(x) is an indicator function:

```
φ_{C,y}(x) = ∏_{v∈C} 𝟙{x_v = y_v}
```

This equals 1 if the configuration x matches assignment y on clique C, and 0 otherwise.

### Partition Function

```
Z(θ) = ∑_x exp(∑_{C∈𝒞} ∑_{y∈𝒳_C} θ_{C,y} φ_{C,y}(x))
```

Computing Z(θ) requires summing over all 2^n configurations, which is exponential in n.

---

## Hamiltonian Construction

### Diagonal Hamiltonian

The Hamiltonian H_θ encodes the model parameters:

```
H_θ = -∑_{C∈𝒞} ∑_{y∈𝒳_C} θ_{C,y} Φ_{C,y}
```

Where Φ_{C,y} are diagonal matrices with (Φ_{C,y})_{j,j} = φ_{C,y}(x_j).

### Key Property

The Hamiltonian is **diagonal**, so:
- P_θ(x_j) ∝ exp(-H_{j,j})
- Only O(2^n) storage needed (not O(4^n) for full matrix)

---

## Common Graph Structures

### Chain Model

```
v0 — v1 — v2 — v3 — v4
```

Cliques: {0,1}, {1,2}, {2,3}, {3,4}

```python
from QuantumDGM import create_chain_model
model = create_chain_model(5)
```

### Star Model

```
     v1
      |
v4 — v0 — v2
      |
     v3
```

Cliques: {0,1}, {0,2}, {0,3}, {0,4}

```python
from QuantumDGM import create_star_model
model = create_star_model(5, center=0)
```

### Ring Model

```
v0 — v1
|     |
v3 — v2
```

Cliques: {0,1}, {1,2}, {2,3}, {3,0}

```python
from QuantumDGM import DiscreteGraphicalModel
model = DiscreteGraphicalModel(4, [{0,1}, {1,2}, {2,3}, {3,0}])
```

---

## Parameters

### Shift Invariance

Due to the overcomplete parameterization, we can restrict θ_{C,y} to be negative without loss of generality. This is why `set_random_parameters()` defaults to `low=-5.0, high=0.0`.

### Effect on Distribution

- **More negative** parameters → Lower probability for matching configurations
- **Less negative** parameters → Higher probability

---

## Inference Tasks

### Computing Probabilities

```python
probs = model.compute_probabilities()
# probs[j] = P_θ(x_j) for configuration j
```

### Sampling

```python
# Exact classical sampling
samples = model.sample_exact(n_samples=1000)

# Quantum sampling
from QuantumDGM import QCGMSampler
sampler = QCGMSampler(model)
samples, rate = sampler.sample(n_samples=1000)
```

### Entropy

```python
entropy = model.compute_entropy()  # H(P_θ) in nats
```

---

## See Also

- [Quantum Circuits](quantum_circuits.md) - How circuits encode P_θ
- [API Reference](api_reference.md) - DiscreteGraphicalModel class details
- [Paper](https://arxiv.org/abs/2206.00398) - Piatkowski & Zoufal (2022)
