from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def get_entanglement_pairs(strategy: str, n: int, graph=None) -> list[tuple[int, int]]:
    if strategy == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if strategy == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if strategy == "clique":
        if graph is None:
            raise ValueError("graph is required for clique strategy")
        nodes = sorted(graph.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        return [(idx[u], idx[v]) for u, v in graph.edges()]
    raise ValueError(f"Unknown strategy: {strategy}")


def build_variational_circuit(
    n: int,
    depth: int,
    entanglement_pairs: Iterable[tuple[int, int]],
) -> tuple[QuantumCircuit, ParameterVector]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    n_params = n * depth
    params = ParameterVector("theta", n_params)
    qc = QuantumCircuit(n)
    p_idx = 0
    for layer in range(depth):
        for q in range(n):
            qc.ry(params[p_idx], q)
            p_idx += 1
        if layer < depth - 1:
            for i, j in entanglement_pairs:
                qc.cx(i, j)
    return qc, params


def random_init(n: int, depth: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, size=n * depth)

