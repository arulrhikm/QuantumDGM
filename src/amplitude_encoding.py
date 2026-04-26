from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


def energy_from_pairwise_mrf(n: int, mrf_params: dict, config_bits: list[int]) -> float:
    e = 0.0
    for key, theta in mrf_params.items():
        nodes = sorted(list(key))
        prod = 1
        for v in nodes:
            prod *= int(config_bits[v])
        e += float(theta) * prod
    return float(e)


def pairwise_probs_from_params(n: int, mrf_params: dict) -> np.ndarray:
    log_probs = np.zeros(2**n, dtype=np.float64)
    for x in range(2**n):
        bits = [(x >> i) & 1 for i in range(n)]
        log_probs[x] = energy_from_pairwise_mrf(n, mrf_params, bits)
    log_probs -= float(np.max(log_probs))
    probs = np.exp(log_probs)
    probs = np.maximum(probs.astype(np.float64), 0.0)
    s = float(np.sum(probs))
    if s <= 0.0:
        out = np.zeros_like(probs)
        out[0] = 1.0
        return out
    return probs / s


def build_amplitude_circuit(n: int, mrf_params: dict, _graph=None, with_measurements: bool = False) -> QuantumCircuit:
    probs = pairwise_probs_from_params(n, mrf_params)
    # Real amplitudes on the probability simplex (helps StatePreparation numerics).
    amplitudes = np.sqrt(probs.astype(np.float64)).astype(np.complex128)
    amplitudes = np.nan_to_num(amplitudes, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(amplitudes))
    if norm <= 0.0:
        amplitudes = np.zeros(2**n, dtype=np.complex128)
        amplitudes[0] = 1.0
    else:
        amplitudes = amplitudes / norm
    qc = QuantumCircuit(n, n if with_measurements else 0)
    # Use unitary state preparation for backend compatibility (notably BQ MPS).
    qc.append(StatePreparation(amplitudes), list(range(n)))
    if with_measurements:
        qc.measure(range(n), range(n))
    return qc

