from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from src.amplitude_encoding import build_amplitude_circuit
from src.backends import BackendAdapter
from src.metrics import fidelity


def mps_param_count(n: int, chi_max: int) -> int:
    return int(2 * chi_max**2 * max(1, n - 2) + 4 * chi_max)


def find_chi_matching_vqc_params(n_vqc_params: int, n: int, chi_cap: int = 512) -> int:
    for chi in range(1, chi_cap + 1):
        if mps_param_count(n, chi) >= n_vqc_params:
            return chi
    return chi_cap


def _build_scalable_mrf_proxy_circuit(n: int, mrf_params: dict, graph, with_measurements: bool) -> QuantumCircuit:
    """
    Build a scalable circuit surrogate for large-n MRFs.
    Avoids dense 2^n amplitude construction while still reflecting node/edge structure.
    """
    qc = QuantumCircuit(n, n if with_measurements else 0)

    # Local fields -> RY biases
    for u in range(n):
        theta_u = float(mrf_params.get(frozenset([u]), 0.0))
        qc.ry(2.0 * np.tanh(theta_u), u)

    # Pairwise terms -> ZZ-style entanglers
    for u, v in graph.edges():
        theta_uv = float(mrf_params.get(frozenset([u, v]), 0.0))
        qc.cx(u, v)
        qc.rz(2.0 * np.tanh(theta_uv), v)
        qc.cx(u, v)

    if with_measurements:
        qc.measure(range(n), range(n))
    return qc


def mps_approximate_probs(
    n: int,
    mrf_params: dict,
    graph,
    chi_max: int,
    n_shots: int = 100_000,
    dense_output: bool = True,
) -> dict:
    adapter = BackendAdapter()
    # Dense amplitude prep scales as O(2^n) memory; switch to scalable proxy for large n.
    if n <= 20:
        circ = build_amplitude_circuit(n, mrf_params, graph, with_measurements=True)
    else:
        circ = _build_scalable_mrf_proxy_circuit(n, mrf_params, graph, with_measurements=True)
    result = adapter.sample_probs(circ, shots=n_shots, prefer_mps=True, chi_max=chi_max, dense_output=dense_output)
    return {
        "probs": result.probs,
        "counts": result.counts,
        "backend_used": result.backend_used,
        "requested_backend": result.requested_backend,
        "fell_back_to_aer": result.fell_back_to_aer,
        "backend_note": result.note,
        "n_mps_params": mps_param_count(n, chi_max),
    }


def fidelity_from_probs(approx_probs: np.ndarray, target_probs: np.ndarray) -> float:
    return fidelity(approx_probs, target_probs)


def fidelity_from_counts(approx_counts: dict[str, int], target_counts: dict[str, int]) -> float:
    n_a = max(1, int(sum(approx_counts.values())))
    n_b = max(1, int(sum(target_counts.values())))
    keys = set(approx_counts.keys()) | set(target_counts.keys())
    overlap = 0.0
    for k in keys:
        pa = float(approx_counts.get(k, 0)) / n_a
        pb = float(target_counts.get(k, 0)) / n_b
        overlap += float(np.sqrt(pa * pb))
    return float(overlap**2)

