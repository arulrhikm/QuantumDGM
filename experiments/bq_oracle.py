from __future__ import annotations

import numpy as np

from src.amplitude_encoding import build_amplitude_circuit
from src.backends import BackendAdapter
from src.metrics import fidelity, kl_divergence, tv_distance


def get_exact_probs_bq(n: int, mrf_params: dict, graph) -> tuple[np.ndarray, str]:
    adapter = BackendAdapter()
    circ = build_amplitude_circuit(n, mrf_params, graph, with_measurements=False)
    res = adapter.statevector_probs(circ, prefer_bq=True)
    return res.probs, res.backend_used


def evaluate_trained_circuit_bq(circuit_no_measurement, target_probs: np.ndarray) -> dict:
    adapter = BackendAdapter()
    res = adapter.statevector_probs(circuit_no_measurement, prefer_bq=True)
    approx = res.probs
    return {
        "backend_used": res.backend_used,
        "requested_backend": res.requested_backend,
        "fell_back_to_aer": res.fell_back_to_aer,
        "backend_note": res.note,
        "fidelity": fidelity(approx, target_probs),
        "kl": kl_divergence(approx, target_probs),
        "tv": tv_distance(approx, target_probs),
    }

