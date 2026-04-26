from __future__ import annotations

import numpy as np

from src.backends import BackendAdapter
from src.metrics import kl_divergence


def kl_loss(circuit, param_vec, param_values, target_probs, backend: BackendAdapter, prefer_bq: bool) -> float:
    bound = circuit.assign_parameters(dict(zip(param_vec, param_values)))
    probs = backend.statevector_probs(bound, prefer_bq=prefer_bq).probs
    return kl_divergence(probs, target_probs)


def train_parameter_shift(
    circuit,
    param_vec,
    target_probs,
    n_iters: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    prefer_bq_eval: bool = False,
) -> tuple[np.ndarray, list[float]]:
    backend = BackendAdapter()
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.1, 0.1, len(param_vec))
    history: list[float] = []
    shift = np.pi / 2.0
    for _ in range(n_iters):
        loss = kl_loss(circuit, param_vec, params, target_probs, backend, prefer_bq_eval)
        history.append(float(loss))
        grads = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += shift
            p_minus[i] -= shift
            l_plus = kl_loss(circuit, param_vec, p_plus, target_probs, backend, prefer_bq_eval)
            l_minus = kl_loss(circuit, param_vec, p_minus, target_probs, backend, prefer_bq_eval)
            grads[i] = 0.5 * (l_plus - l_minus)
        params -= lr * grads
    return params, history

