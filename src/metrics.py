from __future__ import annotations

import numpy as np


def fidelity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(np.sqrt(np.clip(p, 0.0, 1.0) * np.clip(q, 0.0, 1.0))) ** 2)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum((p + epsilon) * np.log((p + epsilon) / (q + epsilon))))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(p - q)))


def estimate_probs_from_counts(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    probs = np.zeros(2**n_qubits, dtype=np.float64)
    total = max(1, int(sum(counts.values())))
    for bitstr, c in counts.items():
        idx = int(bitstr[::-1], 2)  # qiskit little-endian -> model index
        probs[idx] += c / total
    return probs


def ess_ratio_from_series(x: np.ndarray, max_lag: int = 100) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return 1.0
    max_lag = min(max_lag, len(x) - 1)
    y = x - float(np.mean(x))
    denom = float(np.dot(y, y))
    if denom <= 0:
        return 1.0
    s = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(y[:-lag], y[lag:]) / denom)
        if rho <= 0:
            break
        s += rho
    tau = 1.0 + 2.0 * s
    return float(min(1.0, 1.0 / tau))

