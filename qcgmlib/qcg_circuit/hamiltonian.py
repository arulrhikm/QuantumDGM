"""
Construct the Hamiltonian H_θ = -∑_{C,y} θ_{C,y} Φ_{C,y}.
"""
import numpy as np
from qcgmlib.models.pauli_markov import compute_pauli_markov_statistic

def hamiltonian(theta_dict, n):
    """
    theta_dict: dict mapping (tuple(C), tuple(y)) -> float
    n: number of variables/qubits
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for (C, y), theta_val in theta_dict.items():
        Φ_C_y = compute_pauli_markov_statistic(list(C), list(y), n)
        H -= theta_val * Φ_C_y
    return H
