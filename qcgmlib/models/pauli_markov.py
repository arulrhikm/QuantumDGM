"""
Algorithm 1 — Compute Pauli–Markov sufficient statistics Φ_{C,y}.
"""
from qcgmlib.utils.pauli import I, Z, kron

def compute_pauli_markov_statistic(C, y, n):
    """
    Compute Φ_{C,y} as a symbolic Kronecker product of Pauli operators.
    Args:
        C (list[int]): Indices of clique vertices.
        y (list[int]): Binary assignment for clique vertices.
        n (int): Total number of variables (qubits).
    Returns:
        numpy.ndarray: The Φ_{C,y} operator (2^n x 2^n matrix).
    """
    Φ = [[1]]
    for v in range(n):
        if v not in C:
            Φ = kron(Φ, I)
        else:
            idx = C.index(v)
            Φ = kron(Φ, (I - Z) / 2 if y[idx] == 1 else (I + Z) / 2)
    return Φ
