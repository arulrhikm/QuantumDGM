import numpy as np

# Standard Pauli matrices and Hadamard gate
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def kron(*matrices):
    """Compute Kronecker product over a variable number of matrices."""
    result = np.array([[1]], dtype=complex)
    for M in matrices:
        result = np.kron(result, M)
    return result
