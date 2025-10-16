"""
Quantum Circuit construction for Discrete Graphical Models (Theorem 3.4).
"""
from qiskit import QuantumCircuit
from qcgmlib.utils.circuits import hadamard_layer, rz_layer, cx_chain

def build_qcg_circuit(n, cliques, theta):
    """
    Build quantum circuit C_θ for a graphical model.
    Args:
        n (int): number of binary variables
        cliques (list[list[int]]): list of maximal cliques
        theta (list[float]): model parameters
    Returns:
        QuantumCircuit: parameterized quantum circuit
    """
    m = n + 1 + len(cliques)
    qc = QuantumCircuit(m, name="QCGM")

    hadamard_layer(qc, range(m))

    for i, C in enumerate(cliques):
        angle = float(theta[i])
        rz_layer(qc, C, angle)
        qc.barrier()

    cx_chain(qc, 0, range(n, m))

    qc.barrier()
    qc.measure_all()
    return qc
