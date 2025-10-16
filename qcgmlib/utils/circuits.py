from qiskit import QuantumCircuit

def hadamard_layer(qc: QuantumCircuit, qubits):
    """Apply Hadamard gates to a list of qubits."""
    for q in qubits:
        qc.h(q)
    return qc

def rz_layer(qc: QuantumCircuit, qubits, angle):
    """Apply RZ rotation to specified qubits."""
    for q in qubits:
        qc.rz(angle, q)
    return qc

def cx_chain(qc: QuantumCircuit, control, targets):
    """Apply a chain of CX gates from one control to several targets."""
    for t in targets:
        qc.cx(control, t)
    return qc
