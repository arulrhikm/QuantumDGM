"""
Classical simulation / execution of the quantum circuit using BlueQubit.
"""

from qiskit import transpile
import bluequbit

def simulate(qc, shots=1024, device="cpu", api_token: str = None, asynchronous: bool = False):
    """
    Run circuit using BlueQubit backend, return measurement counts.
    
    Args:
        qc: Qiskit QuantumCircuit
        shots: number of measurement shots (for sampling)
        device: which BlueQubit device/simulator to use (e.g. "cpu", "gpu", "pauli-path", etc.)
        api_token: optional BlueQubit API token (or read from env)
        asynchronous: if True, submit asynchronously and return job handle
    
    Returns:
        dict: measurement counts (if synchronous) or job handle (if asynchronous)
    """
    # Initialize the BlueQubit client
    bq_client = bluequbit.init(api_token)
    
    # Transpile the circuit for BlueQubit/Qiskit compatibility
    # (you may want to transpile with some basis gates depending on constraints)
    compiled = transpile(qc)
    
    # Submit job to BlueQubit
    job = bq_client.run(compiled, device=device, shots=shots, asynchronous=asynchronous)
    
    if asynchronous:
        return job  # caller can wait or poll
    else:
        # job is completed (blocking), so fetch counts
        result = job
        counts = result.get_counts()
        return counts
