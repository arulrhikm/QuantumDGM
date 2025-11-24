"""
circuit.py - FINAL WORKING VERSION
===================================

This version explicitly manages all registers to avoid Qiskit auto-creation issues.
"""

import numpy as np
import itertools
from typing import List, Set, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize


class QuantumCircuitBuilder:
    """
    Quantum circuit builder - GUARANTEED WORKING VERSION.
    """
    
    @staticmethod
    def build_circuit_direct_initialization(model) -> QuantumCircuit:
        """
        Build circuit using direct state initialization.
        
        CRITICAL FIX: Must reorder probabilities to match Qiskit's qubit ordering!
        
        The issue:
        - Model orders states as: (v0,v1,v2,...) with itertools.product([0,1], repeat=n)
          This gives: (0,0), (0,1), (1,0), (1,1) for n=2
        
        - Qiskit Initialize expects computational basis order: |00⟩, |01⟩, |10⟩, |11⟩
          In little-endian: qubit[0] is rightmost bit
          |01⟩ means qubit[1]=0, qubit[0]=1
        
        - Mapping: model state (v0,v1) -> qubits q[0]=v0, q[1]=v1
          -> Qiskit basis state |v1 v0⟩ (big-endian notation)
          -> Qiskit index = v1*2^1 + v0*2^0 = v1*2 + v0
        
        - For model state (0,1): v0=0, v1=1 -> Qiskit index = 1*2+0 = 2
        - But model has this at index 1!
        - So we must reorder: model[1] -> qiskit[2]
        """
        n = model.n_vars
        
        # Get exact probabilities from model
        probs = model.compute_probabilities()
        
        # CRITICAL: Reorder probabilities to match Qiskit's basis ordering
        # Generate all states in model order
        import itertools
        states_model = list(itertools.product([0, 1], repeat=n))
        
        # Create mapping: model_index -> qiskit_index
        # For state (v0, v1, ..., v_{n-1}), Qiskit index is:
        # sum(v_i * 2^i for i in range(n))
        probs_reordered = np.zeros_like(probs)
        for model_idx, state in enumerate(states_model):
            qiskit_idx = sum(state[i] * (2**i) for i in range(n))
            probs_reordered[qiskit_idx] = probs[model_idx]
        
        # Compute amplitudes from reordered probabilities
        amplitudes = np.sqrt(probs_reordered)
        
        # Create registers EXPLICITLY
        qreg = QuantumRegister(n, 'q')
        creg = ClassicalRegister(n, 'c')
        
        # Create circuit with these explicit registers
        qc = QuantumCircuit(qreg, creg)
        
        # Use Initialize gate to set the state
        init_gate = Initialize(amplitudes)
        qc.append(init_gate, qreg)
        
        # Measure with explicit register mapping
        qc.measure(qreg, creg)
        
        return qc
    
    @staticmethod
    def build_circuit_amplitude_encoding(model) -> QuantumCircuit:
        """Alias for direct initialization."""
        return QuantumCircuitBuilder.build_circuit_direct_initialization(model)
    
    @staticmethod
    def build_circuit(model, use_aux: bool = False) -> QuantumCircuit:
        """Main circuit builder."""
        return QuantumCircuitBuilder.build_circuit_direct_initialization(model)
    
    @staticmethod
    def build_circuit_simplified(model) -> QuantumCircuit:
        """Simplified circuit (same as direct initialization)."""
        return QuantumCircuitBuilder.build_circuit_direct_initialization(model)
    
    @staticmethod
    def circuit_depth_estimate(model) -> int:
        """Estimate circuit depth."""
        return 2 ** model.n_vars
    
    @staticmethod
    def required_qubits(model) -> int:
        """Required qubits."""
        return model.n_vars


def quick_test():
    """Quick test of circuit builder."""
    print("Quick Circuit Test")
    print("=" * 60)
    
    from qcgm.model import DiscreteGraphicalModel
    from qiskit_aer import AerSimulator
    
    # Create model
    model = DiscreteGraphicalModel(2, [{0, 1}])
    model.set_random_parameters(low=-2.0, high=-0.5, seed=42)
    
    exact_probs = model.compute_probabilities()
    print(f"Exact probs: {exact_probs}")
    
    # Build circuit
    circuit = QuantumCircuitBuilder.build_circuit_simplified(model)
    
    print(f"\nCircuit: {circuit.num_qubits} qubits, {circuit.num_clbits} clbits")
    print(f"Registers: Q={len(circuit.qregs)}, C={len(circuit.cregs)}")
    
    # Run
    backend = AerSimulator()
    job = backend.run(circuit, shots=5000)
    result = job.result()
    counts = result.get_counts()
    
    print(f"\nSample bitstring: {list(counts.keys())[0]}")
    
    # Parse
    samples = []
    for bitstring, count in counts.items():
        clean = ''.join(bitstring.split())
        if len(clean) == model.n_vars:
            sample = [int(b) for b in reversed(clean)]
            samples.extend([sample] * count)
    
    print(f"Parsed {len(samples)} samples")
    
    # Estimate
    from qcgm.utils import estimate_distribution, compute_fidelity
    if len(samples) > 0:
        samples = np.array(samples)
        est_probs = estimate_distribution(samples, model.n_vars)
        fidelity = compute_fidelity(exact_probs, est_probs)
        
        print(f"Est probs: {est_probs}")
        print(f"Fidelity: {fidelity:.6f}")
        
        if fidelity > 0.99:
            print("\n✓ SUCCESS!")
            return True
        else:
            print("\n✗ FAILED")
            return False
    else:
        print("\n✗ NO SAMPLES PARSED")
        return False


if __name__ == "__main__":
    quick_test()