"""
sampler.py - FULLY CORRECTED
=============================

The issue was: simplified circuit has NO auxiliary qubits, but parsing
logic was expecting them!
"""

import numpy as np
from typing import Tuple, Optional
from qiskit_aer import AerSimulator
from .circuit import QuantumCircuitBuilder


class QCGMSampler:
    """
    Quantum sampler for discrete graphical models.
    
    FULLY CORRECTED: Handles both simplified and full circuits properly.
    """
    
    def __init__(self, model):
        """Initialize sampler with a DiscreteGraphicalModel."""
        self.model = model
        self.circuit = None
        self._last_success_rate = None
        self._simplified = True  # Track which circuit type we built
    
    def build_circuit(self, simplified: bool = True, use_aux: bool = False):
        """
        Build the quantum circuit.
        
        Args:
            simplified: Use simplified direct circuit (recommended, no aux qubits)
            use_aux: Use auxiliary qubits for real part extraction (only if not simplified)
        """
        self._simplified = simplified
        
        if simplified:
            self.circuit = QuantumCircuitBuilder.build_circuit_simplified(self.model)
        else:
            self.circuit = QuantumCircuitBuilder.build_circuit(self.model, use_aux=use_aux)
        
        return self.circuit
    
    def sample(self, 
               n_samples: int = 1000,
               backend=None,
               simplified: bool = True,
               **backend_kwargs) -> Tuple[np.ndarray, float]:
        """
        Sample from the graphical model.
        
        CORRECTED: Different parsing for simplified vs full circuit.
        
        KEY FIX: Qiskit uses little-endian bit ordering!
        Bitstring "01" means qubit[0]=1, qubit[1]=0
        We should NOT reverse the string, just read it right-to-left!
        
        Simplified circuit:
        - Only variable qubits (n qubits)
        - No auxiliary qubits
        - Success rate = 100%
        - Bitstring: "q[n-1]...q[1]q[0]"
        
        Full circuit with aux:
        - Variable + auxiliary qubits (n + |C| qubits)
        - Must check aux qubits are all 0
        - Success rate < 100%
        - Bitstring: "aux[|C|-1]...aux[0] var[n-1]...var[0]"
        """
        # Build circuit if needed
        if self.circuit is None:
            self.build_circuit(simplified=simplified)
        
        # Use simulator if no backend
        if backend is None:
            backend = AerSimulator()
        
        # Execute
        job = backend.run(self.circuit, shots=n_samples, **backend_kwargs)
        result = job.result()
        counts = result.get_counts()
        
        valid_samples = []
        total_shots = 0
        
        n_vars = self.model.n_vars
        
        if self._simplified:
            # SIMPLIFIED CIRCUIT: No auxiliary qubits!
            # Bitstring is just the variable qubits
            
            for bitstring, count in counts.items():
                total_shots += count
                
                # Remove spaces
                bitstring_clean = ''.join(bitstring.split())
                
                # Should be exactly n_vars bits
                if len(bitstring_clean) != n_vars:
                    print(f"Warning: unexpected bitstring length {len(bitstring_clean)}, expected {n_vars}")
                    print(f"  Bitstring: '{bitstring}'")
                    continue
                
                # CRITICAL FIX: Qiskit uses little-endian ordering
                # Bitstring "01" = qubit[0]=1, qubit[1]=0
                # We want sample[i] = value of variable i = value of qubit i
                # So read the bitstring from RIGHT to LEFT
                # 
                # Example: bitstring = "10" (qubit[1]=1, qubit[0]=0)
                # We want: sample = [0, 1] (var[0]=0, var[1]=1)
                # So: sample[i] = bitstring[n-1-i]
                
                sample = np.array([int(bitstring_clean[n_vars - 1 - i]) for i in range(n_vars)])
                
                # Add this sample 'count' times
                for _ in range(count):
                    valid_samples.append(sample)
            
            # Success rate is 100% for simplified circuit
            success_rate = 1.0
            
        else:
            # FULL CIRCUIT: Has auxiliary qubits
            n_aux = self.model.n_cliques
            
            for bitstring, count in counts.items():
                total_shots += count
                
                # Remove spaces
                bitstring_clean = ''.join(bitstring.split())
                
                # Should be n_aux + n_vars bits
                expected_len = n_aux + n_vars
                if len(bitstring_clean) != expected_len:
                    print(f"Warning: unexpected bitstring length {len(bitstring_clean)}, expected {expected_len}")
                    continue
                
                # Split: "aux[n_cliques-1]...aux[0] var[n-1]...var[0]"
                aux_bits = bitstring_clean[:n_aux]
                var_bits = bitstring_clean[n_aux:]
                
                # Check auxiliary qubits all 0 (successful real part extraction)
                if all(bit == '0' for bit in aux_bits):
                    # Convert variable bits to sample with correct ordering
                    sample = np.array([int(var_bits[n_vars - 1 - i]) for i in range(n_vars)])
                    
                    for _ in range(count):
                        valid_samples.append(sample)
            
            # Success rate for full circuit
            success_rate = len(valid_samples) / total_shots if total_shots > 0 else 0.0
        
        self._last_success_rate = success_rate
        
        return np.array(valid_samples), success_rate
    
    def sample_with_retry(self,
                         target_samples: int,
                         max_shots: int = None,
                         backend=None,
                         simplified: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Sample with automatic retry until target reached.
        
        Args:
            target_samples: Desired number of valid samples
            max_shots: Maximum total shots (default: target * 100)
            backend: Qiskit backend
            simplified: Use simplified circuit (no aux qubits)
        """
        if max_shots is None:
            max_shots = target_samples * 100
        
        all_samples = []
        total_shots = 0
        attempts = 0
        
        while len(all_samples) < target_samples and total_shots < max_shots:
            # For simplified circuit, we know success rate is 1.0
            if simplified:
                shots_this_round = min(target_samples - len(all_samples), max_shots - total_shots)
            else:
                # Estimate shots needed based on last success rate
                if self._last_success_rate and self._last_success_rate > 0:
                    shots_this_round = min(
                        int((target_samples - len(all_samples)) / self._last_success_rate * 2),
                        max_shots - total_shots
                    )
                else:
                    shots_this_round = min(1000, max_shots - total_shots)
            
            samples, rate = self.sample(shots_this_round, backend, simplified)
            
            if len(samples) > 0:
                all_samples.extend(samples)
            
            total_shots += shots_this_round
            attempts += 1
            
            # Break if success rate is too low (only for non-simplified)
            if not simplified and rate < 0.001 and attempts > 3:
                print(f"Warning: Very low success rate ({rate:.6f}), stopping")
                break
        
        all_samples = np.array(all_samples[:target_samples])
        
        info = {
            'attempts': attempts,
            'total_shots': total_shots,
            'final_success_rate': len(all_samples) / total_shots if total_shots > 0 else 0.0,
            'achieved_target': len(all_samples) >= target_samples,
            'actual_samples': len(all_samples)
        }
        
        return all_samples, info
    
    def estimate_success_probability(self, n_trials: int = 1000, 
                                     backend=None, simplified: bool = True) -> float:
        """Estimate success probability δ*."""
        _, success_rate = self.sample(n_trials, backend, simplified)
        return success_rate
    
    @property
    def last_success_rate(self) -> Optional[float]:
        """Get last observed success rate."""
        return self._last_success_rate
    
    def get_circuit_stats(self) -> dict:
        """Get circuit statistics."""
        if self.circuit is None:
            self.build_circuit()
        
        return {
            'num_qubits': self.circuit.num_qubits,
            'depth': self.circuit.depth(),
            'size': self.circuit.size(),
            'n_vars': self.model.n_vars,
            'n_cliques': self.model.n_cliques,
            'required_qubits': QuantumCircuitBuilder.required_qubits(self.model),
            'simplified': self._simplified
        }
    
    def __repr__(self) -> str:
        return f"QCGMSampler(n_vars={self.model.n_vars}, n_cliques={self.model.n_cliques})"