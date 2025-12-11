"""
circuit.py
===================================

Quantum circuit builders for discrete graphical models.

This module provides:
1. Exact circuit construction via amplitude encoding (for n <= 10)
2. Approximate variational circuits (for n > 10)

The approximate circuits use hardware-efficient ansatzes that can
represent the target distribution with tunable accuracy.
"""

import numpy as np
import itertools
from typing import List, Set, Tuple, Optional, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize


class QuantumCircuitBuilder:
    """
    Quantum circuit builder
    """
    
    @staticmethod
    def build_circuit_direct_initialization(model) -> QuantumCircuit:
        """
        Build circuit using direct state initialization.
        
        Must reorder probabilities to match Qiskit's qubit ordering!
        
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


class ApproximateCircuitBuilder:
    """
    Approximate circuit builder for large graphical models (n > 10).
    
    Uses variational quantum circuits (hardware-efficient ansatz) that can
    approximate the target distribution without computing all 2^n amplitudes.
    
    Key features:
    - O(n * depth) parameters instead of O(2^n)
    - Structure-aware entanglement based on clique structure
    - Supports gradient-based optimization for parameter learning
    
    Example:
        >>> model = DiscreteGraphicalModel(12, cliques)  # 12 variables
        >>> builder = ApproximateCircuitBuilder(depth=4)
        >>> circuit, params = builder.build_variational_circuit(model)
        >>> # params can be optimized to match target distribution
    """
    
    def __init__(self, depth: int = 3, entanglement: str = 'clique'):
        """
        Initialize the approximate circuit builder.
        
        Args:
            depth: Number of variational layers (more = more expressive)
            entanglement: Entanglement strategy:
                - 'clique': Entangle based on graphical model cliques
                - 'linear': Sequential CX gates (q0-q1, q1-q2, ...)
                - 'full': All-to-all entanglement (expensive)
        """
        self.depth = depth
        self.entanglement = entanglement
    
    def build_variational_circuit(self, model, 
                                  initial_params: Optional[np.ndarray] = None,
                                  seed: int = None) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Build a variational circuit that approximates the model distribution.
        
        Args:
            model: DiscreteGraphicalModel instance
            initial_params: Initial parameter values (random if None)
            seed: Random seed for parameter initialization
        
        Returns:
            Tuple of (QuantumCircuit, parameters array)
        """
        n = model.n_vars
        
        # Calculate number of parameters: 2 per qubit per layer (RY, RZ)
        n_params = 2 * n * self.depth
        
        # Initialize parameters
        if initial_params is not None:
            params = np.array(initial_params, dtype=np.float64)
            if len(params) != n_params:
                raise ValueError(f"Expected {n_params} parameters, got {len(params)}")
        else:
            rng = np.random.default_rng(seed)
            params = rng.uniform(0, 2 * np.pi, size=n_params)
        
        # Create circuit
        qreg = QuantumRegister(n, 'q')
        creg = ClassicalRegister(n, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Get entanglement edges based on strategy
        entanglement_edges = self._get_entanglement_edges(model)
        
        # Build variational layers
        param_idx = 0
        for layer in range(self.depth):
            # Single-qubit rotation layer
            for i in range(n):
                qc.ry(params[param_idx], qreg[i])
                param_idx += 1
                qc.rz(params[param_idx], qreg[i])
                param_idx += 1
            
            # Entanglement layer
            for (i, j) in entanglement_edges:
                qc.cx(qreg[i], qreg[j])
        
        # Measurement
        qc.measure(qreg, creg)
        
        return qc, params
    
    def _get_entanglement_edges(self, model) -> List[Tuple[int, int]]:
        """Get entanglement edges based on strategy."""
        n = model.n_vars
        
        if self.entanglement == 'clique':
            # Entangle qubits that are in the same clique
            edges = set()
            for clique in model.cliques:
                clique_list = sorted(clique)
                for i in range(len(clique_list) - 1):
                    edges.add((clique_list[i], clique_list[i + 1]))
            return list(edges)
        
        elif self.entanglement == 'linear':
            # Linear chain: q0-q1, q1-q2, ..., q(n-2)-q(n-1)
            return [(i, i + 1) for i in range(n - 1)]
        
        elif self.entanglement == 'full':
            # Full entanglement (expensive for large n)
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        else:
            raise ValueError(f"Unknown entanglement strategy: {self.entanglement}")
    
    def build_circuit_with_target(self, model, 
                                  n_optimization_steps: int = 100,
                                  learning_rate: float = 0.1,
                                  seed: int = None,
                                  loss: str = 'kl',
                                  initial_params: Optional[np.ndarray] = None,
                                  verbose: bool = False,
                                  callback=None) -> Tuple[QuantumCircuit, np.ndarray, Dict]:
        """
        Build and optimize a variational circuit to match target distribution.
        
        Uses gradient-free optimization (COBYLA) to find parameters that
        minimize the specified loss to the target distribution.
        
        Args:
            model: DiscreteGraphicalModel instance
            n_optimization_steps: Maximum optimization iterations
            learning_rate: Step size for parameter updates (rhobeg)
            seed: Random seed for parameter initialization
            loss: Loss function - 'kl', 'fidelity', or 'l2'
            initial_params: Starting parameters (random if None)
            verbose: Print optimization progress
            callback: Optional callback function(params, cost, iteration)
        
        Returns:
            Tuple of (optimized circuit, optimal parameters, optimization info)
        
        Example:
            >>> model = DiscreteGraphicalModel(8, [{i,i+1} for i in range(7)])
            >>> model.set_random_parameters()
            >>> builder = ApproximateCircuitBuilder(depth=3)
            >>> circuit, params, info = builder.build_circuit_with_target(
            ...     model, n_optimization_steps=50, verbose=True)
        
        Note:
            For n > 15, this may be slow. Consider using fewer optimization
            steps or a simpler entanglement strategy.
        """
        from scipy.optimize import minimize
        
        n = model.n_vars
        target_probs = model.compute_probabilities()
        
        # Build initial circuit to get parameter shape
        if initial_params is None:
            _, initial_params = self.build_variational_circuit(model, seed=seed)
        else:
            initial_params = np.array(initial_params, dtype=np.float64)
        
        # Track iterations
        iteration_counter = {'count': 0, 'costs': [], 'best_cost': float('inf')}
        
        # Cost function
        def cost_function(params):
            # Build circuit with current params
            qc, _ = self.build_variational_circuit(model, initial_params=params)
            
            # Simulate to get distribution (using statevector for speed)
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            
            # Remove measurements for statevector simulation
            qc_sv = qc.remove_final_measurements(inplace=False)
            
            backend = AerSimulator(method='statevector')
            qc_sv.save_statevector()
            job = backend.run(transpile(qc_sv, backend), shots=1)
            result = job.result()
            statevector = result.get_statevector()
            
            # Get probabilities from statevector
            probs = np.abs(statevector.data) ** 2
            
            # Reorder to match model ordering
            probs_reordered = self._reorder_probs_from_qiskit(probs, n)
            
            # Compute loss
            epsilon = 1e-10
            if loss == 'kl':
                # KL divergence: D_KL(P || Q) = sum P log(P/Q)
                cost = np.sum(target_probs * np.log((target_probs + epsilon) / (probs_reordered + epsilon)))
            elif loss == 'fidelity':
                # Negative fidelity (to minimize)
                fid = np.sum(np.sqrt(target_probs * probs_reordered))
                cost = -fid
            elif loss == 'l2':
                # L2 distance
                cost = np.sum((target_probs - probs_reordered) ** 2)
            else:
                raise ValueError(f"Unknown loss: {loss}")
            
            # Track progress
            iteration_counter['count'] += 1
            iteration_counter['costs'].append(cost)
            if cost < iteration_counter['best_cost']:
                iteration_counter['best_cost'] = cost
            
            # Callback
            if callback is not None:
                callback(params, cost, iteration_counter['count'])
            
            # Verbose logging
            if verbose and iteration_counter['count'] % max(1, n_optimization_steps // 10) == 0:
                print(f"  Iteration {iteration_counter['count']:3d}: {loss}={cost:.6f}")
            
            return cost
        
        # Optimize
        if verbose:
            print(f"Training variational circuit:")
            print(f"  Parameters: {len(initial_params)}")
            print(f"  Depth: {self.depth}")
            print(f"  Entanglement: {self.entanglement}")
            print(f"  Loss: {loss}")
        
        # COBYLA requires maxiter >= n_params + 2
        min_maxiter = len(initial_params) + 2
        effective_maxiter = max(n_optimization_steps, min_maxiter)
        
        if verbose and effective_maxiter > n_optimization_steps:
            print(f"  Note: Increasing maxiter from {n_optimization_steps} to {effective_maxiter} (COBYLA requirement)")
        
        result = minimize(
            cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': effective_maxiter, 'rhobeg': learning_rate}
        )
        
        optimal_params = result.x
        final_circuit, _ = self.build_variational_circuit(model, initial_params=optimal_params)
        
        # Compute final metrics
        final_qc_sv = final_circuit.remove_final_measurements(inplace=False)
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        backend = AerSimulator(method='statevector')
        final_qc_sv.save_statevector()
        job = backend.run(transpile(final_qc_sv, backend), shots=1)
        final_sv = job.result().get_statevector()
        final_probs = np.abs(final_sv.data) ** 2
        final_probs_reordered = self._reorder_probs_from_qiskit(final_probs, n)
        
        # Compute fidelity for info
        from qcgm.utils import compute_fidelity
        final_fidelity = compute_fidelity(target_probs, final_probs_reordered)
        
        info = {
            'success': result.success,
            'final_cost': result.fun,
            'final_fidelity': final_fidelity,
            'n_iterations': iteration_counter['count'],
            'n_params': len(optimal_params),
            'depth': self.depth,
            'entanglement': self.entanglement,
            'loss_history': iteration_counter['costs'],
            'optimizer': 'COBYLA'
        }
        
        if verbose:
            print(f"\nOptimization complete:")
            print(f"  Final {loss}: {result.fun:.6f}")
            print(f"  Final fidelity: {final_fidelity:.6f}")
            print(f"  Iterations: {iteration_counter['count']}")
        
        return final_circuit, optimal_params, info
    
    def _reorder_probs_from_qiskit(self, probs: np.ndarray, n: int) -> np.ndarray:
        """
        Reorder probabilities from Qiskit ordering to model ordering.
        
        Qiskit uses little-endian: index j = sum(bit_i * 2^i)
        Model uses big-endian: state (v0,v1,...,vn-1) -> index = sum(v_i * 2^(n-1-i))
        
        So for Qiskit index j:
          - Extract bits: v_i = (j >> i) & 1
          - Model index = sum(v_i * 2^(n-1-i))
        """
        probs_reordered = np.zeros_like(probs)
        for qiskit_idx in range(len(probs)):
            # Extract state from Qiskit index (little-endian)
            state = tuple((qiskit_idx >> i) & 1 for i in range(n))
            # Convert to model index (big-endian)
            model_idx = sum(state[i] * (2 ** (n - 1 - i)) for i in range(n))
            probs_reordered[model_idx] = probs[qiskit_idx]
        return probs_reordered
    
    @staticmethod
    def estimate_optimal_depth(model) -> int:
        """
        Estimate optimal circuit depth based on model structure.
        
        Rule of thumb:
        - More cliques = more entanglement needed = deeper circuit
        - Tree structures need less depth than dense graphs
        
        Args:
            model: DiscreteGraphicalModel instance
        
        Returns:
            Recommended circuit depth
        """
        n = model.n_vars
        n_cliques = model.n_cliques
        
        # Base depth scales with sqrt(n)
        base_depth = max(2, int(np.sqrt(n)))
        
        # Add depth for dense clique structures
        avg_clique_size = np.mean([len(c) for c in model.cliques]) if model.cliques else 0
        density_factor = 1 + int(avg_clique_size / 3)
        
        return min(base_depth * density_factor, 10)  # Cap at 10 layers


def smart_circuit_builder(model, 
                          max_exact_vars: int = 10,
                          approx_depth: int = None,
                          optimize_approx: bool = False,
                          verbose: bool = False) -> Tuple[QuantumCircuit, Dict]:
    """
    Automatically choose the best circuit building strategy.
    
    For small models (n <= max_exact_vars), uses exact amplitude encoding.
    For larger models, uses approximate variational circuits.
    
    Args:
        model: DiscreteGraphicalModel instance
        max_exact_vars: Maximum variables for exact circuit (default 10)
        approx_depth: Depth for approximate circuit (auto if None)
        optimize_approx: Whether to optimize approximate circuit params
        verbose: Print info about strategy chosen
    
    Returns:
        Tuple of (QuantumCircuit, info dict)
    
    Example:
        >>> # Small model - uses exact circuit
        >>> model_small = DiscreteGraphicalModel(4, [{0,1}, {1,2}, {2,3}])
        >>> circuit, info = smart_circuit_builder(model_small)
        
        >>> # Large model - uses approximate circuit  
        >>> model_large = DiscreteGraphicalModel(15, cliques)
        >>> circuit, info = smart_circuit_builder(model_large, optimize_approx=True)
    """
    n = model.n_vars
    
    if n <= max_exact_vars:
        # Use exact circuit
        if verbose:
            print(f"Using exact amplitude encoding ({n} variables)")
        
        circuit = QuantumCircuitBuilder.build_circuit_simplified(model)
        info = {
            'method': 'exact',
            'n_vars': n,
            'circuit_depth': circuit.depth(),
            'gate_count': circuit.size()
        }
    else:
        # Use approximate circuit
        if verbose:
            print(f"Using approximate variational circuit ({n} variables)")
        
        if approx_depth is None:
            approx_depth = ApproximateCircuitBuilder.estimate_optimal_depth(model)
        
        builder = ApproximateCircuitBuilder(depth=approx_depth, entanglement='clique')
        
        if optimize_approx:
            circuit, params, opt_info = builder.build_circuit_with_target(
                model, verbose=verbose
            )
            info = {
                'method': 'approximate_optimized',
                'n_vars': n,
                'depth': approx_depth,
                **opt_info
            }
        else:
            circuit, params = builder.build_variational_circuit(model)
            info = {
                'method': 'approximate',
                'n_vars': n,
                'depth': approx_depth,
                'n_params': len(params),
                'circuit_depth': circuit.depth(),
                'gate_count': circuit.size()
            }
    
    return circuit, info


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