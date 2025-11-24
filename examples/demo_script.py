"""
demo_script.py - Complete test of QCGM implementation
====================================================

Run this to verify your QCGM implementation works correctly.
"""

import numpy as np
from qiskit_aer import AerSimulator

# Import your QCGM modules
from qcgm.model import DiscreteGraphicalModel
from qcgm.circuit import QuantumCircuitBuilder
from qcgm.sampler import QCGMSampler
from qcgm.utils import (
    compute_fidelity,
    estimate_distribution,
    compare_distributions,
    print_comparison,
    generate_state_labels
)


def test_simple_model():
    """Test with a simple 2-variable model."""
    print("\n" + "=" * 70)
    print("TEST 1: Simple 2-Variable Model")
    print("=" * 70)
    
    # Create model
    model = DiscreteGraphicalModel(2, [{0, 1}])
    model.set_random_parameters(low=-2.0, high=-0.5, seed=42)
    
    # Get exact distribution
    exact_probs = model.compute_probabilities()
    print(f"\nExact probabilities:")
    labels = generate_state_labels(2)
    for label, prob in zip(labels, exact_probs):
        print(f"  P(x = {label}) = {prob:.6f}")
    
    # Create sampler and sample
    sampler = QCGMSampler(model)
    samples, success_rate = sampler.sample(n_samples=10000, simplified=True)
    
    print(f"\nQuantum sampling:")
    print(f"  Samples collected: {len(samples)}")
    print(f"  Success rate: {success_rate:.4f}")
    
    # Estimate distribution from samples
    estimated_probs = estimate_distribution(samples, model.n_vars)
    print(f"\nEstimated probabilities:")
    for label, prob in zip(labels, estimated_probs):
        print(f"  P̂(x = {label}) = {prob:.6f}")
    
    # Compute metrics
    metrics = compare_distributions(exact_probs, estimated_probs, 
                                    labels=('Exact', 'Quantum'))
    print_comparison(metrics)
    
    # Check if test passed
    if metrics['fidelity'] > 0.99:
        print("✓ TEST PASSED - Fidelity > 0.99")
        return True
    else:
        print("✗ TEST FAILED - Fidelity too low")
        return False


def test_chain_model():
    """Test with 3-variable chain model."""
    print("\n" + "=" * 70)
    print("TEST 2: 3-Variable Chain Model (v0 - v1 - v2)")
    print("=" * 70)
    
    # Create chain model
    model = DiscreteGraphicalModel(3, [{0, 1}, {1, 2}])
    model.set_random_parameters(low=-3.0, high=-0.1, seed=123)
    
    # Get exact distribution
    exact_probs = model.compute_probabilities()
    print(f"\nModel entropy: {model.compute_entropy():.4f} nats")
    print(f"\nExact distribution (top 4 states):")
    labels = generate_state_labels(3)
    sorted_indices = np.argsort(exact_probs)[::-1]
    for i in sorted_indices[:4]:
        print(f"  P(x = {labels[i]}) = {exact_probs[i]:.6f}")
    
    # Sample using quantum circuit
    sampler = QCGMSampler(model)
    samples, success_rate = sampler.sample(n_samples=20000, simplified=True)
    
    print(f"\nQuantum sampling:")
    print(f"  Samples: {len(samples)}")
    print(f"  Success rate: {success_rate:.4f}")
    
    # Estimate distribution
    estimated_probs = estimate_distribution(samples, model.n_vars)
    
    print(f"\nEstimated distribution (top 4 states):")
    for i in sorted_indices[:4]:
        print(f"  P̂(x = {labels[i]}) = {estimated_probs[i]:.6f}")
    
    # Compare
    metrics = compare_distributions(exact_probs, estimated_probs,
                                    labels=('Exact', 'Quantum'))
    print_comparison(metrics)
    
    if metrics['fidelity'] > 0.98:
        print("✓ TEST PASSED - Fidelity > 0.98")
        return True
    else:
        print("✗ TEST FAILED - Fidelity too low")
        return False


def test_circuit_stats():
    """Test circuit statistics."""
    print("\n" + "=" * 70)
    print("TEST 3: Circuit Statistics")
    print("=" * 70)
    
    model = DiscreteGraphicalModel(3, [{0, 1}, {1, 2}])
    model.set_random_parameters()
    
    sampler = QCGMSampler(model)
    stats = sampler.get_circuit_stats()
    
    print(f"\nCircuit properties:")
    print(f"  Number of qubits: {stats['num_qubits']}")
    print(f"  Circuit depth: {stats['depth']}")
    print(f"  Circuit size (gates): {stats['size']}")
    print(f"  Model variables: {stats['n_vars']}")
    print(f"  Model cliques: {stats['n_cliques']}")
    print(f"  Simplified: {stats['simplified']}")
    
    # Verify qubit count
    expected_qubits = model.n_vars  # Simplified circuit uses only variable qubits
    if stats['num_qubits'] == expected_qubits:
        print(f"\n✓ Correct qubit count ({expected_qubits})")
        return True
    else:
        print(f"\n✗ Wrong qubit count: {stats['num_qubits']} vs {expected_qubits}")
        return False


def test_different_structures():
    """Test different graph structures."""
    print("\n" + "=" * 70)
    print("TEST 4: Different Graph Structures")
    print("=" * 70)
    
    structures = [
        ("Independent", 2, []),
        ("Single edge", 2, [{0, 1}]),
        ("Chain", 3, [{0, 1}, {1, 2}]),
        ("Star", 3, [{0, 1}, {0, 2}]),
    ]
    
    results = []
    
    for name, n_vars, cliques in structures:
        print(f"\n{name} ({n_vars} vars, {len(cliques)} cliques):")
        
        model = DiscreteGraphicalModel(n_vars, cliques)
        model.set_random_parameters(seed=42)
        
        exact_probs = model.compute_probabilities()
        
        sampler = QCGMSampler(model)
        samples, _ = sampler.sample(n_samples=5000, simplified=True)
        
        estimated_probs = estimate_distribution(samples, model.n_vars)
        fidelity = compute_fidelity(exact_probs, estimated_probs)
        
        print(f"  Samples: {len(samples)}, Fidelity: {fidelity:.6f}")
        
        passed = fidelity > 0.98
        results.append(passed)
        print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    
    all_passed = all(results)
    print(f"\n{'✓ All structures passed!' if all_passed else '✗ Some structures failed'}")
    
    return all_passed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("QCGM IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Simple 2-variable model", test_simple_model),
        ("3-variable chain model", test_chain_model),
        ("Circuit statistics", test_circuit_stats),
        ("Different structures", test_different_structures),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            print(f"\n✗ TEST CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed, error in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12s} - {name}")
        if error:
            print(f"             Error: {error}")
    
    total_passed = sum(1 for _, p, _ in results if p)
    total_tests = len(results)
    
    print(f"\n{total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your QCGM implementation is working!")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)