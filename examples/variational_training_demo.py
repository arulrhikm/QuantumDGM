"""
Simple demonstration of variational circuit training.

This shows how to train a quantum circuit to approximate distributions
for models too large for exact amplitude encoding.
"""

import numpy as np
from qcgm import (
    DiscreteGraphicalModel, 
    ApproximateCircuitBuilder,
    smart_circuit_builder,
    compute_fidelity
)

print("=" * 70)
print("VARIATIONAL CIRCUIT TRAINING DEMO")
print("=" * 70)

# ============================================================================
# Example 1: Manual Training
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 1: Manual Training")
print("=" * 70)

# Create a model with 10 variables (too large for exact on laptop)
print("\nCreating 10-variable chain model...")
model = DiscreteGraphicalModel(10, [{i, i+1} for i in range(9)])
model.set_random_parameters(low=-2.0, high=-0.5, seed=42)

# Get target distribution (for comparison)
target_probs = model.compute_probabilities()
print(f"  ✓ Model created: {model.n_vars} variables, {model.n_cliques} cliques")

# Build and train variational circuit
print("\nTraining variational circuit...")
builder = ApproximateCircuitBuilder(depth=3, entanglement='linear')

circuit, params, info = builder.build_circuit_with_target(
    model,
    n_optimization_steps=80,
    verbose=True,
    seed=42
)

print(f"\n✓ Training complete!")
print(f"  Final fidelity: {info['final_fidelity']:.4f}")
print(f"  Circuit depth: {circuit.depth()}")
print(f"  Parameters optimized: {len(params)}")
print(f"  Iterations: {info['n_iterations']}")


# ============================================================================
# Example 2: Smart Circuit Builder (Automatic)
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Smart Circuit Builder (Automatic)")
print("=" * 70)

# Small model - will use exact
model_small = DiscreteGraphicalModel(6, [{0,1}, {1,2}, {2,3}, {3,4}, {4,5}])
model_small.set_random_parameters(seed=123)

print("\nBuilding circuit for small model (n=6)...")
circuit_small, info_small = smart_circuit_builder(
    model_small,
    max_exact_vars=10,
    verbose=True
)
print(f"  ✓ Method: {info_small['method']}")

# Large model - will use approximate + training
model_large = DiscreteGraphicalModel(12, [{i, i+1} for i in range(11)])
model_large.set_random_parameters(seed=456)

print("\nBuilding circuit for large model (n=12)...")
circuit_large, info_large = smart_circuit_builder(
    model_large,
    max_exact_vars=10,
    optimize_approx=True,  # Enable training
    approx_depth=3,
    verbose=True
)
print(f"  ✓ Method: {info_large['method']}")
if 'final_fidelity' in info_large:
    print(f"  ✓ Fidelity: {info_large['final_fidelity']:.4f}")


# ============================================================================
# Example 3: Comparing Loss Functions
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Comparing Loss Functions")
print("=" * 70)

model = DiscreteGraphicalModel(7, [{0,1,2}, {2,3,4}, {4,5,6}])
model.set_random_parameters(seed=789)

loss_functions = ['kl', 'fidelity', 'l2']
results = {}

for loss_fn in loss_functions:
    print(f"\nTraining with loss='{loss_fn}'...")
    builder = ApproximateCircuitBuilder(depth=3, entanglement='clique')
    
    circuit, params, info = builder.build_circuit_with_target(
        model,
        loss=loss_fn,
        n_optimization_steps=40,
        verbose=False,
        seed=42
    )
    
    results[loss_fn] = info['final_fidelity']
    print(f"  ✓ Final fidelity: {info['final_fidelity']:.4f}")
    print(f"  ✓ Iterations: {info['n_iterations']}")

print("\nLoss Function Comparison:")
print(f"{'Loss Function':15s} {'Fidelity':>10s}")
print("-" * 30)
for loss_fn, fid in results.items():
    print(f"{loss_fn:15s} {fid:10.4f}")


# ============================================================================
# Example 4: Custom Callback
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Custom Callback for Monitoring")
print("=" * 70)

model = DiscreteGraphicalModel(6, [{0,1}, {1,2,3}, {3,4,5}])
model.set_random_parameters(seed=999)

# Track progress
progress = {'steps': [], 'costs': [], 'best': float('inf')}

def progress_callback(params, cost, iteration):
    progress['steps'].append(iteration)
    progress['costs'].append(cost)
    if cost < progress['best']:
        progress['best'] = cost
        print(f"  → New best at iteration {iteration}: {cost:.6f}")

print("\nTraining with progress tracking...")
builder = ApproximateCircuitBuilder(depth=2, entanglement='linear')

circuit, params, info = builder.build_circuit_with_target(
    model,
    n_optimization_steps=50,
    callback=progress_callback,
    verbose=False,
    seed=42
)

print(f"\n✓ Training complete:")
print(f"  Total iterations: {len(progress['steps'])}")
print(f"  Initial cost: {progress['costs'][0]:.6f}")
print(f"  Final cost: {progress['costs'][-1]:.6f}")
print(f"  Best cost: {progress['best']:.6f}")
print(f"  Final fidelity: {info['final_fidelity']:.4f}")


# ============================================================================
# Example 5: Recommended Depth
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 5: Automatic Depth Selection")
print("=" * 70)

models = [
    (8, [{i, i+1} for i in range(7)], "chain"),
    (8, [{0,1,2,3}, {4,5,6,7}], "two cliques"),
    (8, [{i, j} for i in range(8) for j in range(i+1, 8)], "complete"),
]

for n, cliques, structure in models:
    model = DiscreteGraphicalModel(n, cliques)
    recommended = ApproximateCircuitBuilder.estimate_optimal_depth(model)
    avg_clique_size = np.mean([len(c) for c in model.cliques])
    
    print(f"\n{structure:15s} (n={n}, avg_clique={avg_clique_size:.1f}):")
    print(f"  Recommended depth: {recommended}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
✓ Variational circuit training enables quantum sampling for large models

Key Benefits:
  • Works for n > 10 (where exact methods fail)
  • Parameters: O(n × depth) instead of O(2^n)
  • Tunable accuracy via depth and optimization steps
  • Multiple loss functions available
  • Automatic or manual configuration

Usage:
  1. For small models (n ≤ 10): Use exact circuits
  2. For large models (n > 10): Use variational training
  3. Use smart_circuit_builder() for automatic selection

Next Steps:
  • Run examples/test_variational_training.py for full tests
  • See VARIATIONAL_TRAINING.md for complete documentation
  • Experiment with different depths and entanglement strategies
""")

print("=" * 70)
print("✨ Demo complete!")
print("=" * 70)

