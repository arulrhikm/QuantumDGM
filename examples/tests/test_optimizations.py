"""
Test script for QCGM optimizations:
1. Sparse diagonal Hamiltonian
2. Approximate circuit builder
"""

import numpy as np
import time

from qcgm import DiscreteGraphicalModel, ApproximateCircuitBuilder, smart_circuit_builder

print('=' * 60)
print('OPTIMIZATION TESTS')
print('=' * 60)

# Test 1: Sparse Hamiltonian performance
print('\n1. SPARSE DIAGONAL HAMILTONIAN TEST')
print('-' * 40)

for n in [4, 6, 8]:
    cliques = [{i, i+1} for i in range(n-1)]
    model = DiscreteGraphicalModel(n, cliques)
    model.set_random_parameters(seed=42)
    
    # Time the optimized diagonal computation
    start = time.perf_counter()
    for _ in range(10):
        diag = model.compute_hamiltonian_diagonal()
    diag_time = (time.perf_counter() - start) / 10
    
    # Time the full matrix computation  
    start = time.perf_counter()
    for _ in range(10):
        H = model.compute_hamiltonian()
    full_time = (time.perf_counter() - start) / 10
    
    print(f'  n={n}: diagonal={diag_time*1000:.2f}ms, full_matrix={full_time*1000:.2f}ms')
    print(f'       Memory: diagonal={2**n * 8 / 1024:.1f}KB vs full={4**n * 8 / 1024:.1f}KB')

# Test 2: Cache effectiveness
print('\n2. CACHE EFFECTIVENESS TEST')
print('-' * 40)

model = DiscreteGraphicalModel(6, [{0,1}, {1,2}, {2,3}, {3,4}, {4,5}])
model.set_random_parameters(seed=42)

# First call (cache miss)
start = time.perf_counter()
probs1 = model.compute_probabilities()
first_time = time.perf_counter() - start

# Second call (cache hit)
start = time.perf_counter()
probs2 = model.compute_probabilities()
cached_time = time.perf_counter() - start

print(f'  First call: {first_time*1000:.2f}ms')
print(f'  Cached call: {cached_time*1000:.4f}ms')
print(f'  Speedup: {first_time/max(cached_time, 1e-9):.0f}x')
print(f'  Results match: {np.allclose(probs1, probs2)}')

# Test cache invalidation
model.set_random_parameters(seed=123)
probs3 = model.compute_probabilities()
print(f'  After param change: distributions different = {not np.allclose(probs1, probs3)}')

# Test 3: Approximate circuit builder
print('\n3. APPROXIMATE CIRCUIT BUILDER TEST')
print('-' * 40)

model = DiscreteGraphicalModel(8, [{i, i+1} for i in range(7)])
model.set_random_parameters(seed=42)

builder = ApproximateCircuitBuilder(depth=3, entanglement='clique')
circuit, params = builder.build_variational_circuit(model, seed=42)

print(f'  Variables: {model.n_vars}')
print(f'  Circuit depth: {circuit.depth()}')
print(f'  Gate count: {circuit.size()}')
print(f'  Parameters: {len(params)}')
print(f'  Entanglement edges: {len(builder._get_entanglement_edges(model))}')

# Test different entanglement strategies
for ent in ['clique', 'linear', 'full']:
    builder = ApproximateCircuitBuilder(depth=2, entanglement=ent)
    edges = builder._get_entanglement_edges(model)
    print(f'  Entanglement "{ent}": {len(edges)} edges')

# Test 4: Smart circuit builder
print('\n4. SMART CIRCUIT BUILDER TEST')
print('-' * 40)

# Small model - should use exact
model_small = DiscreteGraphicalModel(4, [{0,1}, {1,2}, {2,3}])
model_small.set_random_parameters(seed=42)
circuit_small, info_small = smart_circuit_builder(model_small)
print(f'  Small model (n=4): method={info_small["method"]}')

# Larger model - should use approximate
model_large = DiscreteGraphicalModel(12, [{i, i+1} for i in range(11)])
model_large.set_random_parameters(seed=42)
circuit_large, info_large = smart_circuit_builder(model_large, max_exact_vars=10)
print(f'  Large model (n=12): method={info_large["method"]}, params={info_large["n_params"]}')

# Test 5: Verify correctness of sparse diagonal
print('\n5. CORRECTNESS VERIFICATION')
print('-' * 40)

model = DiscreteGraphicalModel(5, [{0,1}, {1,2}, {2,3,4}])
model.set_random_parameters(seed=42)

# Get diagonal via new method
diag_sparse = model.compute_hamiltonian_diagonal()

# Get diagonal via full matrix
# (temporarily bypass cache)
model._invalidate_cache()
H_full = model.compute_hamiltonian()
diag_full = np.diag(H_full)

match = np.allclose(diag_sparse, diag_full)
print(f'  Sparse diagonal matches full matrix diagonal: {match}')
if not match:
    print(f'  Max difference: {np.max(np.abs(diag_sparse - diag_full))}')

# Test 6: Memory savings calculation
print('\n6. MEMORY SAVINGS')
print('-' * 40)

for n in [8, 10, 12, 15, 20]:
    full_matrix_bytes = (2**n * 2**n) * 8  # float64
    diagonal_bytes = (2**n) * 8
    
    print(f'  n={n:2d}: Full matrix={full_matrix_bytes/1024/1024:>10.1f}MB, Diagonal={diagonal_bytes/1024:>8.1f}KB, Ratio={full_matrix_bytes/diagonal_bytes:>6.0f}x')

print('\n' + '=' * 60)
print('✓ ALL OPTIMIZATION TESTS COMPLETED!')
print('=' * 60)

