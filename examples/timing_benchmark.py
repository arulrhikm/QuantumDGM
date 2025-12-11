"""
Wall-Clock Timing Benchmark for Honest Comparison
==================================================

Measures actual sampling times for:
1. Quantum (amplitude encoding)
2. Classical (inverse-CDF)
3. Rejection sampling  
4. Gibbs (after burn-in)
5. Gibbs (from scratch)

Generates data for Table in Section 5.3.1
"""

import numpy as np
import time
from QuantumDGM import DiscreteGraphicalModel, QCGMSampler, estimate_distribution
import itertools

print('=' * 80)
print('WALL-CLOCK TIMING BENCHMARKS')
print('=' * 80)

# Test configuration
n_vars = 8
n_samples = 1000
n_trials = 5  # Average over multiple trials for stability

# Create test model (chain graph)
model = DiscreteGraphicalModel(n_vars, [{i, i+1} for i in range(n_vars-1)])
model.set_random_parameters(low=-2.0, high=-0.5, seed=42)

# Pre-compute probabilities (needed by all methods)
print(f'\nPre-computing distribution for n={n_vars}...')
start = time.time()
exact_probs = model.compute_probabilities()
precompute_time = time.time() - start
print(f'  Preprocessing time: {precompute_time:.3f}s (O(2^n) cost)')

# Generate state list for classical methods
states = list(itertools.product([0, 1], repeat=n_vars))

results = {}

# ============================================================================
# 1. QUANTUM (Amplitude Encoding)
# ============================================================================
print(f'\n{"="*80}')
print('1. Quantum (Amplitude Encoding)')
print('='*80)

quantum_times = []
for trial in range(n_trials):
    sampler = QCGMSampler(model)
    
    start = time.time()
    samples,_ = sampler.sample(n_samples=n_samples, simplified=True)
    elapsed = time.time() - start
    
    quantum_times.append(elapsed)
    print(f'  Trial {trial+1}: {elapsed:.3f}s ({len(samples)} samples)')

quantum_mean = np.mean(quantum_times)
quantum_std = np.std(quantum_times)
results['quantum'] = {'mean': quantum_mean, 'std': quantum_std, 
                      'note': 'Circuit + measurement'}
print(f'\nQuantum: {quantum_mean:.3f}s ± {quantum_std:.3f}s')

# ============================================================================
# 2. CLASSICAL (Inverse-CDF)
# ============================================================================
print(f'\n{"="*80}')
print('2. Classical (Inverse-CDF)')
print('='*80)

# Build CDF once (part of O(2^n) preprocessing)
cdf = np.cumsum(exact_probs)

cdf_times = []
for trial in range(n_trials):
    start = time.time()
    # Sample using inverse CDF
    uniform_samples = np.random.random(n_samples)
    indices = np.searchsorted(cdf, uniform_samples)
    samples = np.array([states[i] for i in indices])
    elapsed = time.time() - start
    cdf_times.append(elapsed)
    print(f'  Trial {trial+1}: {elapsed:.3f}s')

cdf_mean = np.mean(cdf_times)
cdf_std = np.std(cdf_times)
results['cdf'] = {'mean': cdf_mean, 'std': cdf_std,
                  'note': 'After $2^n$ preprocessing'}
print(f'\nInverse-CDF: {cdf_mean:.3f}s ± {cdf_std:.3f}s')

# ============================================================================
# 3. REJECTION SAMPLING
# ============================================================================
print(f'\n{"="*80}')
print('3. Rejection Sampling')
print('='*80)

max_prob = np.max(exact_probs)
acceptance_rate_total = 0

rej_times = []
for trial in range(n_trials):
    start = time.time()
    samples = []
    proposals = 0
    
    while len(samples) < n_samples:
        idx = np.random.randint(len(states))
        proposals += 1
        if np.random.random() < exact_probs[idx] / max_prob:
            samples.append(states[idx])
    
    elapsed = time.time() - start
    acceptance_rate = n_samples / proposals
    acceptance_rate_total += acceptance_rate
    rej_times.append(elapsed)
    print(f'  Trial {trial+1}: {elapsed:.3f}s (acceptance: {acceptance_rate*100:.1f}%)')

rej_mean = np.mean(rej_times)
rej_std = np.std(rej_times)
avg_acceptance = (acceptance_rate_total / n_trials) * 100
results['rejection'] = {'mean': rej_mean, 'std': rej_std,
                        'note': f'{avg_acceptance:.1f}% acceptance rate'}
print(f'\nRejection: {rej_mean:.3f}s ± {rej_std:.3f}s')

# ============================================================================
# 4. GIBBS SAMPLING (Simple implementation)
# ============================================================================
print(f'\n{"="*80}')
print('4. Gibbs Sampling (Simple Markov Chain)')
print('='*80)

def simple_gibbs(n_vars, exact_probs, states, n_total, n_burn_in=1000):
    """Simple Gibbs with direct probability lookup"""
    state = np.random.randint(0, 2, n_vars)
    samples = []
    
    # Create lookup dict
    prob_dict = {tuple(s): p for s, p in zip(states, exact_probs)}
    
    for iter_num in range(n_burn_in + n_total):
        for i in range(n_vars):
            # Try both values for variable i
            state[i] = 0
            prob_0 = prob_dict.get(tuple(state), 1e-10)
            state[i] = 1
            prob_1 = prob_dict.get(tuple(state), 1e-10)
            
            # Sample proportionally
            p = prob_1 / (prob_0 + prob_1 + 1e-10)
            state[i] = 1 if np.random.random() < p else 0
        
        # Collect after burn-in
        if iter_num >= n_burn_in:
            samples.append(state.copy())
    
    return np.array(samples)

# Gibbs AFTER burn-in (pre-warmed)
print('\n4a. Gibbs (after burn-in)')
gibbs_after_times = []
for trial in range(n_trials):
    # Pre-burn (not counted)
    _ = simple_gibbs(n_vars, exact_probs, states, 1, n_burn_in=1000)
    
    # Now measure sampling time AFTER burn-in
    start = time.time()
    samples = simple_gibbs(n_vars, exact_probs, states, n_samples, n_burn_in=0)
    elapsed = time.time() - start
    gibbs_after_times.append(elapsed)
    print(f'  Trial {trial+1}: {elapsed:.3f}s')

gibbs_after_mean = np.mean(gibbs_after_times)
gibbs_after_std = np.std(gibbs_after_times)
results['gibbs_after'] = {'mean': gibbs_after_mean, 'std': gibbs_after_std,
                          'note': '1000 burn-in + 1000 samples'}
print(f'\nGibbs (after burn-in): {gibbs_after_mean:.3f}s ± {gibbs_after_std:.3f}s')

# Gibbs FROM SCRATCH (includes burn-in cost)
print('\n4b. Gibbs (from scratch)')
gibbs_full_times = []
for trial in range(n_trials):
    start = time.time()
    # Includes burn-in cost
    samples = simple_gibbs(n_vars, exact_probs, states, n_samples, n_burn_in=1000)
    elapsed = time.time() - start
    gibbs_full_times.append(elapsed)
    print(f'  Trial {trial+1}: {elapsed:.3f}s')

gibbs_full_mean = np.mean(gibbs_full_times)
gibbs_full_std = np.std(gibbs_full_times)
results['gibbs_full'] = {'mean': gibbs_full_mean, 'std': gibbs_full_std,
                         'note': 'Includes burn-in cost'}
print(f'\nGibbs (from scratch): {gibbs_full_mean:.3f}s ± {gibbs_full_std:.3f}s')

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print(f'\n{"="*80}')
print('SUMMARY: Wall-Clock Times for 1000 Samples (n=8)')
print('='*80)

print(f'\n{"Method":<30} | {"Time (s)":<12} | {"Notes"}')
print('-' * 80)
print(f'{"Quantum (amplitude)":<30} | {quantum_mean:>5.2f} ± {quantum_std:>4.2f} | {results["quantum"]["note"]}')
print(f'{"Classical (inverse-CDF)":<30} | {cdf_mean:>5.2f} ± {cdf_std:>4.2f} | {results["cdf"]["note"]}')
print(f'{"Rejection sampling":<30} | {rej_mean:>5.2f} ± {rej_std:>4.2f} | {results["rejection"]["note"]}')
print(f'{"Gibbs (after burn-in)":<30} | {gibbs_after_mean:>5.2f} ± {gibbs_after_std:>4.2f} | {results["gibbs_after"]["note"]}')
print(f'{"Gibbs (from scratch)":<30} | {gibbs_full_mean:>5.2f} ± {gibbs_full_std:>4.2f} | {results["gibbs_full"]["note"]}')

print(f'\nPreprocessing (all methods): {precompute_time:.2f}s')

# LaTeX table for paper
print(f'\n{"="*80}')
print('LaTeX TABLE CODE:')
print('='*80)
print(r"""
\begin{table}[h]
\centering
\caption{Wall-clock sampling time comparison (1000 samples, n=8 chain).}
\label{tab:timing}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Time (s)} & \textbf{Notes} \\
\midrule
""" + f"""Quantum (amplitude)       & {quantum_mean:.2f} $\\pm$ {quantum_std:.2f} & Circuit + measurement \\\\
Classical (inverse-CDF)   & {cdf_mean:.2f} $\\pm$ {cdf_std:.2f} & After $2^n$ preprocessing \\\\
Rejection sampling        & {rej_mean:.2f} $\\pm$ {rej_std:.2f} & {avg_acceptance:.1f}\\% acceptance rate \\\\
Gibbs (after burn-in)     & {gibbs_after_mean:.2f} $\\pm$ {gibbs_after_std:.2f} & 1000 burn-in + 1000 samples \\\\
Gibbs (from scratch)      & {gibbs_full_mean:.2f} $\\pm$ {gibbs_full_std:.2f} & Includes burn-in cost \\\\
""" + r"""\midrule
\multicolumn{3}{l}{\textit{Preprocessing (all methods): """ + f"{precompute_time:.2f}s" + r"""}} \\
\bottomrule
\end{tabular}
\end{table}
""")

print('\n✓ Benchmark complete! Use table above in paper Section 5.3.1')
