"""
SAMPLING METHODS COMPARISON: Generate clean 1×3 comparison figure
Creates sampling_methods_comparison.png (no title, publication ready)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from QuantumDGM import DiscreteGraphicalModel, QCGMSampler, compute_fidelity, estimate_distribution, generate_state_labels

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

def create_demo_model(seed=42):
    """Create a small demo model for visualization"""
    model = DiscreteGraphicalModel(3, [{0, 1}, {1, 2}])
    model.set_random_parameters(low=-2.0, high=-0.5, seed=seed)
    return model

print("=" * 80)
print("   SAMPLING METHODS COMPARISON: Quantum vs Classical")
print("=" * 80)

model = create_demo_model(seed=42)
exact_probs = model.compute_probabilities()
n_vars = model.n_vars
labels = generate_state_labels(n_vars)
n_samples = 2000

# Three methods that KNOW the distribution
# 1. Quantum (amplitude encoding)
sampler = QCGMSampler(model)
q_samples, _ = sampler.sample(n_samples=n_samples)
q_probs = estimate_distribution(q_samples, n_vars)
q_fid = compute_fidelity(exact_probs, q_probs)

# 2. Classical inverse CDF (equivalent!)
states = list(itertools.product([0, 1], repeat=n_vars))
cdf_indices = np.random.choice(len(states), size=n_samples, p=exact_probs)
cdf_samples = np.array([states[i] for i in cdf_indices])
cdf_probs = estimate_distribution(cdf_samples, n_vars)
cdf_fid = compute_fidelity(exact_probs, cdf_probs)

# 3. Rejection sampling (also equivalent!)
max_prob = max(exact_probs)
rej_samples = []
while len(rej_samples) < n_samples:
    idx = np.random.randint(len(states))
    if np.random.random() < exact_probs[idx] / max_prob:
        rej_samples.append(states[idx])
rej_samples = np.array(rej_samples[:n_samples])
rej_probs = estimate_distribution(rej_samples, n_vars)
rej_fid = compute_fidelity(exact_probs, rej_probs)

x = np.arange(len(labels))
width = 0.35

print("\nGenerating 1×3 comparison figure for paper...")

# Main Figure: 1×3 Combined Comparison (NO TITLE)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
fig.patch.set_facecolor('white')

methods = [
    ('Quantum', q_probs, q_fid, '#5e3aff'),
    ('Classical (Inverse-CDF)', cdf_probs, cdf_fid, '#ff6b35'),
    ('Rejection', rej_probs, rej_fid, '#f39c12')
]

for idx, (name, probs, fid, color) in enumerate(methods):
    ax = axes[idx]
    ax.set_facecolor('#f8f9fa')
    
    # Bars
    ax.bar(x - width/2, exact_probs, width, label='Target', 
           color='#28a745', alpha=0.65, edgecolor='black', linewidth=1)
    ax.bar(x + width/2, probs, width, label=name, 
           color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Labels and styling
    ax.set_xlabel('State', fontsize=12, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, max(exact_probs) * 1.2)
    ax.legend(fontsize=10, framealpha=0.95, loc='upper left')  # Changed to upper left
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Fidelity overlay (stays in upper right)
    ax.text(0.98, 0.98, f'F = {fid:.3f}', transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=color, linewidth=2, alpha=0.9))

output_path = os.path.join(FIGURES_DIR, 'sampling_methods_comparison.png')
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
print(f'  ✓ Saved: {output_path}')

plt.show()

print(f"\n  Results ({n_samples} samples):")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │ Quantum (amplitude):     F = {q_fid:.4f}   │")
print(f"  │ Classical (inverse CDF): F = {cdf_fid:.4f}   │")
print(f"  │ Classical (rejection):   F = {rej_fid:.4f}   │")
print(f"  └─────────────────────────────────────────┘")
print(f"\n  ✓ All methods achieve similar fidelity (as expected)")
print(f"\n  ✓ Figure ready for paper (1×3, no title)")
