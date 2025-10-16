"""
Placeholder for inference methods:
- MAP estimation
- Parameter learning
- Partition function estimation
"""
import numpy as np

def map_estimation(H):
    """Return the index of the smallest eigenvalue (MAP state)."""
    vals, vecs = np.linalg.eigh(H)
    min_idx = np.argmin(vals)
    return min_idx, vecs[:, min_idx]

def estimate_partition_function(H):
    """Compute approximate partition function via trace(exp(-H))."""
    eigvals = np.linalg.eigvals(H)
    return np.sum(np.exp(-eigvals.real))
