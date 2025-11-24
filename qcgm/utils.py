"""
utils.py - FIXED VERSION
=========================

Utility functions for comparing distributions and analyzing results.

Fixed: NumPy 2.0 compatibility issues
"""

import numpy as np
import itertools
from typing import Union


def compute_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute fidelity between two probability distributions.
    
    The fidelity is defined as:
    F(P, Q) = (∑_x √(P(x)Q(x)))²
    
    This is a common measure in quantum information theory and is used
    in the paper to assess the quality of quantum sampling (Section 6.1).
    
    Properties:
    - F(P, Q) ∈ [0, 1]
    - F(P, Q) = 1 if and only if P = Q
    - F(P, Q) = 0 if P and Q have disjoint support
    - Related to Hellinger distance: H²(P, Q) = 1 - √F(P, Q)
    
    Args:
        p (np.ndarray): First probability distribution
        q (np.ndarray): Second probability distribution
    
    Returns:
        float: Fidelity value in [0, 1]
    
    Example:
        >>> p = np.array([0.5, 0.3, 0.2])
        >>> q = np.array([0.6, 0.25, 0.15])
        >>> f = compute_fidelity(p, q)
        >>> print(f"Fidelity: {f:.4f}")
    
    Note:
        Input arrays should be normalized probability distributions.
        The function does not check for normalization.
    """
    if len(p) != len(q):
        raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")
    
    # Compute fidelity: (∑ √(p_i * q_i))²
    sqrt_products = np.sqrt(p * q)
    fidelity = float(np.sum(sqrt_products)) ** 2
    
    return fidelity


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Hellinger distance between two probability distributions.
    
    H(P, Q) = √(1 - √F(P, Q))
    
    where F is the fidelity. This is mentioned in Section 6.1 of the paper.
    
    Args:
        p: First distribution
        q: Second distribution
    
    Returns:
        float: Hellinger distance in [0, 1]
    """
    fidelity = compute_fidelity(p, q)
    return float(np.sqrt(1 - np.sqrt(fidelity)))


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler divergence from Q to P.
    
    KL(P || Q) = ∑_x P(x) log(P(x) / Q(x))
    
    Args:
        p: Target distribution
        q: Approximate distribution
        epsilon: Small constant to avoid log(0)
    
    Returns:
        float: KL divergence (non-negative, not symmetric)
    
    Note:
        KL divergence is not a true distance metric (not symmetric).
        Returns infinity if there exists x where p[x] > 0 but q[x] = 0.
    """
    if len(p) != len(q):
        raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")
    
    # Add epsilon to avoid log(0)
    p_safe = p + epsilon
    q_safe = q + epsilon
    
    log_ratio = np.log(p_safe / q_safe)
    kl = float(np.sum(p_safe * log_ratio))
    
    return kl


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute total variation distance between two distributions.
    
    TV(P, Q) = (1/2) ∑_x |P(x) - Q(x)|
    
    Args:
        p: First distribution
        q: Second distribution
    
    Returns:
        float: Total variation distance in [0, 1]
    """
    abs_diff = np.abs(p - q)
    tv = 0.5 * float(np.sum(abs_diff))
    return tv


def estimate_distribution(samples: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Estimate probability distribution from samples.
    
    Computes empirical frequencies for all 2^n possible states.
    Used to convert samples from quantum circuit into a distribution
    for comparison with the exact model.
    
    Args:
        samples (np.ndarray): Array of samples, shape (n_samples, n_vars)
                             Each row is a binary configuration
        n_vars (int): Number of variables
    
    Returns:
        np.ndarray: Estimated probability distribution of length 2^n
    
    Example:
        >>> samples = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
        >>> dist = estimate_distribution(samples, n_vars=2)
        >>> print(dist)  # [0.5, 0.25, 0.0, 0.25] for states 00, 01, 10, 11
    
    Note:
        Returns uniform distribution if no samples provided.
    """
    num_states = 2 ** n_vars
    
    if len(samples) == 0:
        # Return uniform distribution if no samples
        return np.ones(num_states, dtype=np.float64) / num_states
    
    # Count occurrences of each state
    counts = np.zeros(num_states, dtype=np.float64)
    
    # Generate all possible states for indexing
    states = list(itertools.product([0, 1], repeat=n_vars))
    
    for sample in samples:
        # Convert sample to state index
        state_tuple = tuple(sample)
        state_idx = states.index(state_tuple)
        counts[state_idx] += 1
    
    # Normalize to get probabilities
    total = float(len(samples))
    return counts / total


def estimate_distribution_fast(samples: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Fast version of estimate_distribution using bit operations.
    
    More efficient for large numbers of samples.
    
    Args:
        samples: Sample array
        n_vars: Number of variables
    
    Returns:
        np.ndarray: Estimated distribution
    """
    num_states = 2 ** n_vars
    
    if len(samples) == 0:
        return np.ones(num_states, dtype=np.float64) / num_states
    
    counts = np.zeros(num_states, dtype=np.float64)
    
    # Convert each sample to integer index using bit operations
    for sample in samples:
        idx = sum(int(bit) << i for i, bit in enumerate(sample))
        counts[idx] += 1
    
    total = float(len(samples))
    return counts / total


def compare_distributions(p: np.ndarray, q: np.ndarray, 
                         labels: tuple = ('P', 'Q')) -> dict:
    """
    Compute multiple distance/similarity measures between distributions.
    
    Args:
        p: First distribution
        q: Second distribution
        labels: Names for the distributions (for printing)
    
    Returns:
        dict: Dictionary with all computed metrics
    
    Example:
        >>> metrics = compare_distributions(exact_probs, quantum_probs,
        ...                                 labels=('Exact', 'Quantum'))
        >>> print(f"Fidelity: {metrics['fidelity']:.4f}")
    """
    max_err = float(np.max(np.abs(p - q)))
    mean_err = float(np.mean(np.abs(p - q)))
    
    metrics = {
        'fidelity': compute_fidelity(p, q),
        'hellinger': hellinger_distance(p, q),
        'kl_divergence': kl_divergence(p, q),
        'total_variation': total_variation_distance(p, q),
        'max_absolute_error': max_err,
        'mean_absolute_error': mean_err,
        'labels': labels
    }
    
    return metrics


def print_comparison(metrics: dict):
    """
    Pretty print distribution comparison metrics.
    
    Args:
        metrics: Dictionary from compare_distributions()
    """
    p_label, q_label = metrics['labels']
    
    print("=" * 60)
    print(f"Distribution Comparison: {p_label} vs {q_label}")
    print("=" * 60)
    print(f"Fidelity (F):              {metrics['fidelity']:.6f}")
    print(f"Hellinger distance (H):    {metrics['hellinger']:.6f}")
    print(f"KL divergence (KL):        {metrics['kl_divergence']:.6f}")
    print(f"Total variation (TV):      {metrics['total_variation']:.6f}")
    print(f"Max absolute error:        {metrics['max_absolute_error']:.6f}")
    print(f"Mean absolute error:       {metrics['mean_absolute_error']:.6f}")
    print("=" * 60)


def sample_statistics(samples: np.ndarray) -> dict:
    """
    Compute statistics about a sample set.
    
    Args:
        samples: Array of samples
    
    Returns:
        dict: Statistics including counts, entropy estimate, etc.
    """
    if len(samples) == 0:
        return {
            'n_samples': 0,
            'n_vars': 0,
            'n_unique': 0,
            'empirical_entropy': 0.0
        }
    
    n_samples, n_vars = samples.shape
    
    # Count unique samples
    unique_samples = np.unique(samples, axis=0)
    n_unique = len(unique_samples)
    
    # Estimate entropy from empirical distribution
    dist = estimate_distribution(samples, n_vars)
    log_dist = np.log2(dist + 1e-10)
    empirical_entropy = float(-np.sum(dist * log_dist))
    
    return {
        'n_samples': n_samples,
        'n_vars': n_vars,
        'n_unique': n_unique,
        'empirical_entropy': empirical_entropy,
        'max_entropy': n_vars  # Maximum possible entropy for n binary variables
    }


def generate_state_labels(n_vars: int) -> list:
    """
    Generate binary state labels for plotting.
    
    Args:
        n_vars: Number of variables
    
    Returns:
        list: List of binary strings like ['000', '001', '010', ...]
    
    Example:
        >>> labels = generate_state_labels(3)
        >>> print(labels)
        ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [format(i, f'0{n_vars}b') for i in range(2 ** n_vars)]


def confidence_interval(p_hat: float, n: int, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for a proportion using normal approximation.
    
    Useful for estimating uncertainty in success rates and probabilities.
    
    Args:
        p_hat: Estimated proportion
        n: Sample size
        confidence: Confidence level (default: 0.95)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    from scipy import stats
    
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margin = z * se
    
    lower = max(0.0, p_hat - margin)
    upper = min(1.0, p_hat + margin)
    
    return (float(lower), float(upper))