"""
model.py
========================================

Discrete graphical model representation and exact inference.
Implements the exponential family formulation from Equations (1)-(2).
"""

import numpy as np
import itertools
from typing import List, Set, Dict, Tuple
from .statistics import PauliMarkovStatistics


class DiscreteGraphicalModel:
    """
    Represents a discrete graphical model over binary variables.
    
    The model is defined by:
    - P_θ(X = x) = (1/Z(θ)) exp(∑_{C∈𝒞} ∑_{y∈𝒳_C} θ_{C,y} φ_{C,y}(x))
    
    where:
    - 𝒞 is the set of maximal cliques
    - θ are the canonical parameters
    - φ are the sufficient statistics
    - Z(θ) is the partition function
    
    This implements an overcomplete exponential family representation
    as described in Section 2.1 of the paper.
    """
    
    def __init__(self, n_vars: int, cliques: List[Set[int]]):
        """
        Initialize a graphical model.
        
        Args:
            n_vars (int): Number of binary variables (n)
            cliques (List[Set[int]]): List of maximal cliques
                                      Each clique is a set of variable indices
        
        Example:
            >>> # Chain model: v0 - v1 - v2
            >>> model = DiscreteGraphicalModel(3, [{0, 1}, {1, 2}])
            
            >>> # Star model: v0 connected to v1, v2, v3
            >>> model = DiscreteGraphicalModel(4, [{0, 1}, {0, 2}, {0, 3}])
        """
        self.n_vars = n_vars
        self.cliques = cliques
        self.n_cliques = len(cliques)
        
        # Initialize parameters θ_{C,y} for each (clique, assignment) pair
        self.theta = {}
        for clique in cliques:
            clique_tuple = tuple(sorted(clique))
            clique_size = len(clique)
            
            # Initialize parameters for all 2^|C| assignments
            for y in itertools.product([0, 1], repeat=clique_size):
                self.theta[(clique_tuple, y)] = 0.0
    
    def set_parameters(self, theta_dict: Dict[Tuple, float]):
        """
        Set model parameters from a dictionary.
        
        Args:
            theta_dict: Dictionary mapping (clique_tuple, assignment) to parameter value
        
        Example:
            >>> model = DiscreteGraphicalModel(2, [{0, 1}])
            >>> params = {
            ...     ((0, 1), (0, 0)): -1.5,
            ...     ((0, 1), (0, 1)): -2.0,
            ...     ((0, 1), (1, 0)): -1.8,
            ...     ((0, 1), (1, 1)): -0.5
            ... }
            >>> model.set_parameters(params)
        """
        self.theta.update(theta_dict)
    
    def set_random_parameters(self, low: float = -5.0, high: float = 0.0, seed: int = None):
        """
        Set random parameters uniformly in [low, high).
        
        Due to shift invariance of overcomplete families, we can restrict
        parameters to be negative without loss of generality.
        
        Args:
            low: Lower bound (default: -5.0)
            high: Upper bound (default: 0.0)
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)
        
        for key in self.theta.keys():
            self.theta[key] = np.random.uniform(low, high)
    
    def compute_hamiltonian(self) -> np.ndarray:
        """
        Compute the Hamiltonian H_θ (Theorem 3.3).
        
        H_θ = -∑_{C∈𝒞} ∑_{y∈𝒳_C} θ_{C,y} Φ_{C,y}
        
        The diagonal of H_θ contains -θ^T φ(x) for each configuration x.
        
        Returns:
            np.ndarray: Hamiltonian matrix of size 2^n × 2^n
        
        Note:
            - H_θ is diagonal (and thus self-adjoint)
            - H_θ is NOT unitary (eigenvalues are not on unit circle)
            - exp(-H_θ) gives unnormalized probabilities
        """
        dim = 2 ** self.n_vars
        H = np.zeros((dim, dim), dtype=np.float64)
        
        # Sum over all cliques and assignments
        for clique in self.cliques:
            clique_tuple = tuple(sorted(clique))
            clique_size = len(clique)
            
            for y in itertools.product([0, 1], repeat=clique_size):
                theta_cy = self.theta[(clique_tuple, y)]
                Phi_cy = PauliMarkovStatistics.compute_phi(self.n_vars, clique, y)
                H -= theta_cy * Phi_cy
        
        return H
    
    def compute_probabilities(self) -> np.ndarray:
        """
        Compute exact probabilities using the Hamiltonian.
        
        P_θ(x_j) = exp(-H_θ)_{j,j} / Tr(exp(-H_θ))
        
        Returns:
            np.ndarray: Probability distribution over all 2^n states
        
        Example:
            >>> model = DiscreteGraphicalModel(2, [{0, 1}])
            >>> model.set_random_parameters()
            >>> probs = model.compute_probabilities()
            >>> print(f"Sum: {probs.sum():.6f}")  # Should be 1.0
            >>> print(f"Shape: {probs.shape}")     # (4,)
        """
        H = self.compute_hamiltonian()
        
        # Since H is diagonal, we can compute exp(-H) efficiently
        # Extract diagonal as a 1D array for NumPy 2.0 compatibility
        diag_H = np.asarray(np.diagonal(H), dtype=np.float64).copy()
        unnormalized = np.exp(-diag_H)
        
        # Normalize to get probabilities
        # Use .sum() method and ensure it's a scalar
        Z = float(unnormalized.sum())
        if Z == 0:
            # Fallback to uniform if all probabilities are zero
            return np.ones(len(unnormalized)) / len(unnormalized)
        return unnormalized / Z
    
    def compute_partition_function(self) -> float:
        """
        Compute the partition function Z(θ).
        
        Z(θ) = ∑_x exp(θ^T φ(x))
        
        Returns:
            float: Partition function value
        """
        H = self.compute_hamiltonian()
        diag_H = np.asarray(np.diagonal(H), dtype=np.float64).copy()
        exp_vals = np.exp(-diag_H)
        return float(exp_vals.sum())
    
    def compute_log_partition_function(self) -> float:
        """
        Compute log Z(θ) in a numerically stable way.
        
        Returns:
            float: log(Z(θ))
        """
        H = self.compute_hamiltonian()
        log_probs = -np.asarray(np.diagonal(H), dtype=np.float64).copy()
        max_log_prob = float(log_probs.max())
        shifted = log_probs - max_log_prob
        exp_shifted = np.exp(shifted)
        log_sum = float(np.log(exp_shifted.sum()))
        return max_log_prob + log_sum
    
    def compute_entropy(self) -> float:
        """
        Compute the entropy H(P_θ) = -∑_x P_θ(x) log P_θ(x).
        
        Returns:
            float: Entropy in nats
        """
        probs = self.compute_probabilities()
        # Ensure probs is a proper array
        probs = np.asarray(probs, dtype=np.float64)
        # Avoid log(0) with small epsilon
        log_probs = np.log(probs + 1e-10)
        # Use explicit multiplication and sum
        prod = probs * log_probs
        entropy = -float(prod.sum())
        return entropy
    
    def sample_exact(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate exact samples using the true distribution.
        
        This uses inverse transform sampling with the exact probabilities.
        Serves as ground truth for comparing quantum sampling.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            np.ndarray: Array of samples, shape (n_samples, n_vars)
                       Each row is a binary configuration
        
        Example:
            >>> model = DiscreteGraphicalModel(3, [{0, 1}, {1, 2}])
            >>> model.set_random_parameters()
            >>> samples = model.sample_exact(1000)
            >>> samples.shape
            (1000, 3)
        """
        probs = self.compute_probabilities()
        
        # Generate all possible states
        states = list(itertools.product([0, 1], repeat=self.n_vars))
        
        # Sample from discrete distribution
        indices = np.random.choice(len(states), size=n_samples, p=probs)
        
        return np.array([states[i] for i in indices])
    
    def compute_marginals(self, variable_set: Set[int]) -> np.ndarray:
        """
        Compute marginal probabilities for a subset of variables.
        
        P(X_S = x_S) = ∑_{x_{̄S}} P(X = x)
        
        Args:
            variable_set: Set of variable indices to marginalize over
        
        Returns:
            np.ndarray: Marginal distribution
        """
        joint_probs = self.compute_probabilities()
        # Ensure it's a proper array
        joint_probs = np.asarray(joint_probs, dtype=np.float64)
        
        var_list = sorted(variable_set)
        n_marginal = len(var_list)
        
        marginal = np.zeros(2 ** n_marginal, dtype=np.float64)
        
        for state_idx, prob in enumerate(joint_probs):
            # Extract configuration
            config = tuple((state_idx >> i) & 1 for i in range(self.n_vars))
            
            # Project to marginal variables
            marginal_config = tuple(config[v] for v in var_list)
            marginal_idx = sum(bit << i for i, bit in enumerate(marginal_config))
            
            marginal[marginal_idx] += prob
        
        return marginal
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"DiscreteGraphicalModel(n_vars={self.n_vars}, "
                f"n_cliques={self.n_cliques})")