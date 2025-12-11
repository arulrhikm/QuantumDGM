"""
statistics.py
=============

Pauli-Markov sufficient statistics computation for discrete graphical models.
Implements Algorithm 1 from the paper.
"""

import numpy as np
from typing import Set, Tuple


class PauliMarkovStatistics:
    """
    Computes Pauli-Markov sufficient statistics for graphical models.
    
    The Pauli-Markov sufficient statistics Φ_{C,y} are diagonal matrices
    that encode the sufficient statistics of a graphical model in a form
    suitable for quantum computation.
    """
    
    @staticmethod
    def compute_phi(n: int, clique: Set[int], y: Tuple[int, ...]) -> np.ndarray:
        """
        Compute Φ_{C,y} using Algorithm 1 from the paper.
        
        This function computes a diagonal matrix where each diagonal entry
        corresponds to the indicator function φ_{C,y}(x) for a specific
        configuration x.
        
        The algorithm runs in linear time O(n) by computing Kronecker products
        of 2×2 matrices based on whether each variable is in the clique and
        its assignment.
        
        Args:
            n (int): Number of variables in the graphical model
            clique (Set[int]): Set of variable indices in the clique C
            y (Tuple[int, ...]): Assignment to variables in the clique
                                 (binary values, must match clique size)
        
        Returns:
            np.ndarray: Diagonal matrix Φ_{C,y} of size 2^n × 2^n
        
        Example:
            >>> stats = PauliMarkovStatistics()
            >>> # For 3 variables with clique {0, 1} and assignment (0, 1)
            >>> Phi = stats.compute_phi(3, {0, 1}, (0, 1))
            >>> Phi.shape
            (8, 8)
        
        Note:
            - The matrix is diagonal and sparse
            - (Φ_{C,y})_{j,j} = φ_{C,y}(x_j) where x_j is the j-th configuration
            - φ_{C,y}(x) = ∏_{v∈C} 𝟙{x_v = y_v}
        """
        # Pauli matrices and identity
        I = np.array([[1, 0], [0, 1]], dtype=np.float64)
        Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
        
        # Initialize with scalar 1
        Phi = np.array([[1]], dtype=np.float64)
        
        # Create mapping from clique variables to their assignments
        clique_list = sorted(clique)
        y_dict = {v: y[i] for i, v in enumerate(clique_list)}
        
        # Algorithm 1: Build Φ via Kronecker products
        for v in range(n):
            if v not in clique:
                # Variable not in clique: tensor with identity
                Phi = np.kron(Phi, I)
            elif y_dict[v] == 1:
                # Variable in clique with assignment 1: tensor with (I - Z)/2
                Phi = np.kron(Phi, (I - Z) / 2)
            else:  # y_dict[v] == 0
                # Variable in clique with assignment 0: tensor with (I + Z)/2
                Phi = np.kron(Phi, (I + Z) / 2)
        
        return Phi
    
    @staticmethod
    def verify_sufficient_statistic(Phi: np.ndarray, n: int, 
                                    clique: Set[int], y: Tuple[int, ...]) -> bool:
        """
        Verify that a Pauli-Markov sufficient statistic matrix is correct.
        
        This checks that the diagonal entries of Φ_{C,y} match the indicator
        function φ_{C,y}(x) for all configurations x.
        
        Args:
            Phi: The matrix to verify
            n: Number of variables
            clique: The clique
            y: The assignment
        
        Returns:
            bool: True if the matrix is correct
        """
        clique_list = sorted(clique)
        y_dict = {v: y[i] for i, v in enumerate(clique_list)}
        
        for j in range(2 ** n):
            # Extract binary configuration for state j
            config = tuple((j >> i) & 1 for i in range(n))
            
            # Compute expected value of φ_{C,y}(config)
            expected = 1
            for v in clique:
                if config[v] != y_dict[v]:
                    expected = 0
                    break
            
            # Check diagonal entry
            if not np.isclose(Phi[j, j], expected):
                return False
        
        return True
    
    @staticmethod
    def compute_all_phi_for_clique(n: int, clique: Set[int]) -> dict:
        """
        Compute all Pauli-Markov statistics for a given clique.
        
        For a clique of size k, this computes 2^k matrices Φ_{C,y},
        one for each possible assignment y.
        
        Args:
            n: Number of variables
            clique: The clique
        
        Returns:
            dict: Mapping from assignment tuple y to matrix Φ_{C,y}
        """
        import itertools
        
        clique_size = len(clique)
        phi_dict = {}
        
        for y in itertools.product([0, 1], repeat=clique_size):
            phi_dict[y] = PauliMarkovStatistics.compute_phi(n, clique, y)
        
        return phi_dict