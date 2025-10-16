"""
Simple representation of a discrete graphical model structure
with visualization and classical sampling utilities.
"""

from typing import List, Dict, Tuple
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt


class GraphicalModel:
    def __init__(self, n_vars: int, cliques: List[List[int]], theta: List[float]):
        """
        Args:
            n_vars (int): Number of variables (binary)
            cliques (List[List[int]]): List of maximal cliques (each a list of node indices)
            theta (List[float]): Parameters (weights) per clique
        """
        self.n = n_vars
        self.cliques = cliques
        self.theta = np.array(theta, dtype=float)

    # -------------------------------------------------------------------------
    #  Basic info
    # -------------------------------------------------------------------------
    def describe(self):
        """Print a human-readable summary of the model."""
        print(f"Graphical Model with {self.n} variables")
        for i, C in enumerate(self.cliques):
            print(f"  Clique {i}: {C}, θ={self.theta[i]}")

    # -------------------------------------------------------------------------
    #  Visualization
    # -------------------------------------------------------------------------
    def visualize(self):
        """Visualize the graph structure using NetworkX."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for C in self.cliques:
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    G.add_edge(C[i], C[j])

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(5, 4))
        nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=700, font_weight="bold")
        plt.title("Graphical Model Structure")
        plt.show()

    # -------------------------------------------------------------------------
    #  Classical Sampling (for correctness testing)
    # -------------------------------------------------------------------------
    def energy(self, x: np.ndarray) -> float:
        """
        Compute unnormalized negative energy: -θᵀφ(x)
        where φ(x) = indicator that all clique bits are 1.
        """
        energy = 0.0
        for i, C in enumerate(self.cliques):
            # simple potential: clique contributes if all variables are 1
            if np.all(x[C] == 1):
                energy -= self.theta[i]
        return energy

    def prob_distribution(self) -> Dict[str, float]:
        """
        Compute normalized probability distribution P(x) over all 2^n states.
        Returns a dictionary {bitstring: probability}.
        """
        all_states = list(itertools.product([0, 1], repeat=self.n))
        unnormalized = np.array([np.exp(-self.energy(np.array(s))) for s in all_states])
        Z = np.sum(unnormalized)
        probs = unnormalized / Z
        return {"".join(map(str, s)): p for s, p in zip(all_states, probs)}

    def sample(self, n_samples: int = 1000) -> Dict[str, int]:
        """
        Draw n_samples from the classical model using the full probability table.
        Returns counts {bitstring: count}.
        """
        probs_dict = self.prob_distribution()
        states = list(probs_dict.keys())
        probs = np.array(list(probs_dict.values()))

        samples = np.random.choice(states, size=n_samples, p=probs)
        counts = {s: int(np.sum(samples == s)) for s in states}
        return counts