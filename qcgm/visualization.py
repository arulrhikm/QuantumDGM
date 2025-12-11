"""
visualization.py
================

Visualization utilities for quantum circuits and graphical models.

"""

import numpy as np
from typing import Optional, Tuple, List

# Check for required dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


def _check_dependencies():
    """Check if visualization dependencies are available."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
    if not _HAS_NETWORKX:
        raise ImportError(
            "networkx is required for graph visualization. "
            "Install with: pip install networkx"
        )


def visualize_graphical_model(model, 
                              title: str = "Graphical Model Structure", 
                              figsize: Tuple[float, float] = (10, 6), 
                              save_path: Optional[str] = None):
    """
    Visualize the structure of a graphical model.
    
    Args:
        model: DiscreteGraphicalModel instance
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> from qcgm import DiscreteGraphicalModel
        >>> from qcgm.visualization import visualize_graphical_model
        >>> model = DiscreteGraphicalModel(3, [{0, 1}, {1, 2}])
        >>> fig = visualize_graphical_model(model, "Chain Model")
        >>> plt.show()
    """
    _check_dependencies()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(model.n_vars))
    
    # Add edges from cliques
    for clique in model.cliques:
        clique_list = list(clique)
        for i in range(len(clique_list)):
            for j in range(i+1, len(clique_list)):
                G.add_edge(clique_list[i], clique_list[j])
    
    # Layout
    if model.n_vars <= 3:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#3498db', 
                           node_size=2000, alpha=0.9, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=3, alpha=0.6, 
                           edge_color='#2c3e50', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           labels={i: f'v{i}' for i in range(model.n_vars)},
                           font_size=16, font_weight='bold', 
                           font_color='white', ax=ax)
    
    # Add clique information
    clique_text = "Cliques: " + ", ".join(["{" + ", ".join([f"v{v}" for v in sorted(c)]) + "}" 
                                           for c in model.cliques])
    ax.text(0.5, -0.15, clique_text, transform=ax.transAxes,
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_circuit_diagram(model, 
                              sampler=None, 
                              title: str = "Quantum Circuit",
                              figsize: Tuple[float, float] = (12, 4), 
                              save_path: Optional[str] = None):
    """
    Create a simplified visualization of the quantum circuit.
    
    Args:
        model: DiscreteGraphicalModel instance
        sampler: QCGMSampler instance (optional, for circuit stats)
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> from qcgm import DiscreteGraphicalModel, QCGMSampler
        >>> from qcgm.visualization import visualize_circuit_diagram
        >>> model = DiscreteGraphicalModel(2, [{0, 1}])
        >>> sampler = QCGMSampler(model)
        >>> fig = visualize_circuit_diagram(model, sampler)
        >>> plt.show()
    """
    _check_dependencies()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_qubits = model.n_vars
    
    # Draw qubit lines
    for i in range(n_qubits):
        ax.plot([0, 10], [i, i], 'k-', linewidth=2)
        ax.text(-0.5, i, f'q[{i}]', fontsize=12, ha='right', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(10.5, i, f'c[{i}]', fontsize=12, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # State initialization box
    init_box = FancyBboxPatch((0.5, -0.4), 1.5, n_qubits-0.2,
                              boxstyle="round,pad=0.1", 
                              edgecolor='#e74c3c', facecolor='#e74c3c',
                              alpha=0.3, linewidth=2)
    ax.add_patch(init_box)
    ax.text(1.25, n_qubits/2 - 0.5, 'Initialize\nState', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Measurement symbols
    for i in range(n_qubits):
        # Measurement box
        measure_box = FancyBboxPatch((8.5, i-0.3), 0.8, 0.6,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='#2ecc71', facecolor='white',
                                     linewidth=2)
        ax.add_patch(measure_box)
        
        # Measurement symbol (arc)
        theta = np.linspace(0, np.pi, 30)
        x_arc = 8.9 + 0.2 * np.cos(theta)
        y_arc = i + 0.2 * np.sin(theta)
        ax.plot(x_arc, y_arc, 'k-', linewidth=1.5)
        ax.plot([8.9, 9.1], [i, i+0.2], 'k-', linewidth=1.5)
        
        # Classical bit connection
        ax.plot([9.3, 10], [i, i], 'k--', linewidth=2)
    
    # Add annotations
    ax.text(5, n_qubits + 0.5, 
           f'Circuit: {n_qubits} qubits, simplified version',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Get actual circuit stats if sampler provided
    if sampler is not None:
        stats = sampler.get_circuit_stats()
        info_text = f"Depth: {stats['depth']} | Gates: {stats['size']}"
    else:
        info_text = f"Qubits: {n_qubits}"
    
    ax.text(5, -1.2, info_text, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1.5, n_qubits + 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_model_structures(figsize: Tuple[float, float] = (14, 10), 
                             save_path: Optional[str] = None):
    """
    Visualize different graphical model structures side-by-side.
    
    Args:
        figsize: Figure size as (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> from qcgm.visualization import compare_model_structures
        >>> fig = compare_model_structures()
        >>> plt.show()
    """
    _check_dependencies()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Common Graphical Model Structures', 
                fontsize=16, fontweight='bold')
    
    structures = [
        ("Independent (no edges)", 3, []),
        ("Chain (v0-v1-v2)", 3, [{0, 1}, {1, 2}]),
        ("Star (v0 center)", 4, [{0, 1}, {0, 2}, {0, 3}]),
        ("Complete (all connected)", 4, [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]),
    ]
    
    for idx, (title, n_vars, cliques) in enumerate(structures):
        ax = axes[idx // 2, idx % 2]
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(n_vars))
        
        for clique in cliques:
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    G.add_edge(clique_list[i], clique_list[j])
        
        # Layout
        if "star" in title.lower():
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color='#3498db', 
                              node_size=1200, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.6, 
                              edge_color='#2c3e50', ax=ax)
        nx.draw_networkx_labels(G, pos, 
                               labels={i: f'v{i}' for i in range(n_vars)},
                               font_size=12, font_weight='bold', 
                               font_color='white', ax=ax)
        
        # Info box
        n_edges = G.number_of_edges()
        info = f"Vars: {n_vars} | Cliques: {len(cliques)} | Edges: {n_edges}"
        ax.text(0.5, -0.15, info, transform=ax.transAxes,
               ha='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_distribution_comparison(exact_probs: np.ndarray, 
                                 quantum_probs: np.ndarray, 
                                 labels: List[str], 
                                 fidelity: float, 
                                 figsize: Tuple[float, float] = (14, 9),
                                 save_path: Optional[str] = None):
    """
    Create comprehensive distribution comparison plots.
    
    Args:
        exact_probs: Exact probability distribution
        quantum_probs: Quantum-sampled distribution
        labels: State labels (e.g., ['00', '01', '10', '11'])
        fidelity: Fidelity score
        figsize: Figure size as (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> from qcgm import DiscreteGraphicalModel, QCGMSampler
        >>> from qcgm.utils import compute_fidelity, estimate_distribution, generate_state_labels
        >>> from qcgm.visualization import plot_distribution_comparison
        >>> 
        >>> model = DiscreteGraphicalModel(2, [{0, 1}])
        >>> model.set_random_parameters()
        >>> sampler = QCGMSampler(model)
        >>> samples, _ = sampler.sample(1000)
        >>> 
        >>> exact = model.compute_probabilities()
        >>> quantum = estimate_distribution(samples, 2)
        >>> labels = generate_state_labels(2)
        >>> fidelity = compute_fidelity(exact, quantum)
        >>> 
        >>> fig = plot_distribution_comparison(exact, quantum, labels, fidelity)
        >>> plt.show()
    """
    _check_dependencies()
    
    # Ensure figsize is a tuple
    if not isinstance(figsize, tuple):
        figsize = (14, 9)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('QCGM: Quantum vs Classical Sampling', 
                 fontsize=16, fontweight='bold')
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot 1: Exact
    axes[0, 0].bar(x, exact_probs, color='#3498db', alpha=0.85, edgecolor='black')
    axes[0, 0].set_title('Exact Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Quantum
    axes[0, 1].bar(x, quantum_probs, color='#e74c3c', alpha=0.85, edgecolor='black')
    axes[0, 1].set_title(f'Quantum (F={fidelity:.4f})', fontweight='bold')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Comparison
    axes[1, 0].bar(x - width/2, exact_probs, width, label='Exact', 
                   color='#3498db', alpha=0.85, edgecolor='black')
    axes[1, 0].bar(x + width/2, quantum_probs, width, label='Quantum',
                   color='#e74c3c', alpha=0.85, edgecolor='black')
    axes[1, 0].set_title('Side-by-Side Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Error
    errors = np.abs(exact_probs - quantum_probs)
    axes[1, 1].bar(x, errors, color='#f39c12', alpha=0.85, edgecolor='black')
    axes[1, 1].set_title('Absolute Error', fontweight='bold')
    axes[1, 1].set_ylabel('|P_exact - P_quantum|')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axhline(np.mean(errors), color='red', linestyle='--', 
                       alpha=0.7, label=f'Mean: {np.mean(errors):.4f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def analyze_circuit_complexity(max_vars: int = 6, 
                               figsize: Tuple[float, float] = (14, 5),
                               save_path: Optional[str] = None):
    """
    Analyze and plot circuit complexity for different model sizes.
    
    Args:
        max_vars: Maximum number of variables to test
        figsize: Figure size as (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> from qcgm.visualization import analyze_circuit_complexity
        >>> fig = analyze_circuit_complexity(max_vars=5)
        >>> plt.show()
    """
    _check_dependencies()
    
    from qcgm import DiscreteGraphicalModel
    from qcgm.circuit import QuantumCircuitBuilder
    
    # Ensure figsize is a tuple
    if not isinstance(figsize, tuple):
        figsize = (14, 5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Test different sizes
    sizes = range(2, max_vars + 1)
    depths = []
    gate_counts = []
    
    for n in sizes:
        # Create simple chain
        cliques = [{i, i+1} for i in range(n-1)]
        test_model = DiscreteGraphicalModel(n, cliques)
        test_model.set_random_parameters()
        
        test_circuit = QuantumCircuitBuilder.build_circuit(test_model)
        depths.append(test_circuit.depth())
        gate_counts.append(test_circuit.size())
    
    # Plot depth
    ax1.plot(sizes, depths, 'o-', linewidth=2, markersize=10, 
            color='#3498db', label='Circuit Depth')
    ax1.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax1.set_title('Circuit Depth vs Model Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(sizes))
    
    # Plot gate count
    ax2.plot(sizes, gate_counts, 's-', linewidth=2, markersize=10,
            color='#e74c3c', label='Gate Count')
    ax2.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Gates', fontsize=12, fontweight='bold')
    ax2.set_title('Gate Count vs Model Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(sizes))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def show_qiskit_circuit(model, 
                       max_gates: int = 50, 
                       save_path: Optional[str] = None):
    """
    Display the actual Qiskit circuit (if not too large).
    
    Args:
        model: DiscreteGraphicalModel instance
        max_gates: Maximum number of gates to display
        save_path: Path to save figure (optional)
    
    Returns:
        QuantumCircuit: The circuit object
    
    Example:
        >>> from qcgm import DiscreteGraphicalModel
        >>> from qcgm.visualization import show_qiskit_circuit
        >>> model = DiscreteGraphicalModel(2, [{0, 1}])
        >>> model.set_random_parameters()
        >>> circuit = show_qiskit_circuit(model)
    """
    from qcgm.circuit import QuantumCircuitBuilder
    
    print("Generating actual Qiskit circuit...")
    
    # Build the actual circuit
    circuit = QuantumCircuitBuilder.build_circuit(model)
    
    print(f"\n✓ Circuit built")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Classical bits: {circuit.num_clbits}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Gates: {circuit.size()}")
    
    # Try to draw circuit
    if _HAS_MATPLOTLIB:
        try:
            from qiskit.visualization import circuit_drawer
            
            # For large circuits, warn user
            if circuit.size() <= max_gates:
                fig = circuit_drawer(circuit, output='mpl', style='iqx', fold=80)
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"\n✓ Saved: {save_path}" if save_path else "\n✓ Circuit displayed")
            else:
                print(f"\n⚠ Circuit too large ({circuit.size()} gates) to display")
                print(f"  Use circuit.draw() in Jupyter for interactive view")
                
                # Show text representation of first few gates
                print("\n  First few operations:")
                for i, instruction in enumerate(circuit.data[:10]):
                    gate = instruction.operation.name
                    qubits = [q._index for q in instruction.qubits]
                    print(f"    {i+1}. {gate} on qubit(s) {qubits}")
                if circuit.size() > 10:
                    print(f"    ... and {circuit.size() - 10} more operations")
        except Exception as e:
            print(f"\n⚠ Could not draw circuit: {e}")
            print("  Circuit object is available as return value")
    
    return circuit


# Module availability check
if not (_HAS_MATPLOTLIB and _HAS_NETWORKX):
    import warnings
    warnings.warn(
        "Visualization module partially unavailable. "
        "Install matplotlib and networkx for full functionality: "
        "pip install matplotlib networkx",
        ImportWarning
    )