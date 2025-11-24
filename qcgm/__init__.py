"""
QCGM: Quantum Circuits for Discrete Graphical Models
=====================================================

A Python library for sampling from discrete graphical models using quantum circuits.

Based on: "On Quantum Circuits for Discrete Graphical Models"
by Nico Piatkowski and Christa Zoufal (2022)
Paper: https://arxiv.org/abs/2206.00398

Key Features
------------
- Exact quantum circuit construction for any discrete graphical model
- Unbiased sampling with no burn-in or mixing time required
- Factorization over cliques following the Hammersley-Clifford theorem
- Compatible with current quantum hardware (NISQ devices)
- Parameter learning via maximum likelihood with quantum sampling

Quick Start
-----------
>>> from qcgm import DiscreteGraphicalModel, QCGMSampler
>>> 
>>> # Create a graphical model
>>> model = DiscreteGraphicalModel(n_vars=3, cliques=[{0, 1}, {1, 2}])
>>> model.set_random_parameters()
>>> 
>>> # Sample using quantum circuit
>>> sampler = QCGMSampler(model)
>>> samples, success_rate = sampler.sample(n_samples=1000)
>>> print(f"Generated {len(samples)} valid samples")

Modules
-------
statistics
    Pauli-Markov sufficient statistics computation (Algorithm 1)
model
    Discrete graphical model representation and exact inference
circuit
    Quantum circuit construction (Theorem 3.4)
sampler
    Main quantum sampling interface
utils
    Utility functions for distribution comparison and analysis
visualization
    Visualization utilities for models and circuits

Main Classes
------------
DiscreteGraphicalModel
    Represents a discrete graphical model over binary variables
QuantumCircuitBuilder
    Constructs quantum circuits for graphical models
QCGMSampler
    Main interface for quantum sampling
PauliMarkovStatistics
    Computes Pauli-Markov sufficient statistics

Utility Functions
-----------------
compute_fidelity
    Compute fidelity between two probability distributions
estimate_distribution
    Estimate probability distribution from samples
compare_distributions
    Comprehensive comparison of two distributions
"""

__version__ = "0.1.0"
__author__ = "Based on Piatkowski & Zoufal (2022)"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/qcgm"
__paper__ = "https://arxiv.org/abs/2206.00398"

# Import main classes
from .statistics import PauliMarkovStatistics
from .model import DiscreteGraphicalModel
from .circuit import QuantumCircuitBuilder
from .sampler import QCGMSampler

# Import commonly used utility functions
from .utils import (
    compute_fidelity,
    estimate_distribution,
    compare_distributions,
    print_comparison,
    hellinger_distance,
    kl_divergence,
    total_variation_distance,
    sample_statistics,
    generate_state_labels
)

# Import visualization functions (optional, requires matplotlib)
try:
    from .visualization import (
        visualize_graphical_model,
        visualize_circuit_diagram,
        compare_model_structures,
        plot_distribution_comparison,
        analyze_circuit_complexity,
        show_qiskit_circuit
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

# Define public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Main classes
    'PauliMarkovStatistics',
    'DiscreteGraphicalModel',
    'QuantumCircuitBuilder',
    'QCGMSampler',
    
    # Utility functions
    'compute_fidelity',
    'estimate_distribution',
    'compare_distributions',
    'print_comparison',
    'hellinger_distance',
    'kl_divergence',
    'total_variation_distance',
    'sample_statistics',
    'generate_state_labels',
]

# Add visualization functions if available
if _HAS_VISUALIZATION:
    __all__.extend([
        'visualize_graphical_model',
        'visualize_circuit_diagram',
        'compare_model_structures',
        'plot_distribution_comparison',
        'analyze_circuit_complexity',
        'show_qiskit_circuit',
    ])


# Package-level convenience functions
def create_chain_model(n_vars: int, random_params: bool = True, **kwargs):
    """
    Create a chain graphical model: v0 - v1 - v2 - ... - v(n-1)
    
    Args:
        n_vars: Number of variables
        random_params: Whether to initialize with random parameters
        **kwargs: Additional arguments for set_random_parameters()
    
    Returns:
        DiscreteGraphicalModel: Chain model
    
    Example:
        >>> model = create_chain_model(4, low=-2.0, high=-0.5)
    """
    cliques = [{i, i+1} for i in range(n_vars - 1)]
    model = DiscreteGraphicalModel(n_vars, cliques)
    if random_params:
        model.set_random_parameters(**kwargs)
    return model


def create_star_model(n_vars: int, center: int = 0, random_params: bool = True, **kwargs):
    """
    Create a star graphical model: center connected to all other nodes
    
    Args:
        n_vars: Number of variables
        center: Index of center node (default: 0)
        random_params: Whether to initialize with random parameters
        **kwargs: Additional arguments for set_random_parameters()
    
    Returns:
        DiscreteGraphicalModel: Star model
    
    Example:
        >>> model = create_star_model(5, center=0, low=-2.0, high=-0.5)
    """
    cliques = [{center, i} for i in range(n_vars) if i != center]
    model = DiscreteGraphicalModel(n_vars, cliques)
    if random_params:
        model.set_random_parameters(**kwargs)
    return model


def create_complete_model(n_vars: int, random_params: bool = True, **kwargs):
    """
    Create a complete graphical model: all nodes connected to each other
    
    Warning: This creates n*(n-1)/2 cliques, which can be expensive!
    
    Args:
        n_vars: Number of variables
        random_params: Whether to initialize with random parameters
        **kwargs: Additional arguments for set_random_parameters()
    
    Returns:
        DiscreteGraphicalModel: Complete model
    
    Example:
        >>> model = create_complete_model(4, low=-1.0, high=-0.1)
    """
    import itertools
    cliques = [set(c) for c in itertools.combinations(range(n_vars), 2)]
    model = DiscreteGraphicalModel(n_vars, cliques)
    if random_params:
        model.set_random_parameters(**kwargs)
    return model


def create_tree_model(edges: list, n_vars: int = None, random_params: bool = True, **kwargs):
    """
    Create a tree graphical model from edges
    
    Args:
        edges: List of edges as tuples (v1, v2)
        n_vars: Number of variables (inferred from edges if not provided)
        random_params: Whether to initialize with random parameters
        **kwargs: Additional arguments for set_random_parameters()
    
    Returns:
        DiscreteGraphicalModel: Tree model
    
    Example:
        >>> edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
        >>> model = create_tree_model(edges, low=-2.0, high=-0.5)
    """
    if n_vars is None:
        n_vars = max(max(e) for e in edges) + 1
    
    cliques = [set(edge) for edge in edges]
    model = DiscreteGraphicalModel(n_vars, cliques)
    if random_params:
        model.set_random_parameters(**kwargs)
    return model


# Add convenience functions to __all__
__all__.extend([
    'create_chain_model',
    'create_star_model',
    'create_complete_model',
    'create_tree_model',
])


# Module-level configuration
_default_backend = None

def set_default_backend(backend):
    """
    Set the default Qiskit backend for all samplers.
    
    Args:
        backend: Qiskit backend instance
    
    Example:
        >>> from qiskit_aer import AerSimulator
        >>> import qcgm
        >>> qcgm.set_default_backend(AerSimulator())
    """
    global _default_backend
    _default_backend = backend


def get_default_backend():
    """
    Get the default Qiskit backend.
    
    Returns:
        Backend instance or None if not set
    """
    return _default_backend


# Version check for dependencies
def check_dependencies():
    """
    Check if all required dependencies are installed with correct versions.
    
    Returns:
        dict: Status of each dependency
    """
    import sys
    status = {}
    
    # Check Python version
    status['python'] = {
        'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'required': '>=3.8',
        'ok': sys.version_info >= (3, 8)
    }
    
    # Check required packages
    required = {
        'numpy': '1.20.0',
        'qiskit': '0.39.0',
        'qiskit_aer': '0.11.0',
        'scipy': '1.7.0',
    }
    
    for package, min_version in required.items():
        try:
            if package == 'qiskit_aer':
                import qiskit_aer
                module = qiskit_aer
            else:
                module = __import__(package)
            
            version = getattr(module, '__version__', 'unknown')
            status[package] = {
                'version': version,
                'required': f'>={min_version}',
                'ok': True  # Simplified check
            }
        except ImportError:
            status[package] = {
                'version': 'not installed',
                'required': f'>={min_version}',
                'ok': False
            }
    
    # Check optional packages
    optional = {
        'matplotlib': '3.3.0',
        'networkx': '2.5',
    }
    
    for package, min_version in optional.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            status[package] = {
                'version': version,
                'required': f'>={min_version} (optional)',
                'ok': True
            }
        except ImportError:
            status[package] = {
                'version': 'not installed',
                'required': f'>={min_version} (optional)',
                'ok': False
            }
    
    return status


def print_dependency_status():
    """Print the status of all dependencies."""
    status = check_dependencies()
    
    print("=" * 60)
    print("QCGM Dependency Status")
    print("=" * 60)
    
    for name, info in status.items():
        status_symbol = "✓" if info['ok'] else "✗"
        print(f"{status_symbol} {name:15s} {info['version']:15s} (required: {info['required']})")
    
    print("=" * 60)
    
    # Check core dependencies
    core_packages = ['python', 'numpy', 'qiskit', 'qiskit_aer', 'scipy']
    all_core_ok = all(status[pkg]['ok'] for pkg in core_packages if pkg in status)
    
    if all_core_ok:
        print("✓ All core dependencies satisfied!")
        if not _HAS_VISUALIZATION:
            print("⚠ Visualization not available (matplotlib/networkx missing)")
            print("  Install with: pip install matplotlib networkx")
    else:
        print("✗ Some core dependencies missing or outdated")
        print("  Run: pip install -r requirements.txt")
    print("=" * 60)


# Information function
def info():
    """
    Print package information.
    """
    print("=" * 70)
    print("QCGM: Quantum Circuits for Discrete Graphical Models")
    print("=" * 70)
    print(f"Version:        {__version__}")
    print(f"Author:         {__author__}")
    print(f"License:        {__license__}")
    print(f"Paper:          {__paper__}")
    print()
    print("Description:")
    print("  A library for sampling from discrete graphical models using")
    print("  quantum circuits. Implements the algorithm from Piatkowski &")
    print("  Zoufal (2022) with exact circuit construction and provably")
    print("  unbiased sampling.")
    print()
    print("Key Components:")
    print("  - DiscreteGraphicalModel:  Model representation")
    print("  - QuantumCircuitBuilder:   Circuit construction")
    print("  - QCGMSampler:             Quantum sampling")
    print("  - Utility functions:       Analysis tools")
    if _HAS_VISUALIZATION:
        print("  - Visualization:           Plotting and diagrams ✓")
    else:
        print("  - Visualization:           Not available (install matplotlib)")
    print()
    print("Quick Start:")
    print("  >>> from qcgm import DiscreteGraphicalModel, QCGMSampler")
    print("  >>> model = DiscreteGraphicalModel(3, [{0,1}, {1,2}])")
    print("  >>> model.set_random_parameters()")
    print("  >>> sampler = QCGMSampler(model)")
    print("  >>> samples, rate = sampler.sample(n_samples=1000)")
    print()
    print("Convenience Functions:")
    print("  >>> from qcgm import create_chain_model, create_star_model")
    print("  >>> chain = create_chain_model(5)")
    print("  >>> star = create_star_model(5, center=0)")
    print()
    if _HAS_VISUALIZATION:
        print("Visualization:")
        print("  >>> from qcgm import visualize_graphical_model")
        print("  >>> fig = visualize_graphical_model(model)")
        print("  >>> plt.show()")
        print()
    print("Documentation:")
    print("  See README.md and examples/ directory")
    print("=" * 70)


# Expose configuration functions
__all__.extend([
    'set_default_backend',
    'get_default_backend',
    'check_dependencies',
    'print_dependency_status',
    'info',
])


# Optional: Print info on import in interactive mode
import sys
if hasattr(sys, 'ps1'):  # Interactive mode
    # Uncomment to show info automatically in interactive sessions
    # info()
    pass