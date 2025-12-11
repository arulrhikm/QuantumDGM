"""
Setup script for QuantumDGM package.

Install with:
    pip install git+https://github.com/arulrhikm/QuantumDGM.git
    pip install -e .  # Development mode
    pip install .[viz]  # With visualization
    pip install .[dev]  # With development tools
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_file = os.path.join('QuantumDGM', '__init__.py')
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read long description from README
def get_long_description():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Core dependencies
INSTALL_REQUIRES = [
    'numpy>=1.20.0',
    'qiskit>=1.0.0',
    'qiskit-aer>=0.13.0',
    'scipy>=1.7.0',
]

# Optional dependencies
EXTRAS_REQUIRE = {
    # Visualization support
    'viz': [
        'matplotlib>=3.5.0,<4.0.0',
        'networkx>=2.5,<4.0.0',
        'seaborn>=0.11.0,<1.0.0',
        'qiskit-ibm-runtime>=0.15.0',
    ],
    
    # Jupyter notebook support
    'notebooks': [
        'jupyter>=1.0.0',
        'ipywidgets>=7.7.0',
        'notebook>=6.4.0',
        'jupyterlab>=3.4.0',
    ],
    
    # Development tools
    'dev': [
        'pytest>=7.0.0,<9.0.0',
        'pytest-cov>=3.0.0,<5.0.0',
        'pytest-mock>=3.10.0',
        'black>=22.0.0,<25.0.0',
        'flake8>=4.0.0,<8.0.0',
        'isort>=5.10.0',
        'mypy>=0.950,<2.0.0',
        'sphinx>=4.5.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
}

# Add 'all' option that includes everything
EXTRAS_REQUIRE['all'] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name='QuantumDGM',
    version=get_version(),
    author='Arul Rhik Mazumder, Bryan Zhang',
    author_email='arulm@andrew.cmu.edu',
    description='Quantum Circuits for Discrete Graphical Models',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/arulrhikm/QuantumDGM',
    project_urls={
        'Documentation': 'https://github.com/arulrhikm/QuantumDGM',
        'Source': 'https://github.com/arulrhikm/QuantumDGM',
        'Paper': 'https://arxiv.org/abs/2206.00398',
    },
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='quantum computing, graphical models, quantum circuits, sampling, qiskit',
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)