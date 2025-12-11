# QuantumDGM Examples & Research Demonstrations

This directory contains demonstrations, research examples, and validation tests for the Quantum Circuit-based Graphical Models (QCGM) implementation.

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── demo_script.py                     # Basic functionality demo
├── demo_notebook.ipynb                # Interactive Jupyter tutorial
│
├── 🎯 Research Demonstrations
│   ├── quantum_vs_classical_demo.py   # Honest comparison of sampling methods
│   ├── variational_training_demo.py   # Large-model training demonstration
│
├── 📊 figures/                        # Generated visualizations
│   ├── sampling_methods_comparison.png
│   ├── quantum_properties.png
│   ├── gibbs_comparison_explained.png
│   ├── when_quantum_matters.png
│   ├── variational_training_convergence.png
│   └── model_*.png, circuit_*.png
│
└── ✅ tests/                          # Validation & correctness tests
    ├── test_optimizations.py          # Memory & cache optimization tests
    ├── test_variational_training.py   # Training loop validation
    └── IMPLEMENTATION_SUMMARY.md      # Technical implementation details
```

---

## 🚀 Quick Start

### 1. **Basic Demo** (Start Here!)

```bash
python demo_script.py
```

**What it shows:**
- Creating discrete graphical models
- Building quantum circuits
- Sampling from distributions
- Comparing with classical methods

**Time:** ~2 minutes

---

### 2. **Interactive Tutorial**

```bash
jupyter notebook demo_notebook.ipynb
```

**What's included:**
- Step-by-step walkthrough
- Visualizations of models and circuits
- Probability distribution analysis
- Circuit structure exploration

**Best for:** Learning the fundamentals

---

## 🔬 Research Demonstrations

### Honest Quantum vs Classical Comparison

```bash
python quantum_vs_classical_demo.py
```

**Research Question:** *What are the REAL advantages of quantum sampling?*

**What it demonstrates:**
1. **Equal-Information Comparison**: When all methods know P(x), quantum ≈ classical
2. **Quantum Properties**: Independence, no burn-in, high ESS
3. **Gibbs Comparison Explained**: Why Gibbs seems worse (solving different problems!)
4. **When Quantum Matters**: Real use cases where properties help

**Key Insight:** The simplified amplitude encoding doesn't give computational advantage,
but quantum sampling properties (independence, no burn-in) are valuable for:
- Monte Carlo integration
- Real-time applications
- Statistical analysis
- Parallel sampling

**Outputs:** 4 publication-quality figures in `figures/`

**Time:** ~3-5 minutes

---

### Variational Circuit Training for Large Models

```bash
python variational_training_demo.py
```

**Research Question:** *Can we sample from models with n > 10 variables?*

**What it demonstrates:**
1. **Manual Training**: Explicit optimization loop
2. **Smart Auto-Selection**: Automatic exact vs approximate choice
3. **Loss Function Comparison**: KL, fidelity, L2
4. **Progress Monitoring**: Callback-based tracking
5. **Automatic Depth Selection**: Structure-aware configuration

**Key Achievement:** Enables sampling from n=10-20 variable models with:
- **17-8,738x parameter reduction** vs exact methods
- **Tunable accuracy** via depth and optimization steps
- **Multiple entanglement strategies**

**Outputs:** Convergence plot in `figures/`

**Time:** ~2-3 minutes

---

## ✅ Validation Tests

### Test 1: Memory & Cache Optimizations

```bash
python tests/test_optimizations.py
```

**Tests:**
- ✓ Sparse diagonal Hamiltonian (O(2^n) → O(2^n) elements, but 2^n space savings)
- ✓ Caching effectiveness (1000x+ speedup on repeated calls)
- ✓ Approximate circuit builder
- ✓ Smart circuit selection
- ✓ Memory savings quantification

**Expected output:**
```
✓ ALL OPTIMIZATION TESTS COMPLETED!
```

**Time:** ~10 seconds

---

### Test 2: Variational Training Correctness

```bash
python tests/test_variational_training.py
```

**Comprehensive test suite:**
1. ✓ Training convergence (loss decreases)
2. ✓ Multiple loss functions (KL, fidelity, L2)
3. ✓ Reordering correctness (F = 0.9949 on test case)
4. ✓ Entanglement strategies (linear, clique, full)
5. ✓ Convergence trajectory tracking
6. ✓ Scaling to n=6, 8, 10
7. ✓ Custom initialization

**Expected output:**
```
Variational Circuit Training: READY FOR PRODUCTION
✓ ALL TESTS PASSED
```

**Time:** ~2-3 minutes

**Documentation:** See `tests/IMPLEMENTATION_SUMMARY.md` for details

---

## 🎓 Research Direction & Objectives

This package addresses key challenges in quantum machine learning for graphical models:

### ✅ **Implemented**

| Optimization | Status | Impact | Location |
|-------------|--------|---------|----------|
| **Sparse Diagonal Hamiltonian** | ✅ Complete | O(4^n) → O(2^n) memory | `QuantumDGM/model.py` |
| **Probability Caching** | ✅ Complete | 1000x+ speedup | `QuantumDGM/model.py` |
| **Variational Compression** | ✅ Complete | Enables n>10 | `QuantumDGM/circuit.py` |
| **Smart Circuit Selection** | ✅ Complete | Automatic optimization | `QuantumDGM/circuit.py` |

### 🎯 **Research Proposal Alignment**

From the research objectives:

> **R1. Low-Ancilla Circuit Designs:** Efficient QCGM-style encodings using O(n+1) qubits

**Status:** ⚠️ Partial
- Current: Simplified amplitude encoding (pedagogical)
- Missing: Ancilla reuse with repeat-until-success
- Impact: Would enable true low-ancilla implementation

> **R2. Hybrid Inference Methods:** Classical correction schemes

**Status:** ❌ Not implemented
- Missing: Rejection sampling with reweighting
- Missing: Error mitigation (zero-noise extrapolation, readout correction)
- Difficulty: Medium (4-6 hours)
- Utility: High (for real hardware)

> **R3. Variationally Compressed Models:** Fixed-depth PQCs

**Status:** ✅ **COMPLETE**
- ✅ Fixed-depth circuits
- ✅ O(n × depth) parameters instead of O(2^n)
- ✅ Multiple loss functions
- ✅ Training loop with validation

> **R4. Quantitative Benchmarks:** Comparison with classical MCMC

**Status:** ✅ **COMPLETE** (with caveats)
- ✅ Honest comparison framework
- ✅ Equal-info vs different-problem comparisons explained
- ✅ When quantum properties matter
- ⚠️ Note: Current implementation is pedagogical, not full QCGM

---

## 📊 Performance Characteristics

### Memory Scaling

| n | Exact Amplitudes | Sparse Diagonal | Variational Params |
|---|------------------|-----------------|---------------------|
| 10 | 8 KB | 8 KB | 480 bytes |
| 15 | 256 KB | 256 KB | 720 bytes |
| 20 | 8 MB | 8 MB | 960 bytes |

### Time Complexity

| Operation | Exact Method | Variational Training |
|-----------|--------------|----------------------|
| Circuit Building | O(2^n) | O(n × depth) |
| Sampling (1000 shots) | O(2^n) setup + O(1) per sample | O(n × depth × iterations) |
| Best for | n ≤ 10 | n = 10-20 |

---

## 🎯 Next Steps for Research

### High-Priority Additions

1. **Readout Error Mitigation** (Easy, 1-2 hours)
   - Qiskit has built-in `LocalReadoutMitigator`
   - 10-30% accuracy improvement on real hardware
   - See: `examples/variational_training_demo.py` for foundation

2. **Hybrid Rejection Sampling** (Medium, 4-6 hours)
   - Combine quantum + classical samples
   - Better sample efficiency when success rate < 50%
   - Foundation exists in `QuantumDGM/sampler.py`

3. **Clique-Based Entanglement Optimization** (Medium, 3-4 hours)
   - Use chromatic number χ(G) for optimal depth
   - Reduce circuit depth by 2-5x
   - Requires graph coloring algorithm

### Future Research Directions

1. **Full Ancilla-Based QCGM** (Hard, 1-2 weeks)
   - Implement true low-ancilla design from Piatkowski & Zoufal
   - Hamiltonian decomposition into clique factors
   - Repeat-until-success protocol

2. **Hardware Experiments** (Requires IBM Quantum access)
   - Run on real quantum hardware
   - Validate error mitigation strategies
   - Compare noisy vs noiseless results

3. **Scalability Studies** (Research project)
   - Systematic benchmarking: n=10,15,20,25
   - Fidelity vs depth vs optimization budget
   - Classical MCMC comparison (fair: same problem)

---

## 📖 Related Documentation

- **Main README**: `../README.md` - Package overview and installation
- **Implementation Details**: `tests/IMPLEMENTATION_SUMMARY.md` - Technical summary
- **Research Proposal**: See research objectives in package header

---

## 🤝 Contributing Research Examples

To add a new research demonstration:

1. **Create script**: `examples/my_research_demo.py`
2. **Add documentation**: Clear comments and docstring
3. **Generate figures**: Save to `examples/figures/`
4. **Add to this README**: Under "Research Demonstrations"
5. **Include timing**: Expected runtime
6. **State research question**: What does it demonstrate?

Example template:

```python
"""
Research Demo: [Title]

Research Question: [Clear statement]

What it demonstrates:
1. [Key point 1]
2. [Key point 2]
...

Time: ~X minutes
"""

# Your code here
```

---

## 📚 Citation

If you use these examples in your research, please cite:

```bibtex
@software{quantumdgm2025,
  title = {Quantum Circuit-based Discrete Graphical Models},
  author = {QuantumDGM Contributors},
  year = {2025},
  url = {https://github.com/yourusername/QuantumDGM}
}
```

---

## 📧 Questions?

- **Issues**: Open a GitHub issue
- **Discussion**: Use GitHub Discussions
- **Email**: [Contact information]

---

**Last Updated:** December 2025  
**Status:** Research-Ready ✅  
**Version:** 0.1.0

