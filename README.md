# Quantum Circuits for Discrete Graphical Models

Implementation of the methods proposed in  
**“On Quantum Circuits for Discrete Graphical Models”**  
by *Nico Piatkowski* and *Christa Zoufal*.

This repository provides a quantum algorithm for **unbiased and independent sampling**, **learning**, and **inference** from **discrete graphical models** using **quantum circuits**. The approach is compatible with **multi-body interactions** and can be executed on **current quantum hardware**.

---

## 🚀 Overview

Graphical models are powerful tools for describing structured, high-dimensional probability distributions.  
Sampling from such models—especially those with discrete variables—poses significant computational challenges.

This project implements a **quantum circuit-based method** that:

- Generates **unbiased, independent samples** from general discrete factor models  
- Embeds graphical models into **unitary operators** with provable guarantees  
- Supports **maximum likelihood estimation (MLE)** and **maximum a posteriori (MAP)** inference  
- Is compatible with **modern hybrid quantum-classical optimization techniques**  

The algorithm’s **success probability** is independent of the number of variables, and it provides a **unitary Hammersley–Clifford theorem**, establishing factorization over cliques of the model’s conditional independence structure.

---

## 🧠 Key Features

- ✅ Provably unbiased sampling from discrete graphical models  
- ⚙️ Unitary embedding of model factors  
- 🔁 Hybrid quantum-classical training for parameter learning  
- 🧩 Support for multi-body interactions  
- 🧮 Runnable on current quantum processors and simulators  
- 📈 Includes experiments on quantum simulation and real hardware  

---

## 🧰 Installation

```bash
# Clone this repository
git clone https://github.com/<your-username>/quantum-graphical-models.git
cd quantum-graphical-models

# Install dependencies
pip install -r requirements.txt
