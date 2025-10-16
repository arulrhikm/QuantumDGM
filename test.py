"""
Main testing script for QCGM (Quantum Circuits for Graphical Models)
Implements the construction from:
Piatkowski & Zoufal (2022), "On Quantum Circuits for Discrete Graphical Models"

Process:
1. Build and visualize a simple graphical model.
2. Sample 1000 times from the classical distribution.
3. Construct the corresponding QCGM circuit.
4. Run on BlueQubit backend for 1000 shots.
5. Condition on auxiliary qubits = 0...0 (extract Pθ(x)).
6. Compare and visualize classical vs. quantum distributions.
"""

from qcgmlib.models.graphical_model import GraphicalModel
from qcgmlib.qcg_circuit.qcgm_circuit import build_qcg_circuit
from qcgmlib.sampling.simulator import simulate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def condition_on_auxiliary_zero(counts: dict, n_target: int, n_aux: int) -> dict:
    """
    Condition on auxiliary qubits being all |0...0⟩.
    Extracts the marginal distribution over the target register |x⟩.

    Args:
        counts: dict of {bitstring: count} from the quantum circuit
        n_target: number of target qubits (graphical model variables)
        n_aux: number of auxiliary qubits

    Returns:
        dict[str, int]: filtered counts corresponding to aux=0...0
    """
    conditioned = {}
    for bitstring, c in counts.items():
        # Qiskit bit order: rightmost bit = qubit 0 (little endian)
        bits = bitstring[::-1]  # reverse to make index 0 leftmost
        aux_bits = bits[n_target : n_target + n_aux]
        x_bits = bits[:n_target]
        if all(b == "0" for b in aux_bits):  # condition aux = 0...0
            conditioned[x_bits[::-1]] = conditioned.get(x_bits[::-1], 0) + c
    return conditioned


def main():
    # ---------------------------------------------------------------------
    # 1️⃣ Define and visualize the classical graphical model
    # ---------------------------------------------------------------------
    gm = GraphicalModel(
        n_vars=3,
        cliques=[[0, 1], [1, 2]],  # pairwise dependencies
        theta=[0.4, 0.8]           # weights per clique
    )

    print("\n=== Classical Graphical Model ===")
    gm.describe()
    gm.visualize()  # draw the graph first

    # ---------------------------------------------------------------------
    # 2️⃣ Sample 1000 times from the classical distribution
    # ---------------------------------------------------------------------
    n_samples = 1000
    print(f"\nSampling {n_samples} i.i.d. samples from classical model...")
    classical_counts = gm.sample(n_samples)

    plt.figure(figsize=(7, 4))
    plot_histogram(classical_counts)
    plt.title(f"Classical Graphical Model Sampling ({n_samples} samples)")
    plt.xlabel("Bitstring Outcomes")
    plt.ylabel("Frequency")
    plt.show()

    # ---------------------------------------------------------------------
    # 3️⃣ Build and visualize the quantum circuit (QCGM)
    # ---------------------------------------------------------------------
    print("\n=== Constructing Quantum Circuit for Graphical Model ===")
    qc = build_qcg_circuit(gm.n, gm.cliques, gm.theta)
    print(qc.draw(output="text"))

    # Number of auxiliary qubits = 1 + |C| (see paper)
    n_aux = 1 + len(gm.cliques)

    # ---------------------------------------------------------------------
    # 4️⃣ Simulate using BlueQubit backend (1000 shots)
    # ---------------------------------------------------------------------
    print("\nRunning QCGM circuit on BlueQubit simulator...")
    quantum_counts = simulate(
        qc,
        shots=n_samples,
        device="cpu",
        api_token="lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF"  # 🔑 Replace with your BlueQubit key
    )

    print("\n=== Raw Quantum Joint Sampling Results ===")
    for bitstring, count in sorted(quantum_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{bitstring}: {count}")

    # ---------------------------------------------------------------------
    # 5️⃣ Condition on auxiliary qubits = 0...0 to extract target register
    # ---------------------------------------------------------------------
    print(f"\nConditioning on auxiliary qubits = |0...0⟩ (total {n_aux} aux qubits)...")
    quantum_conditioned = condition_on_auxiliary_zero(quantum_counts, gm.n, n_aux)

    # Normalize counts for display
    total_valid = sum(quantum_conditioned.values())
    total_all = sum(quantum_counts.values())
    retained_fraction = total_valid / total_all if total_all > 0 else 0
    print(f"Retained {retained_fraction:.2%} of shots after conditioning.\n")

    # ---------------------------------------------------------------------
    # 6️⃣ Visualize both histograms side by side
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_histogram(classical_counts)
    plt.title(f"Classical Sampling ({n_samples} samples)")

    plt.subplot(1, 2, 2)
    plot_histogram(quantum_conditioned)
    plt.title(f"Quantum (Conditioned) Sampling — BlueQubit ({n_samples} shots)")
    plt.show()

    # ---------------------------------------------------------------------
    # 7️⃣ Compare distributions numerically
    # ---------------------------------------------------------------------
    gm.compare_distributions(quantum_conditioned, classical_counts)


if __name__ == "__main__":
    main()
