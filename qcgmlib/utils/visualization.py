from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def show_counts(counts, title="Measurement Results"):
    """Plot histogram of measured counts."""
    fig = plot_histogram(counts)
    plt.title(title)
    plt.show()
