from __future__ import annotations

import networkx as nx
import numpy as np


def chain_graph(n: int):
    return nx.path_graph(n)


def grid_graph(n: int):
    s = int(n**0.5)
    if s * s != n:
        raise ValueError("n must be a perfect square")
    return nx.grid_2d_graph(s, s)


def erdos_renyi(n: int, p: float, seed: int = 0):
    return nx.erdos_renyi_graph(n, p, seed=seed)


def two_clique(n: int):
    half = n // 2
    g = nx.complete_graph(half)
    h = nx.complete_graph(half)
    h = nx.relabel_nodes(h, {i: i + half for i in h.nodes()})
    out = nx.compose(g, h)
    out.add_edge(half - 1, half)
    return out


def barbell(n: int):
    k = max(2, n // 3)
    return nx.barbell_graph(k, k)


def barbell_path(n: int):
    if n < 6:
        raise ValueError("barbell_path requires n >= 6")
    k = max(2, n // 4)
    while 2 * k >= n:
        k -= 1
    mid = n - (2 * k)

    g = nx.Graph()
    left = list(range(k))
    right = list(range(k + mid, n))
    g.add_nodes_from(range(n))

    for i in left:
        for j in left:
            if i < j:
                g.add_edge(i, j)
    for i in right:
        for j in right:
            if i < j:
                g.add_edge(i, j)

    if mid == 0:
        g.add_edge(left[-1], right[0])
    else:
        path_nodes = list(range(k, k + mid))
        g.add_edge(left[-1], path_nodes[0])
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            g.add_edge(a, b)
        g.add_edge(path_nodes[-1], right[0])
    return g


def sample_mrf_params(graph, low: float = -5.0, high: float = 0.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    params = {}
    for u in graph.nodes():
        params[frozenset([u])] = float(rng.uniform(low, high))
    for u, v in graph.edges():
        params[frozenset([u, v])] = float(rng.uniform(low, high))
    return params


def family_registry():
    return {
        "chain": lambda n, seed: chain_graph(n),
        "erdos_renyi": lambda n, seed: erdos_renyi(n, p=0.5, seed=seed),
        "two_clique": lambda n, seed: two_clique(n),
        "barbell": lambda n, seed: barbell(n),
        "barbell_path": lambda n, seed: barbell_path(n),
    }

