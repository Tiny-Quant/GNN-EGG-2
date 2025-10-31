import pytest
import torch
import networkx as nx
from src.datasets.graph_perturb import generate_perturbed_pairs, nx_to_pyg, _perturb_graph


def test_generate_perturbed_pairs_small_graph():
    # Create a simple base dataset
    G = nx.cycle_graph(4)
    for n in G.nodes:
        G.nodes[n]["feat"] = torch.tensor([float(n)])
    dataset = [nx_to_pyg(G)]

    pairs = generate_perturbed_pairs(dataset, n_perturbations=3)
    assert len(pairs) == 3, f"Expected 3 pairs but got {len(pairs)}"
    for g1, g2, ged in pairs:
        assert isinstance(ged, float)
        assert hasattr(g1, "x") and hasattr(g2, "edge_index")
        # ensure non-empty tensors
        assert g1.x.ndim == 2
        assert g2.edge_index.ndim == 2
        # additional robustness: GED must be non-negative and integer-like
        assert ged >= 0
        assert abs(ged - round(ged)) < 1e-6, f"GED {ged} not close to an integer"


def test_perturbed_graphs_connected():
    G = nx.path_graph(5)
    for n in G.nodes:
        G.nodes[n]["feat"] = torch.tensor([1.0])
    dataset = [nx_to_pyg(G)]
    pairs = generate_perturbed_pairs(dataset, n_perturbations=1)
    _, g2, _ = pairs[0]

    # Convert back to networkx and check connectedness
    G2 = nx.Graph()
    G2.add_edges_from(g2.edge_index.t().tolist())
    assert nx.is_connected(G2) or G2.number_of_nodes() <= 1


def test_disconnected_graph_reconnects():
    """
    Verify that a disconnected graph becomes connected
    after running _perturb_graph(), due to reconnection logic.
    """
    # Create a disconnected graph: two components
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])  # two disconnected components
    for n in G.nodes:
        G.nodes[n]["feat"] = torch.tensor([1.0])

    # Provide a dummy graph pool for attribute sampling
    graphs = [G]

    # Apply perturbation with low removal but ensure reconnection check runs
    G_pert, edits = _perturb_graph(
        G,
        graphs,
        node_add_prob=0.0,
        node_remove_prob=0.0,
        edge_add_prob=0.0,
        edge_remove_prob=0.0,
        max_edges_per_new_node=1,
    )

    # Convert to undirected view and verify connectivity
    G_pert_undirected = G_pert.to_undirected(as_view=True)
    assert nx.is_connected(G_pert_undirected), (
        "Graph should be reconnected by the perturbation logic"
    )

    # Confirm at least one edge was added for reconnection
    assert edits["edge_add"] >= 1, "At least one new edge should be added to reconnect components"


def test_never_returns_empty_graph():
    """Ensure _perturb_graph never returns an empty graph."""
    G = nx.path_graph(4)
    for n in G.nodes:
        G.nodes[n]["feat"] = torch.tensor([1.0])
    Gp, edits = _perturb_graph(G, [G])
    assert Gp.number_of_nodes() > 0
