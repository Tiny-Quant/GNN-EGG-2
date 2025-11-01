"""Utility functions for visualising PyG graphs across datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


@dataclass
class NodeColoring:
    """Container describing node colour assignments."""

    labels: Optional[torch.Tensor]
    legend: Dict[int, str]


def _infer_discrete_labels(
    data: Data,
    *,
    feature_key: str = "x",
    discrete_blocks: Optional[Sequence[int]] = None,
) -> NodeColoring:
    """Infer categorical node labels from discrete feature blocks.

    Args:
        data: PyG graph.
        feature_key: Attribute name holding node features. Defaults to ``x``.
        discrete_blocks: Optional sequence describing the width of one-hot blocks
            within the feature tensor. When provided, each node is assigned a
            tuple of block-wise argmax indices. The tuples are mapped to integer
            colours so nodes sharing the same discrete pattern receive the same
            colour.

    Returns:
        ``NodeColoring`` describing encoded labels and legend mapping. If no
        discrete structure can be inferred the labels will be ``None``.
    """

    feats = getattr(data, feature_key, None)
    if feats is None:
        return NodeColoring(labels=None, legend={})

    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)

    if feats.ndim != 2 or feats.size(0) == 0:
        return NodeColoring(labels=None, legend={})

    feats = feats.detach().cpu()

    if discrete_blocks:
        start = 0
        block_assignments: List[torch.Tensor] = []
        for width in discrete_blocks:
            if width <= 0 or start + width > feats.size(1):
                break
            block = feats[:, start : start + width]
            block_assignments.append(block.argmax(dim=-1))
            start += width
        if not block_assignments:
            return NodeColoring(labels=None, legend={})

        tuples = [tuple(int(b[i].item()) for b in block_assignments) for i in range(feats.size(0))]
        unique = {code: idx for idx, code in enumerate(sorted(set(tuples)))}
        labels = torch.tensor([unique[t] for t in tuples], dtype=torch.long)
        legend = {idx: "-".join(map(str, code)) for code, idx in unique.items()}
        return NodeColoring(labels=labels, legend=legend)

    # Attempt to detect a single one-hot block across the whole feature tensor
    if feats.size(1) > 1:
        # treat rows close to one-hot as discrete
        approx_binary = torch.isclose(feats, feats.round()).all(dim=1)
        if approx_binary.all():
            labels = feats.argmax(dim=-1)
            legend = {int(c.item()): f"class_{int(c.item())}" for c in labels.unique()}
            return NodeColoring(labels=labels, legend=legend)

    return NodeColoring(labels=None, legend={})


def plot_pyg_graphs(
    graphs: Sequence[Data],
    *,
    titles: Optional[Sequence[str]] = None,
    layout: str = "spring",
    feature_key: str = "x",
    discrete_blocks: Optional[Sequence[int]] = None,
    palette: Optional[Sequence[str]] = None,
    node_size: int = 400,
    layout_seed: int = 42,
    show: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot one or multiple PyG graphs with discrete colour coding.

    Args:
        graphs: Sequence of PyG ``Data`` objects to visualise.
        titles: Optional per-graph titles.
        layout: Layout algorithm name understood by NetworkX (``spring`` by default).
        feature_key: Which node feature attribute to inspect for colouring.
        discrete_blocks: Optional block structure for discrete node features.
        palette: Matplotlib colour palette. Defaults to ``tab20``.
        node_size: Node size passed to ``networkx.draw``.
        layout_seed: Seed controlling layout randomness for reproducibility.
        show: When ``True`` the plot is rendered via ``plt.show``.

    Returns:
        ``(figure, axes)`` tuple for further customisation.
    """

    if len(graphs) == 0:
        raise ValueError("At least one graph is required for plotting.")

    if titles is None:
        titles = ["Graph" for _ in graphs]
    elif len(titles) != len(graphs):
        raise ValueError("Length of titles must match number of graphs")

    cmap = palette or get_cmap("tab20").colors
    if hasattr(cmap, "__call__"):
        colour_list = [cmap(i / max(1, len(graphs))) for i in range(max(1, len(graphs)))]
    else:
        colour_list = list(cmap)

    fig, axes = plt.subplots(1, len(graphs), figsize=(6 * len(graphs), 6))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    else:
        axes = list(axes)

    legend_handles: Dict[str, Line2D] = {}

    for idx, (data, ax, title) in enumerate(zip(graphs, axes, titles)):
        nx_graph = to_networkx(data, to_undirected=True)
        if layout == "spring":
            pos = nx.spring_layout(nx_graph, seed=layout_seed)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(nx_graph)
        else:
            raise ValueError(f"Unsupported layout '{layout}'.")

        colouring = _infer_discrete_labels(
            data,
            feature_key=feature_key,
            discrete_blocks=discrete_blocks,
        )

        if colouring.labels is not None:
            labels = colouring.labels.tolist()
            colours = [colour_list[label % len(colour_list)] for label in labels]
        else:
            labels = None
            colours = colour_list[idx % len(colour_list)]

        nx.draw(
            nx_graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color=colours,
            node_size=node_size,
            edge_color="#555555",
        )
        ax.set_title(title)

        if colouring.labels is not None:
            for label_idx, name in colouring.legend.items():
                if name not in legend_handles:
                    legend_handles[name] = Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colour_list[label_idx % len(colour_list)],
                        markersize=10,
                    )

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
        )

    plt.tight_layout()
    if show:
        plt.show()

    return fig, list(axes)


__all__ = ["plot_pyg_graphs", "NodeColoring"]
