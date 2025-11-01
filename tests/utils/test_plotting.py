import matplotlib

matplotlib.use("Agg")

import torch
from torch_geometric.data import Data

from src.utils.plotting import plot_pyg_graphs


def test_plot_pyg_graphs_with_discrete_blocks():
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index)

    fig, axes = plot_pyg_graphs(
        [graph],
        titles=["toy"],
        discrete_blocks=[3, 2],
        show=False,
    )

    assert len(axes) == 1
    assert fig is not None

    # ensure figure resources are released in test environment
    import matplotlib.pyplot as plt

    plt.close(fig)
