"""
Global pytest fixtures for GNN training experiments.
Ensures reproducibility, isolation, and reusable mini datasets.
"""

import pytest
import shutil
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from src.datasets.ged_dataset import GEDDataset, collate_pairs
from src.models.sim_gnn import SimGNN


# ---------------------------
# Workspace / Paths
# ---------------------------
@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory):
    """
    Shared temporary workspace for test artifacts.
    - Creates /tmp/.../workspaceN once per session
    - Removes old workspaces automatically
    - Persists until the end of test session for debugging
    """
    path = tmp_path_factory.mktemp("workspace")
    workspace_root = path.parent

    for old_dir in workspace_root.iterdir():
        if old_dir.is_dir() and old_dir.name.startswith("workspace") and old_dir != path:
            shutil.rmtree(old_dir, ignore_errors=True)

    print(f"[pytest setup] Temporary workspace: {path}")
    return path


# ---------------------------
# Reproducibility
# ---------------------------
@pytest.fixture(autouse=True, scope="session")
def set_random_seeds():
    """Ensure deterministic results across test runs."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ---------------------------
# Dataset Fixtures
# ---------------------------
@pytest.fixture(scope="session")
def toy_ged_dataset():
    """Create a minimal synthetic GED dataset for GNN tests."""
    g1 = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    )
    g2 = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 2, 1], [1, 3, 2]], dtype=torch.long)
    )
    pairs = [(g1, g2, 0.2), (g2, g1, 0.1)]
    dataset = GEDDataset(pairs)
    return dataset


@pytest.fixture(scope="session")
def toy_dataloader(toy_ged_dataset):
    """Simple DataLoader fixture for small-scale tests."""
    return DataLoader(toy_ged_dataset, batch_size=2, shuffle=False, collate_fn=collate_pairs)


# ---------------------------
# Model Fixtures
# ---------------------------
@pytest.fixture(scope="session")
def toy_simgnn(toy_ged_dataset):
    """Instantiate a small SimGNN model for regression testing."""
    in_dim = toy_ged_dataset[0][0].x.size(1)
    model = SimGNN(in_dim, hidden_dim=8, use_tensor=False)
    return model
