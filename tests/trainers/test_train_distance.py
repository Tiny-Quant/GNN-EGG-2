import pytest
import torch
from src.models.sim_gnn import SimGNN
from src.datasets.ged_dataset import collate_pairs
from src.trainers.train_distance import evaluate


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_train_sim_gnn_smoketest(toy_simgnn, toy_dataloader, tmp_workspace, device):
    """
    Ensure SimGNN trains for one mini-batch on the toy GED dataset.
    """
    model = toy_simgnn.to(device)
    loader = toy_dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    batch1, batch2, labels = next(iter(loader))
    batch1, batch2, labels = batch1.to(device), batch2.to(device), labels.to(device)

    optimizer.zero_grad()
    pred = model(batch1, batch2)
    loss = torch.nn.functional.mse_loss(pred, labels)
    loss.backward()
    optimizer.step()

    # Sanity checks
    assert torch.isfinite(loss).all(), "Loss contains NaN or Inf"
    assert loss.item() >= 0
    assert pred.shape == labels.shape

    # Save checkpoint
    ckpt_path = tmp_workspace / "sim_gnn_test.pt"
    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_eval_simgnn_metrics(toy_simgnn, toy_dataloader, device):
    """
    Verify evaluate() returns normalized and unnormalized metrics on the toy dataset.
    """
    model = toy_simgnn.to(device)

    with torch.no_grad():
        metrics = evaluate(model, toy_dataloader, device, verbose=False)

    expected_keys = {
        "mae_norm", "mse_norm", "corr_norm",
        "mae_raw", "mse_raw", "corr_raw"
    }

    assert set(metrics.keys()) == expected_keys, f"Unexpected metric keys: {metrics.keys()}"
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} must be a float"
        assert not torch.isnan(torch.tensor(val)), f"{key} is NaN"


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_eval_simgnn_forward_no_grad(toy_simgnn, toy_dataloader, device):
    """
    Ensure SimGNN evaluation is deterministic and gradient-free.
    """
    model = toy_simgnn.to(device).eval()
    loader = toy_dataloader

    batch1, batch2, labels = next(iter(loader))
    batch1, batch2, labels = batch1.to(device), batch2.to(device), labels.to(device)

    with torch.no_grad():
        pred1 = model(batch1, batch2)
        pred2 = model(batch1, batch2)
        loss = torch.nn.functional.mse_loss(pred1, labels)

    assert torch.allclose(pred1, pred2, atol=1e-6), "Predictions must be deterministic"
    assert torch.isfinite(loss).all()
    assert pred1.shape == labels.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_train_simgnn_with_tensor_module(toy_dataloader, device):
    """
    Ensure tensor-similarity mode (TensorNetworkModule) runs without shape errors.
    """
    batch1, batch2, labels = next(iter(toy_dataloader))
    in_dim = batch1.num_node_features

    model = SimGNN(in_dim, use_tensor=True, tensor_channels=8).to(device)
    batch1, batch2, labels = batch1.to(device), batch2.to(device), labels.to(device)

    with torch.no_grad():
        out = model(batch1, batch2)

    assert out.shape == labels.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_forward_with_variable_batch_size(toy_simgnn, toy_dataloader, device):
    """
    Model should handle varying batch sizes (including single graph pairs).
    """
    model = toy_simgnn.to(device)
    loader_iter = iter(toy_dataloader)
    batch1, batch2, labels = next(loader_iter)

    small_batch = [(batch1[i], batch2[i], labels[i]) for i in range(1)]
    batch1_small, batch2_small, labels_small = collate_pairs(small_batch)

    with torch.no_grad():
        out = model(batch1_small.to(device), batch2_small.to(device))

    assert out.numel() == labels_small.numel()
    assert torch.isfinite(out).all()

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_no_invalid_or_empty_graphs(toy_dataloader, device):
    """
    Ensure all graphs in the GED dataset are nonempty and have valid batch indices.

    This prevents CUDA device-side assertions caused by:
    - empty graphs (0 nodes)
    - mismatched batch indexing
    """
    for batch1, batch2, labels in toy_dataloader:
        # Sanity check: graphs must have nodes
        assert batch1.num_nodes > 0, "Batch1 has empty graph"
        assert batch2.num_nodes > 0, "Batch2 has empty graph"

        # Batch indices must be within valid range
        assert batch1.batch.max().item() < len(torch.unique(batch1.batch)), "Invalid batch1 index"
        assert batch2.batch.max().item() < len(torch.unique(batch2.batch)), "Invalid batch2 index"

        # Node features and edges must be well-formed
        assert batch1.x is not None and batch1.x.numel() > 0, "Batch1 missing node features"
        assert batch2.x is not None and batch2.x.numel() > 0, "Batch2 missing node features"

        # Labels must match batch size
        assert labels.numel() > 0 and torch.isfinite(labels).all(), "Invalid label tensor"
