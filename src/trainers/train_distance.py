"""
train_distance.py

Unified training interface for differentiable graph distance models.
Includes SimGNN with automatic GED dataset generation.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from torch_geometric.datasets import TUDataset

from src.models.sim_gnn import SimGNN
from src.datasets.ged_dataset import GEDDataset, collate_pairs, ensure_ged_dataset


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device, alpha=0.5):
    model.train()
    total_loss_norm, total_loss_raw = 0.0, 0.0

    for batch1, batch2, labels_norm in loader:
        batch1, batch2, labels_norm = batch1.to(device), batch2.to(device), labels_norm.to(device)
        optimizer.zero_grad()

        preds_norm = model(batch1, batch2)
        loss_norm = F.mse_loss(preds_norm, labels_norm)

        norm_factors = torch.tensor(
            [loader.dataset.get_norm_factor(i) for i in range(len(labels_norm))],
            dtype=torch.float32,
            device=device,
        )

        preds_raw = preds_norm * norm_factors
        labels_raw = labels_norm * norm_factors
        loss_raw = F.mse_loss(preds_raw, labels_raw)

        loss_norm.backward()
        optimizer.step()

        total_loss_norm += loss_norm.item() * labels_norm.size(0)
        total_loss_raw += loss_raw.item() * labels_norm.size(0)

    n = len(loader.dataset)
    return total_loss_norm / n, total_loss_raw / n


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, verbose=False):
    model.eval()
    preds_norm, labels_norm, norm_factors = [], [], []

    dataset = loader.dataset
    for i, (batch1, batch2, lbl_norm) in enumerate(loader):
        batch1, batch2, lbl_norm = batch1.to(device), batch2.to(device), lbl_norm.to(device)
        out = model(batch1, batch2)
        preds_norm.append(out.detach().cpu())
        labels_norm.append(lbl_norm.detach().cpu())

        start = i * loader.batch_size
        end = min(start + len(lbl_norm), len(dataset))
        norm_factors.extend([dataset.get_norm_factor(j) for j in range(start, end)])

    preds_norm = torch.cat(preds_norm)
    labels_norm = torch.cat(labels_norm)
    norm_factors = torch.tensor(norm_factors[: len(preds_norm)], dtype=torch.float32)

    mae_norm = F.l1_loss(preds_norm, labels_norm).item()
    mse_norm = F.mse_loss(preds_norm, labels_norm).item()
    corr_norm = pearsonr(preds_norm.numpy(), labels_norm.numpy())[0] if len(preds_norm) > 1 else 0.0

    preds_raw = preds_norm * norm_factors
    labels_raw = labels_norm * norm_factors
    mae_raw = F.l1_loss(preds_raw, labels_raw).item()
    mse_raw = F.mse_loss(preds_raw, labels_raw).item()
    corr_raw = pearsonr(preds_raw.numpy(), labels_raw.numpy())[0] if len(preds_raw) > 1 else 0.0

    return {
        "mae_norm": mae_norm,
        "mse_norm": mse_norm,
        "corr_norm": corr_norm,
        "mae_raw": mae_raw,
        "mse_raw": mse_raw,
        "corr_raw": corr_raw,
    }


# ---------------------------------------------------------------------
# Generic SimGNN training interface
# ---------------------------------------------------------------------
def train_distance(
    dataset_name: str,
    data_path: str,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    use_tensor: bool,
    alpha: float,
    save_path: str,
    checkpoint_dir: str,
    device,
    overwrite_data: bool = False,
):
    """
    Train a SimGNN distance approximator on any PyG dataset.
    Automatically regenerates GED dataset if missing or overwrite=True.
    """
    print(f"📦 Loading base dataset: {dataset_name}")
    base_dataset = TUDataset(root="data", name=dataset_name)

    ged_path = ensure_ged_dataset(
        base_graphs=list(base_dataset),
        out_path=data_path,
        alpha=alpha,
        overwrite=overwrite_data,
    )

    ged_dataset = GEDDataset.load(ged_path)
    train_loader = DataLoader(
        ged_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
    )

    in_dim = ged_dataset[0][0].x.size(1)
    model = SimGNN(in_dim, hidden_dim=hidden_dim, use_tensor=use_tensor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"📏 Training SimGNN on '{dataset_name}' ({len(ged_dataset)} pairs)...")

    for epoch in range(1, epochs + 1):
        loss_norm, loss_raw = train_epoch(model, train_loader, optimizer, device, alpha=alpha)
        metrics = evaluate(model, train_loader, device)
        print(
            f"Epoch {epoch:03d} | "
            f"Loss(norm)={loss_norm:.4f} | Loss(raw)={loss_raw:.4f} | "
            f"MAE(norm)={metrics['mae_norm']:.4f} | MAE(raw)={metrics['mae_raw']:.4f} | "
            f"Corr(norm)={metrics['corr_norm']:.3f} | Corr(raw)={metrics['corr_raw']:.3f}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved SimGNN model to {save_path}")

    return model, metrics
