# src/trainers/train_explainee.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from src.models.explainee_gnn import ExplaineeGIN
from src.datasets.mutag import load_mutag
from src.datasets.proteins import load_proteins


def train_explainee(
    dataset_name: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    save_path: str,
    checkpoint_dir: str,
    device: torch.device,
):
    """Train a GNN explainee model (currently supports MUTAG)."""

    # --- Load dataset ---
    if dataset_name.lower() == "mutag":
        dataset = load_mutag()
        num_classes = 2
    elif dataset_name.lower() == "proteins":
        dataset = load_proteins()
        num_classes = 2
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not supported yet.")

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # --- Model ---
    in_dim = dataset[0].x.size(1)
    model = ExplaineeGIN(in_dim, hidden_dim, num_layers, dropout, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
                total += batch.num_graphs

        acc = correct / total
        print(f"[Epoch {epoch:03d}] Loss: {total_loss / len(train_loader.dataset):.4f} | Val Acc: {acc:.3f}")

        # --- Checkpointing ---
        ckpt_path = Path(checkpoint_dir) / f"explainee_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)

    # --- Save final model ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Explainee model saved to {save_path}")

    return model, acc
