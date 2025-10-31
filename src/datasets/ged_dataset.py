"""
ged_dataset.py

Create, load, and serve synthetic Graph Edit Distance (GED) datasets
for training neural GED approximators (e.g., SimGNN).

Depends on:
- graph_perturb.generate_perturbed_pairs
"""

from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from src.datasets.graph_perturb import generate_perturbed_pairs


class GEDDataset(Dataset):
    """
    Dataset wrapper around (graph_i, graph_j, {"raw": ..., "norm": ..., "factor": ...}) tuples.
    Each element includes both the raw and normalized GED values and the normalization factor.
    """

    def __init__(self, pairs: List[Tuple[Data, Data, Dict[str, float]]]) -> None:
        super().__init__()
        self.pairs: List[Tuple[Data, Data, Dict[str, float]]] = []
        for g1, g2, label_info in pairs:
            # Backward-compatible load
            if isinstance(label_info, dict):
                label_info = {
                    "raw": float(label_info.get("raw", 0.0)),
                    "norm": float(label_info.get("norm", 0.0)),
                    "factor": float(label_info.get("factor", 1.0)),
                }
            else:
                label_info = {"raw": float(label_info), "norm": float(label_info), "factor": 1.0}
            self.pairs.append((g1, g2, label_info))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, torch.Tensor]:
        g1, g2, label_info = self.pairs[idx]
        return g1, g2, torch.tensor(label_info["norm"], dtype=torch.float32)

    def get_raw_ged(self, idx: int) -> float:
        """Access the raw (unnormalized) GED value."""
        return self.pairs[idx][2]["raw"]

    def get_norm_factor(self, idx: int) -> float:
        """Access the normalization factor."""
        return self.pairs[idx][2]["factor"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.pairs, path)

    @classmethod
    def load(cls, path: str) -> "GEDDataset":
        pairs: List[Tuple[Data, Data, Dict[str, float]]] = torch.load(path, weights_only=False)
        return cls(pairs)


def collate_pairs(
    batch: List[Tuple[Data, Data, torch.Tensor]]
) -> Tuple[Batch, Batch, torch.Tensor]:
    g1_list = [item[0] for item in batch]
    g2_list = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch]).float()

    batch1 = Batch.from_data_list(g1_list)
    batch2 = Batch.from_data_list(g2_list)
    return batch1, batch2, labels


def create_and_save_ged_dataset(
    base_dataset: List[Data],
    out_path: str,
    n_perturbations: int = 5,
    node_add_prob: float = 0.1,
    node_remove_prob: float = 0.05,
    edge_add_prob: float = 0.05,
    edge_remove_prob: float = 0.05,
    sample_mode: str = "beta",
    max_edges_per_new_node: int = 4,
    alpha: float = 0.5,
) -> GEDDataset:
    """
    Generate and save a synthetic GED dataset, storing both raw and normalized values.
    """

    print(f"Generating synthetic GED dataset with {n_perturbations} perturbations per graph...")

    pairs = generate_perturbed_pairs(
        base_dataset,
        n_perturbations=n_perturbations,
        base_node_add_prob=node_add_prob,
        base_node_remove_prob=node_remove_prob,
        base_edge_add_prob=edge_add_prob,
        base_edge_remove_prob=edge_remove_prob,
        sample_mode=sample_mode,
        max_edges_per_new_node=max_edges_per_new_node,
    )

    enriched_pairs: List[Tuple[Data, Data, Dict[str, float]]] = []
    for g1, g2, ged_raw in pairs:
        node_term = g1.num_nodes + g2.num_nodes
        edge_term = g1.num_edges + g2.num_edges
        norm_factor = (alpha * node_term + (1 - alpha) * edge_term) / 2.0
        ged_norm = ged_raw / max(norm_factor, 1.0)

        label_info = {
            "raw": float(ged_raw),
            "norm": float(ged_norm),
            "factor": float(norm_factor),
        }

        g1.raw_ged = ged_raw
        g2.raw_ged = ged_raw
        g1.norm_factor = norm_factor
        g2.norm_factor = norm_factor

        enriched_pairs.append((g1, g2, label_info))

    dataset = GEDDataset(enriched_pairs)
    dataset.save(out_path)
    print(f"Saved GED dataset (raw + normalized) to {out_path} ({len(dataset)} pairs)")
    return dataset

def ensure_ged_dataset(base_graphs: List[Data], out_path: str, alpha=0.5, overwrite=False, n_perturbations=3) -> str:
    """Ensure a GED dataset exists on disk; create if missing."""
    if os.path.exists(out_path) and not overwrite:
        print(f"📂 Using existing GED dataset: {out_path}")
        return out_path
    print(f"🧩 Creating new GED dataset at {out_path} ...")
    create_and_save_ged_dataset(base_graphs, out_path, alpha=alpha, n_perturbations=n_perturbations)
    return out_path

if __name__ == "__main__":
    from src.datasets.mutag import load_mutag

    data = load_mutag()
    subset = data[:5]
    out = "data/mutag_ged_smoketest.pt"
    ds = create_and_save_ged_dataset(subset, out_path=out, n_perturbations=2)
    print("First sample (raw, norm, factor):", ds.get_raw_ged(0), ds[0][2].item(), ds.get_norm_factor(0))
