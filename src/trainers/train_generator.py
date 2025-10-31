"""
Trainer for class-conditional GraphGenerator using:
  - Soft contrastive embedding loss (pull-to-target, push-from-others)
  - Prediction confidence loss (maximize explainee's prob for target class)
  - Edge sparsity penalty

Distance-based terms (SimGNN, etc.) can be added later.

Workflow:
  1. You train an explainee GNN separately and save it.
  2. You compute per-class mean embeddings of that explainee on real data.
  3. For each class c, you train one generator to (a) match that class's embedding,
     (b) be confidently predicted as class c, and (c) stay sparse via edge penalty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from src.models.generator import GraphGenerator
from src.models.losses import (
    EdgePenalty,
    SoftContrastiveEmbedLoss,
    PredictionConfidenceLoss,
)
from src.utils.embeddings import discover_embedding_layers


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------

@dataclass
class GenTrainConfig:
    target_class: int
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 1
    lambda_edge: float = 1.0
    lambda_pred: float = 1.0
    edge_thresh: float = 0.5  # how we binarize adj in SoftContrastiveEmbedLoss/PredictionConfidenceLoss
    layer_names: Optional[Sequence[str]] = None
    save_dir: str = "checkpoints/generators"
    save_name: Optional[str] = None  # default -> f"gen_class{target_class}.pt"
    log_every: int = 20


# ---------------------------------------------------------------------
# Trainer object
# ---------------------------------------------------------------------

class GeneratorTrainer:
    """
    Handles the training loop for a single GraphGenerator targeting one class.

    Loss:
        L = (pull + push) + lambda_pred * L_pred + lambda_edge * L_edge

    explainee is frozen the entire time.
    """

    def __init__(
        self,
        explainee: nn.Module,
        num_classes: int,
        classwise_means: Dict[str, torch.Tensor],
        device: torch.device,
        lambda_edge: float = 1.0,
        lambda_pred: float = 1.0,
        edge_thresh: float = 0.5,
        layer_names: Optional[Sequence[str]] = None,
    ):
        # freeze explainee
        self.explainee = explainee.to(device).eval()
        for p in self.explainee.parameters():
            p.requires_grad_(False)

        self.num_classes = num_classes  # e.g. 2 for MUTAG
        self.means = classwise_means    # dict[layer_name] -> [num_classes, D]
        self.device = device

        # pick which hidden layers we supervise against
        if layer_names is None:
            discovered = list(discover_embedding_layers(self.explainee).keys())
            # only keep layers that exist in the means dict
            self.layer_names = [n for n in discovered if n in self.means]
        else:
            self.layer_names = list(layer_names)

        self.edge_pen = EdgePenalty()
        self.lambda_edge = float(lambda_edge)
        self.lambda_pred = float(lambda_pred)
        self.edge_thresh = float(edge_thresh)

        # last-logged components
        self._last_log = None  # filled after first step

    def _embed_components(
        self,
        gen_out: Dict[str, torch.Tensor],
        target_class: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Return dict with 'pull' and 'push' components.
        """
        loss_mod = SoftContrastiveEmbedLoss(
            explainee=self.explainee,
            classwise_means=self.means,
            target_class=target_class,
            thresh=self.edge_thresh,
            layer_names=self.layer_names,
        )
        return loss_mod.forward_components(gen_out)

    def _pred_loss(
        self,
        gen_out: Dict[str, torch.Tensor],
        target_class: int,
    ) -> torch.Tensor:
        """
        Negative log-prob of target class under the explainee.
        """
        loss_mod = PredictionConfidenceLoss(
            explainee=self.explainee,
            target_class=target_class,
            thresh=self.edge_thresh,
        )
        return loss_mod(gen_out)

    def step(self, generator: GraphGenerator, target_class: int) -> torch.Tensor:
        """
        One optimization step: compute losses, backprop, and return total loss (Tensor).
        Also caches separated components in self._last_log for pretty printing.
        """
        generator.train()
        out = generator()

        # components
        embed_comps = self._embed_components(out, target_class)
        loss_pull = embed_comps["pull"]
        loss_push = embed_comps["push"]
        loss_pred = self._pred_loss(out, target_class)
        loss_edge = self.edge_pen(out["adj"])

        total_loss = (loss_pull + loss_push) + self.lambda_pred * loss_pred + self.lambda_edge * loss_edge
        total_loss.backward()

        # cache scalar logs
        self._last_log = {
            "total": float(total_loss.detach()),
            "pull": float(loss_pull.detach()),
            "push": float(loss_push.detach()),
            "pred": float(loss_pred.detach()),
            "edge": float(loss_edge.detach()),
        }
        return total_loss

    def train(self, generator: GraphGenerator, cfg: GenTrainConfig) -> None:
        """
        Full loop: optimize for cfg.epochs and save checkpoint.
        """
        generator.to(self.device)
        opt = optim.Adam(generator.parameters(), lr=cfg.lr)

        save_name = cfg.save_name or f"gen_class{cfg.target_class}.pt"
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = os.path.join(cfg.save_dir, save_name)

        for epoch in range(1, cfg.epochs + 1):
            opt.zero_grad(set_to_none=True)
            total_loss = self.step(generator, cfg.target_class)
            opt.step()

            if epoch % max(1, cfg.log_every) == 0:
                log = self._last_log or {
                    "total": float(total_loss.detach()),
                    "pull": 0.0, "push": 0.0, "pred": 0.0, "edge": 0.0
                }
                print(
                    f"[Gen|class={cfg.target_class}] epoch {epoch:04d} | "
                    f"total={log['total']:.4f} "
                    f"(pull={log['pull']:.4f}, push={log['push']:.4f}, "
                    f"pred={log['pred']:.4f}, edge={log['edge']:.4f})"
                )

        torch.save(generator.state_dict(), ckpt_path)
        print(f"✅ Saved generator for class {cfg.target_class} to: {ckpt_path}")


# ---------------------------------------------------------------------
# Pipeline-facing convenience function
# ---------------------------------------------------------------------

def train_generator(
    *,
    explainee: nn.Module,
    num_classes: int,
    classwise_means: Dict[str, torch.Tensor],
    target_class: int,
    save_path: str,
    device: torch.device,
    # generator arch spec inferred by adapter
    max_nodes: int,
    num_cont_node_feats: int,
    dis_node_blocks: list,
    num_cont_edge_feats: int,
    dis_edge_blocks: list,
    # hyperparams
    lr: float = 1e-3,
    steps: int = 200,
    batch_size: int = 1,
    temperature: float = 1.0,
    symmetric_adj: bool = True,
    allow_self_loops: bool = False,
    lambda_edge: float = 0.1,
    lambda_pred: float = 1.0,
    log_every: int = 20,
) -> None:
    """
    High-level entry point used by pipeline.py to train ONE generator for ONE class.
    """

    # 1. build generator that matches dataset stats
    generator = GraphGenerator(
        max_nodes=max_nodes,
        num_cont_node_feats=num_cont_node_feats,
        dis_node_blocks=dis_node_blocks,
        num_cont_edge_feats=num_cont_edge_feats,
        dis_edge_blocks=dis_edge_blocks,
        temperature=temperature,
        allow_self_loops=allow_self_loops,
        symmetric_adj=symmetric_adj,
        batch_size=batch_size,
    )

    # 2. wrap config
    cfg = GenTrainConfig(
        target_class=target_class,
        epochs=steps,
        lr=lr,
        batch_size=batch_size,
        lambda_edge=lambda_edge,
        lambda_pred=lambda_pred,
        save_dir=str(Path(save_path).parent),
        save_name=Path(save_path).name,
        log_every=log_every,
    )

    # 3. trainer
    trainer = GeneratorTrainer(
        explainee=explainee,
        num_classes=num_classes,
        classwise_means=classwise_means,
        device=device,
        lambda_edge=lambda_edge,
        lambda_pred=lambda_pred,
        edge_thresh=0.5,
        layer_names=None,  # auto-discover shared layers
    )

    # 4. train + save
    trainer.train(generator, cfg)


def precompute_class_means(
    explainee: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    layer_names: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper for compute_classwise_means —
    used in unit tests and pipelines.
    """
    from src.utils.embeddings import compute_classwise_means
    means = compute_classwise_means(
        model=explainee,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
        layer_names=layer_names,
    )
    print(f"🧭 Prepared classwise means for {len(means)} layers")
    return means
