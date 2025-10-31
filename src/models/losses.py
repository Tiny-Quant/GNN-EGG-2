"""
Losses for generator training.

Includes:
- EdgePenalty: encourages sparse graphs via L2 + budget softplus
- SoftContrastiveEmbedLoss: pulls generated graph embeddings toward the
  target-class mean and pushes away from non-target means, weighted by the
  explainee's own softmax predictions (no GT leakage).
- PredictionConfidenceLoss: maximizes explainee confidence on a target class.

The losses expect:
- a frozen explainee GNN returning logits [B, C] for a PyG Batch/Data input
- precomputed classwise means per-layer: Dict[layer_name, Tensor[C, D_l]]
- generator output dict with:
    adj:       [B, N, N] in [0,1]
    cont_node: [B, N, F_c]  (optional)
    dis_node:  [B, N, ΣC_k] (optional)
- optional: dis/cont edge features can be added later; the embed loss
  uses the explainee forward so include them in the Data if available.

A small straight-through trick is used to discretize edges for PyG Data,
but still pass gradients to the generator through adj.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


# ---------------------------
# Simple utilities
# ---------------------------

def _concat_node_features(cont: Optional[torch.Tensor],
                          dis: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate continuous and discrete node features along the last dim.
    If one is None, returns the other. If both None, returns a single zero feature.
    Shapes (batched):
        cont: [B, N, F_c] or None
        dis:  [B, N, F_d] or None
    Returns:
        X: [B, N, F], with F >= 1
    """
    if cont is None and dis is None:
        return torch.zeros(1, 1, 1, device='cpu')
    if cont is None:
        return dis
    if dis is None:
        return cont
    return torch.cat([cont, dis], dim=-1)


def _straight_through_binary(mask_probs: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """
    Straight-through estimator for binary adjacency:
        hard = (probs > thresh).float()
        STE  = hard.detach() + probs - probs.detach()
    """
    hard = (mask_probs > thresh).float()
    return hard.detach() + mask_probs - mask_probs.detach()


def _build_data_from_gen_output(
    out: Dict[str, torch.Tensor],
    thresh: float = 0.5,
) -> Batch:
    """
    Construct a PyG Batch of graphs from generator outputs.

    Inputs (B,N,...):
        - 'adj'       : [B, N, N]
        - 'cont_node' : [B, N, F_c] or missing
        - 'dis_node'  : [B, N, F_d] or missing

    Returns:
        Batch with:
            x: [sum_i N_i, F]
            edge_index: long [2, E_total]
            batch: [sum_i N_i] graph id per node
    """
    assert "adj" in out, "Generator output must contain 'adj'"

    adj = out["adj"]                     # [B, N, N]
    cont = out.get("cont_node", None)    # [B, N, F_c] or None
    dis  = out.get("dis_node",  None)    # [B, N, F_d] or None

    X = _concat_node_features(cont, dis) # [B, N, F]
    device = (X.device if X is not None else adj.device)

    B, N, _ = adj.shape
    # Straight-through hard edges
    A_ste = _straight_through_binary(adj, thresh=thresh)  # [B, N, N]

    data_list: List[Data] = []
    for b in range(B):
        # Edge indices for graph b
        idx = (A_ste[b] > 0.5).nonzero(as_tuple=False)  # [E, 2]
        if idx.numel() == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index = idx.t().contiguous()

        x_b = X[b] if X is not None else torch.zeros(N, 1, device=device)
        data_list.append(Data(x=x_b, edge_index=edge_index))

    return Batch.from_data_list(data_list)


# ---------------------------
# Edge sparsity penalty
# ---------------------------

class EdgePenalty(nn.Module):
    """
    Encourages sparse adjacency by combining:
        L2 penalty on probabilities  +  Soft budget via softplus

    Args:
        edge_budget: target upper bound on total sum of adj entries.
    """
    def __init__(self, edge_budget: float = 0.0):
        super().__init__()
        self.edge_budget = float(edge_budget)

    def forward(self, adj_probs: torch.Tensor) -> torch.Tensor:
        """
        adj_probs: [B, N, N] in [0,1]
        """
        l2 = torch.norm(adj_probs, p=2)
        budget = F.softplus(adj_probs.sum() - self.edge_budget) ** 2
        return l2 + budget


# ---------------------------
# Soft contrastive embedding loss
# ---------------------------

class SoftContrastiveEmbedLoss(nn.Module):
    """
    Pulls generated graph embeddings toward the target class means and pushes
    away from non-target class means, weighted by the explainee's own softmax.

    For each layer ℓ (tracked in classwise_means):
      Let E_ℓ ∈ R^{B×D_ℓ} be pooled generated embeddings at layer ℓ.
      Let M_ℓ ∈ R^{C×D_ℓ} be class means (precomputed).
      Let p ∈ R^{B×C} be explainee softmax probabilities on generated graphs.

      Define cosine distance d_ℓ(i, c) = 1 - cos(E_ℓ[i], M_ℓ[c]).
      Final layer loss (averaged over batch & classes with signed weights):
          L_ℓ = mean_i  sum_c  p[i,c] * s(c) * d_ℓ(i, c)
          where s(c) = -1 if c==target_class else +1.

      Total loss is average over layers present in classwise_means.

    Notes:
      - model is assumed FROZEN (eval mode outside).
      - classwise_means keys must match layer names discoverable by hooks.
    """
    def __init__(
        self,
        explainee: nn.Module,
        classwise_means: Dict[str, torch.Tensor],
        target_class: int,
        temperature: float = 1.0,
        thresh: float = 0.5,
        layer_names: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.explainee = explainee
        self.classwise_means = classwise_means
        self.target_class = int(target_class)
        self.temperature = float(temperature)
        self.thresh = float(thresh)

        # Which layers to track at runtime
        self.layer_names = list(layer_names) if layer_names is not None else list(classwise_means.keys())

    def _call_explainee(self, batch: Batch) -> torch.Tensor:
        """
        Robustly call explainee on a PyG Batch. Supports models that either
        accept a single Batch/Data OR (x, edge_index, batch_vector).
        """
        try:
            return self.explainee(batch)  # type: ignore[arg-type]
        except TypeError:
            return self.explainee(batch.x, batch.edge_index, batch.batch)  # type: ignore[misc]

    def _capture_embeddings(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Run explainee forward with hooks, return pooled [B, D_l] per layer and probs.
        """
        from torch_geometric.nn import global_mean_pool  # local import (avoid circulars)
        acts, handles = {}, []

        # register only layers for which we have means
        def _mk_hook(name):
            def hook(_m, _i, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                acts[name] = out
            return hook

        # map name->module and attach
        named = dict(self.explainee.named_modules())
        for name in self.layer_names:
            if name in named:
                handles.append(named[name].register_forward_hook(_mk_hook(name)))

        logits = self._call_explainee(batch)     # [B, C]
        probs = F.softmax(logits, dim=-1)        # [B, C]

        # pool any node-level activations to [B, D]
        pooled: Dict[str, torch.Tensor] = {}
        for name, A in acts.items():
            if A.dim() == 2 and hasattr(batch, "batch"):
                if A.size(0) == batch.batch.size(0):
                    pooled[name] = global_mean_pool(A, batch.batch)  # [B, D]
                else:
                    pooled[name] = A.mean(dim=0, keepdim=True)
            elif A.dim() == 1:
                pooled[name] = A.unsqueeze(0)
            else:
                if A.size(0) == probs.size(0):
                    pooled[name] = A
                else:
                    pooled[name] = A.mean(dim=0, keepdim=True)

        # cleanup
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        return {"pooled": pooled, "probs": probs}

    def forward_components(self, gen_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Return separated components for logging:
          - pull: toward target class
          - push: away from non-target classes
        """
        batch = _build_data_from_gen_output(gen_out, thresh=self.thresh)
        caps = self._capture_embeddings(batch)
        pooled = caps["pooled"]   # Dict[layer, [B,D]]
        probs  = caps["probs"]    # [B,C]

        C = probs.size(-1)
        pull_terms, push_terms = [], []

        for name, E in pooled.items():
            if name not in self.classwise_means:
                continue
            M = self.classwise_means[name].to(E.device)  # [C, D]
            E_norm = F.normalize(E, dim=-1)
            M_norm = F.normalize(M, dim=-1)
            cos = E_norm @ M_norm.t()                    # [B, C]
            dist = (1.0 - cos).clamp_min(0.0)            # [B, C]

            # signed weighting
            # pull: c == target
            pull = (dist * probs)[:, self.target_class].mean()
            # push: c != target
            mask = torch.ones(C, device=probs.device, dtype=torch.bool)
            mask[self.target_class] = False
            if mask.any():
                push = (dist * probs)[:, mask].mean()
            else:
                push = torch.zeros((), device=probs.device)
            pull_terms.append(pull)
            push_terms.append(push)

        if not pull_terms:
            zero = torch.tensor(0.0, device=probs.device, requires_grad=True)
            return {"pull": zero, "push": zero}

        return {
            "pull": torch.stack(pull_terms).mean(),
            "push": torch.stack(push_terms).mean(),
        }

    def forward(self, gen_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Scalar loss: (pull + push) averaged across tracked layers.
        """
        comps = self.forward_components(gen_out)
        return comps["pull"] + comps["push"]


# ---------------------------
# Prediction confidence loss
# ---------------------------

class PredictionConfidenceLoss(nn.Module):
    """
    Encourages the explainee to predict the TARGET class with high confidence.
    Defined as negative log-probability of the target class:

        L_pred = - E_b [ log softmax(logits_b)[target_class] ]

    Works on the differentiable Batch created from generator outputs.
    """
    def __init__(self, explainee: nn.Module, target_class: int, thresh: float = 0.5):
        super().__init__()
        self.explainee = explainee
        self.target_class = int(target_class)
        self.thresh = float(thresh)

    def _call_explainee(self, batch: Batch) -> torch.Tensor:
        try:
            return self.explainee(batch)  # type: ignore[arg-type]
        except TypeError:
            return self.explainee(batch.x, batch.edge_index, batch.batch)  # type: ignore[misc]

    def forward(self, gen_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = _build_data_from_gen_output(gen_out, thresh=self.thresh)
        logits = self._call_explainee(batch)           # [B, C]
        log_probs = F.log_softmax(logits, dim=-1)      # [B, C]
        return -log_probs[:, self.target_class].mean()
