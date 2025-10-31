"""
Utilities for extracting per-class, per-layer average embeddings from a
FROZEN explainee GNN.

- Auto-detects GNN layers (torch_geometric.nn.MessagePassing) + Linear heads.
- Captures layer activations via forward hooks.
- Pools node-level activations -> graph embeddings with global mean pool.
- Aggregates classwise averages using SOFT weights from the explainee's own
  predictions (no ground-truth leakage): w_c = p(y=c | x).

Returns:
    ClasswiseMeans: Dict[layer_name, Tensor[C, D_l]]
    where:
        C  = number of classes
        D_l = hidden size of layer l
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing


# ---------------------------
# Layer discovery & hooks
# ---------------------------

def discover_embedding_layers(model: nn.Module) -> OrderedDict[str, nn.Module]:
    """
    Heuristically discover embedding-producing layers in a GNN.
    - Includes all torch_geometric MessagePassing layers (GINConv, GCNConv, etc.)
    - Includes Linear layers (useful for readouts/heads if needed).
    The order follows model.named_modules() pre-order traversal.
    """
    candidates = (MessagePassing, nn.Linear)
    layers: "OrderedDict[str, nn.Module]" = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, candidates):
            layers[name] = module

    if not layers:
        raise RuntimeError(
            "No embedding layers found. Ensure your explainee contains "
            "torch_geometric MessagePassing or nn.Linear modules."
        )
    return layers


def register_activation_hooks(
    model: nn.Module,
    layer_names: Iterable[str]
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Register forward hooks to capture activations by name.
    Returns a dict that will be filled during forward() and a list of handles.
    """
    activations: Dict[str, torch.Tensor] = {}

    def _mk_hook(key: str):
        def hook(_module, _inp, out):
            # Some layers return tuples; take the primary tensor
            if isinstance(out, (tuple, list)):
                out = out[0]
            activations[key] = out
        return hook

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name in layer_names:
        # Defensive: skip names that may not exist at runtime (tied wrappers)
        try:
            module = dict(model.named_modules())[name]
        except KeyError:
            continue
        handles.append(module.register_forward_hook(_mk_hook(name)))
    return activations, handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# ---------------------------
# Classwise means (soft)
# ---------------------------

@torch.no_grad()
def compute_classwise_means(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
    layer_names: Iterable[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-class, per-layer average embeddings using SOFT class weights.

    For each batch sample i:
        - Get logits -> softmax p_i ∈ R^C
        - For each tracked layer ℓ, pool node activations to graph vec g_i^ℓ
        - Accumulate weighted sums: sum_c^ℓ += p_i[c] * g_i^ℓ
        - Accumulate weights: sum_w^ℓ[c] += p_i[c]
    Final mean per class & layer:
        M_ℓ[c] = sum_c^ℓ[c] / max(sum_w^ℓ[c], eps)

    Args:
        model: frozen explainee (set to eval)
        dataloader: PyG DataLoader of graphs
        device: torch device
        num_classes: #classes in the explainee head
        layer_names: optional subset of layer names to track. If None,
                     auto-discover with discover_embedding_layers().

    Returns:
        means: Dict[layer_name, Tensor [C, D_ℓ]]
    """
    model.eval()
    model.to(device)

    # discover layers if needed
    if layer_names is None:
        layers = discover_embedding_layers(model)
        layer_names = list(layers.keys())
    else:
        # keep order stable
        layer_names = list(layer_names)

    sums: Dict[str, torch.Tensor] = {}
    weights: Dict[str, torch.Tensor] = {}
    dims: Dict[str, int] = {}

    # init accumulators lazily on first batch
    for batch in dataloader:
        batch = batch.to(device)
        # set up hooks
        acts, handles = register_activation_hooks(model, layer_names)

        logits = model(batch)  # [B, C] expected
        probs = F.softmax(logits, dim=-1)  # [B, C]
        assert probs.size(-1) == num_classes, "Explainee head size != num_classes"

        # pool each layer's node activations into graph embeddings
        pooled: Dict[str, torch.Tensor] = {}
        for name, act in acts.items():
            # --- Node-level activations: use batch mapping if compatible ---
            if act.dim() == 2 and hasattr(batch, "batch"):
                if act.size(0) == batch.batch.size(0):
                    pooled[name] = global_mean_pool(act, batch.batch)  # [B, D]
                else:
                    # fallback: act has fewer nodes than batch expects (e.g., already pooled)
                    pooled[name] = act.mean(dim=0, keepdim=True)        # [1, D]
            elif act.dim() == 1:
                pooled[name] = act.unsqueeze(0)  # [1, D=1]
            else:
                # Already [B, D] or similar
                if act.size(0) == probs.size(0):
                    pooled[name] = act
                else:
                    # Fallback for mismatched dimensions
                    pooled[name] = act.mean(dim=0, keepdim=True)        # [1, D]

        remove_hooks(handles)

        B = probs.size(0)
        for name, G in pooled.items():
            # Ensure [B, D]
            if G.size(0) != B:
                # tile or pad as needed; fallback: repeat first row
                if G.size(0) == 1:
                    G = G.expand(B, -1)
                else:
                    G = G[:B]

            D = G.size(1)
            if name not in sums:
                sums[name] = torch.zeros(num_classes, D, device=device)
                weights[name] = torch.zeros(num_classes, device=device)
                dims[name] = D

            # accumulate soft-weighted sums per class
            # sums[name][c] += sum_i probs[i, c] * G[i]
            sums[name] += probs.transpose(0, 1) @ G  # [C, D]
            weights[name] += probs.sum(dim=0)        # [C]

    eps = 1e-8
    means: Dict[str, torch.Tensor] = {}
    for name in sums:
        w = weights[name].clamp_min(eps).unsqueeze(1)  # [C,1]
        means[name] = sums[name] / w                   # [C, D]

    return means
