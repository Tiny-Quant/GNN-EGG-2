from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union, Optional, Sequence

import torch
from torch_geometric.data import Data, Dataset


@dataclass
class AdapterSpec:
    """
    Normalized feature spec inferred from a (PyG) dataset.

    Fields:
        max_nodes: largest node count observed across graphs
        num_cont_node_feats: number of continuous node features
        dis_node_blocks: list of block sizes for discrete node features
        num_cont_edge_feats: number of continuous edge features
        dis_edge_blocks: list of block sizes for discrete edge features
    """
    max_nodes: int
    num_cont_node_feats: int
    dis_node_blocks: List[int]
    num_cont_edge_feats: int
    dis_edge_blocks: List[int]


def _is_binary_col(col: torch.Tensor) -> bool:
    """True if the column strictly contains only 0s and 1s."""
    return col.ndim == 1 and bool(((col == 0) | (col == 1)).all())


def _binary_runs(mask: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Given a boolean mask over columns marking binary columns, return contiguous
    (start, end) half-open intervals for runs of True values.
    Example:
      mask = [F,T,T,F,T] -> [(1,3), (4,5)]
    """
    runs: List[Tuple[int, int]] = []
    C = mask.numel()
    i = 0
    while i < C:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < C and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _apply_block_hints(run_len: int, hint_sizes: Optional[Sequence[int]]) -> List[int]:
    """
    Decide how to split a binary run of length run_len into one-hot blocks.

    - If hint_sizes is None: treat whole run as a single categorical attribute,
      i.e. [run_len].
    - If hint_sizes is provided: we trust it completely and return list(hint_sizes),
      but we assert it sums to run_len.

    This lets us support both:
      - MUTAG-style: a single 7-way atom-type one-hot -> [7]
      - synthetic test: run_len=5 but we want 2 attributes -> hint_sizes=[2,3]
    """
    if hint_sizes is None:
        return [run_len]
    total = sum(int(x) for x in hint_sizes)
    assert total == run_len, (
        f"block_hints {list(hint_sizes)} do not sum to run length {run_len}"
    )
    return list(int(x) for x in hint_sizes)


def _find_blocks_and_continuous(
    X: torch.Tensor,
    block_hints: Optional[List[List[int]]] = None,
) -> Tuple[int, List[int]]:
    """
    Detect one-hot (binary) blocks and continuous columns from X (N, C):

    Steps:
      1. Mark which columns are strictly binary {0,1}.
      2. Group consecutive binary cols into 'runs'.
      3. For each run:
         - If block_hints is provided, split according to that hint.
           block_hints is a list aligned with runs, e.g., [[2,3],[4]].
         - Else, treat full run as a single block (no splitting).
      4. Columns not covered by those discrete blocks are counted
         as continuous features.

    Returns:
        (num_continuous_columns, [block_sizes...])
    """
    if X is None or X.ndim != 2 or X.size(1) == 0:
        return 0, []

    C = X.size(1)
    bin_mask = torch.zeros(C, dtype=torch.bool)

    for c in range(C):
        if _is_binary_col(X[:, c]):
            bin_mask[c] = True

    runs = _binary_runs(bin_mask)  # list of (start, end)

    used = torch.zeros(C, dtype=torch.bool)
    blocks: List[int] = []

    # iterate over runs, optionally apply user hints per run
    for run_idx, (s, e) in enumerate(runs):
        run_len = e - s

        # pick hint for this run if provided
        if block_hints is not None and run_idx < len(block_hints):
            split_sizes = _apply_block_hints(run_len, block_hints[run_idx])
        else:
            split_sizes = _apply_block_hints(run_len, None)  # -> [run_len]

        # mark columns as used and record block sizes
        cursor = s
        for size_k in split_sizes:
            blocks.append(size_k)
            used[cursor: cursor + size_k] = True
            cursor += size_k

    num_cont = int((~used).sum().item())
    return num_cont, blocks


class GeneratorAdapter:
    """
    Inspect a (PyG) dataset or iterable of `Data` and infer a normalized
    feature spec.

    Optional arguments:
    - node_block_hints: List of per-run split hints for node features.
      Example: [[2,3]] means "for the *first* binary run, split into 2 then 3".
    - edge_block_hints: Same idea, but for edge_attr.

    If no hints are provided, we assume each binary run is a single one-hot
    categorical block. That matches real-world TU datasets (MUTAG, PROTEINS).
    """

    def __init__(
        self,
        dataset: Union[Dataset, Iterable[Data]],
        node_block_hints: Optional[List[List[int]]] = None,
        edge_block_hints: Optional[List[List[int]]] = None,
    ):
        # normalize to a list of Data
        if hasattr(dataset, "__len__") and len(dataset) > 0 and hasattr(dataset, "__getitem__"):
            data_list = [dataset[i] for i in range(len(dataset))]
        else:
            data_list = list(dataset)
        assert len(data_list) > 0, "Dataset must not be empty."

        # ---- max nodes across all graphs ----
        max_nodes = 0
        for d in data_list:
            if hasattr(d, "x") and d.x is not None:
                n = int(d.x.size(0))
            elif hasattr(d, "num_nodes") and d.num_nodes is not None:
                n = int(d.num_nodes)
            else:
                n = 0
            max_nodes = max(max_nodes, n)

        ex: Data = data_list[0]

        # ---- Node features ----
        num_cont_node_feats = 0
        dis_node_blocks: List[int] = []
        if hasattr(ex, "x") and ex.x is not None and ex.x.ndim == 2:
            num_cont_node_feats, dis_node_blocks = _find_blocks_and_continuous(
                ex.x,
                block_hints=node_block_hints,
            )

        # ---- Edge features ----
        num_cont_edge_feats = 0
        dis_edge_blocks: List[int] = []
        if hasattr(ex, "edge_attr") and ex.edge_attr is not None:
            EA = ex.edge_attr
            if EA.ndim == 1:
                # single scalar edge feature treated as continuous
                num_cont_edge_feats = 1
            elif EA.ndim == 2:
                num_cont_edge_feats, dis_edge_blocks = _find_blocks_and_continuous(
                    EA,
                    block_hints=edge_block_hints,
                )

        self._spec = AdapterSpec(
            max_nodes=max_nodes,
            num_cont_node_feats=num_cont_node_feats,
            dis_node_blocks=dis_node_blocks,
            num_cont_edge_feats=num_cont_edge_feats,
            dis_edge_blocks=dis_edge_blocks,
        )

    # Convenience accessors
    @property
    def spec(self) -> AdapterSpec:
        return self._spec

    @property
    def max_nodes(self) -> int:
        return self._spec.max_nodes

    @property
    def num_cont_node_feats(self) -> int:
        return self._spec.num_cont_node_feats

    @property
    def dis_node_blocks(self) -> List[int]:
        return self._spec.dis_node_blocks

    @property
    def num_cont_edge_feats(self) -> int:
        return self._spec.num_cont_edge_feats

    @property
    def dis_edge_blocks(self) -> List[int]:
        return self._spec.dis_edge_blocks
