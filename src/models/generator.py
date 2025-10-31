import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List


class GraphGenerator(nn.Module):

    def __init__(
        self,
        max_nodes: int,
        num_cont_node_feats: int = 0,
        dis_node_blocks: Optional[List[int]] = None,
        num_cont_edge_feats: int = 0,
        dis_edge_blocks: Optional[List[int]] = None,
        temperature: float = 1.0,
        allow_self_loops: bool = True,
        symmetric_adj: bool = True,
        batch_size: int = 1,
    ):
        super().__init__()
        assert max_nodes > 0, "max_nodes must be positive"
        assert batch_size > 0, "batch_size must be positive"

        self.max_nodes = int(max_nodes)
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.allow_self_loops = bool(allow_self_loops)
        self.symmetric_adj = bool(symmetric_adj)

        # Device anchor
        self._dev_param = nn.Parameter(torch.empty(0), requires_grad=False)

        # ---- adjacency ----
        self.adj_logits = nn.Parameter(
            torch.zeros(batch_size, max_nodes, max_nodes)
        )
        nn.init.xavier_uniform_(self.adj_logits)

        # --------------------------
        # Node features
        # --------------------------
        # ---- continuous node features ----
        self.num_cont_node_feats = int(num_cont_node_feats)
        if num_cont_node_feats > 0:
            self.cont_node = nn.Parameter(
                torch.zeros(self.batch_size, self.max_nodes, self.num_cont_node_feats)
            )
            nn.init.xavier_uniform_(self.cont_node)
        else:
            self.register_parameter("cont_node", None)

        # ---- discrete node features ----
        self.dis_node_blocks = dis_node_blocks or [] 
        self.dis_node_logits = nn.ParameterList()
        for num_classes in self.dis_node_blocks:
            logits = nn.Parameter(
                torch.randn(self.batch_size, self.max_nodes, num_classes)
            )
            nn.init.xavier_uniform_(logits)
            self.dis_node_logits.append(logits)

        # --------------------------
        # Edge features
        # --------------------------
        # Continuous edge features
        self.num_cont_edge_feats = int(num_cont_edge_feats)
        if self.num_cont_edge_feats > 0:
            self.cont_edge = nn.Parameter(
                torch.zeros(
                    self.batch_size, self.max_nodes, self.max_nodes, self.num_cont_edge_feats
                )
            )
            nn.init.xavier_uniform_(self.cont_edge)
        else:
            self.register_parameter("cont_edge", None)

        # Discrete edge features (multiple blocks allowed)
        self.dis_edge_blocks = list(dis_edge_blocks or [])
        self.dis_edge_logits = nn.ParameterList()
        for Dk in self.dis_edge_blocks:
            logits = nn.Parameter(
                torch.zeros(
                    self.batch_size, self.max_nodes, self.max_nodes, int(Dk)
                )
            )
            nn.init.xavier_uniform_(logits)
            self.dis_edge_logits.append(logits)
    
    @property
    def device(self) -> torch.device:
        return self._dev_param.device

    def _symmetrize_2d(self, M: torch.Tensor) -> torch.Tensor:
        """Symmetrize along the last two dims: (...xNxN)."""
        return 0.5 * (M + M.transpose(-1, -2))

    def _symmetrize_edge4d(self, T: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize (B x N x N x C) by swapping node dims (1, 2),
        leaving the class/channel dimension (last dim) untouched.
        """
        return 0.5 * (T + T.transpose(1, 2))
    
    def forward(self):
        B, N = self.batch_size, self.max_nodes

        # Adjacency probabilities
        adj_logits = self.adj_logits
        if self.symmetric_adj:
            adj_logits = 0.5 * (adj_logits + adj_logits.transpose(-1, -2))
        adj = torch.sigmoid(adj_logits)

        if not self.allow_self_loops:
            eye = torch.eye(N, device=self.device).unsqueeze(0).expand_as(adj)
            adj = adj * (1 - eye)

        # ---- continuous node features ----
        if self.cont_node is not None:
            cont_node = self.cont_node
        else:
            # provide a minimal, well-formed tensor even when disabled
            cont_node = torch.zeros(B, N, 0, device=self.device)

        # ---- discrete node features (blocks) ----
        dis_node_blocks_out = []
        for logits in self.dis_node_logits:
            # (B, N, Ck) -> per-node categorical, relaxed one-hot
            block = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
            dis_node_blocks_out.append(block)

        if dis_node_blocks_out:
            dis_node = torch.cat(dis_node_blocks_out, dim=-1)  # (B, N, ΣCk_n)
        else:
            dis_node = torch.zeros(B, N, 0, device=self.device)

        # ---- continuous edge features ----
        if self.cont_edge is not None:
            cont_edge = self.cont_edge
            if self.symmetric_adj:
                cont_edge = self._symmetrize_edge4d(cont_edge)  # (B, N, N, F_ec)
        else:
            cont_edge = torch.zeros(B, N, N, 0, device=self.device)

        # ---- discrete edge features (blocks) ----
        dis_edge_blocks_out = []
        for logits in self.dis_edge_logits:
            # Optionally symmetrize logits over (N,N) before softmax
            le = logits
            # if self.symmetric_adj:
            #     le = self._symmetrize_edge4d(le)  # (B, N, N, Dk)

            # Per-edge categorical distribution
            # Softmax over last dim (classes) independently for each (B,i,j)
            block = F.gumbel_softmax(le, tau=self.temperature, hard=False, dim=-1)
            if self.symmetric_adj:
                block = self._symmetrize_edge4d(block)

            dis_edge_blocks_out.append(block)

        if dis_edge_blocks_out:
            dis_edge = torch.cat(dis_edge_blocks_out, dim=-1)  # (B, N, N, ΣDk_e)
        else:
            dis_edge = torch.zeros(B, N, N, 0, device=self.device)
        
        # ✅ Output format consistent with tests
        result = {
            "adj": adj,
            "cont_node": cont_node,
            "dis_node": dis_node,
            "cont_edge": cont_edge, 
            "dis_edge": dis_edge, 
        }

        return result