import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GINConv
from torch_geometric.data import Batch


class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        def make_mlp(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.ReLU(),
                nn.Linear(out_f, out_f)
            )
        self.conv1 = GINConv(make_mlp(in_dim, hidden_dim))
        self.conv2 = GINConv(make_mlp(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class TensorNetworkModule(nn.Module):
    """
    Tensor-based similarity module similar to SimGNN's tensor layer.

    Computes pairwise bilinear similarities between pooled graph embeddings
    g1 and g2 across multiple tensor channels, and applies a small MLP.
    """
    def __init__(self, dim, channels=8):
        super().__init__()
        self.dim = dim
        self.channels = channels

        # Learnable tensor W_c for each channel
        self.W = nn.Parameter(
            torch.randn(channels, dim, dim) * (1.0 / (dim ** 0.5))
        )

        # Channel combination MLP (no bottlenecking)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),  # output same size
            nn.ReLU()
        )

    def forward(self, g1, g2):
        B = g1.size(0)
        sims = []
        for c in range(self.channels):
            inter = g1 @ self.W[c]
            s = (inter * g2).sum(dim=1, keepdim=True)  # (B, 1)
            sims.append(s)
        S = torch.cat(sims, dim=1)  # (B, channels)
        S = self.fc(S)
        return S  # (B, channels)


class SimGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        hist_bins=16,
        dropout=0.2,
        use_tensor=False,
        tensor_channels=8
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim)
        self.pool = global_mean_pool
        self.hist_bins = hist_bins
        self.use_tensor = use_tensor

        if self.use_tensor:
            self.tensor = TensorNetworkModule(hidden_dim, channels=tensor_channels)
            sim_feat_dim = tensor_channels
        else:
            sim_feat_dim = hist_bins + 3

        mlp_in = hidden_dim * 2 + sim_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _compute_similarity_hist_stats(self, node_embs1, node_embs2, b1, b2):
        device = node_embs1.device
        B = int(max(b1.max().item(), b2.max().item()) + 1)
        sim_feats = []
        for i in range(B):
            m1 = (b1 == i).nonzero(as_tuple=True)[0]
            m2 = (b2 == i).nonzero(as_tuple=True)[0]
            if m1.numel() == 0 or m2.numel() == 0:
                hist = torch.zeros(self.hist_bins, device=device)
                stats = torch.zeros(3, device=device)
                sim_feats.append(torch.cat([hist, stats], dim=0))
                continue
            E1 = node_embs1[m1]
            E2 = node_embs2[m2]
            E1n = F.normalize(E1, p=2, dim=1)
            E2n = F.normalize(E2, p=2, dim=1)
            sim_mat = torch.matmul(E1n, E2n.t())
            flat = sim_mat.reshape(-1)
            hist = torch.histc(flat, bins=self.hist_bins, min=-1.0, max=1.0)
            if hist.sum() > 0:
                hist = hist / hist.sum()
            mean_sim = flat.mean()
            max_sim = flat.max()
            std_sim = flat.std(unbiased=False) if flat.numel() > 1 else torch.tensor(0.0, device=device)
            stats = torch.stack([mean_sim, max_sim, std_sim])
            sim_feats.append(torch.cat([hist, stats], dim=0))
        return torch.stack(sim_feats, dim=0)

    def forward(self, data1: Batch, data2: Batch):
        x1, e1, b1 = data1.x, data1.edge_index, data1.batch
        x2, e2, b2 = data2.x, data2.edge_index, data2.batch

        node_embs1 = self.encoder(x1, e1, b1)
        node_embs2 = self.encoder(x2, e2, b2)

        g_emb1 = self.pool(node_embs1, b1)
        g_emb2 = self.pool(node_embs2, b2)

        if self.use_tensor:
            sim_feats = self.tensor(g_emb1, g_emb2)
        else:
            sim_feats = self._compute_similarity_hist_stats(node_embs1, node_embs2, b1, b2)

        x = torch.cat([g_emb1, g_emb2, sim_feats], dim=1)

        out = self.mlp(x).squeeze(-1)
        return out
