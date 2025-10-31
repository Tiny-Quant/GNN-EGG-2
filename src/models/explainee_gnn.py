# src/models/explainee_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class ExplaineeGIN(nn.Module):
    """
    Graph-level classifier using a simple GIN encoder.
    Designed to be used as the explainee model for explanation tasks.
    """

    def __init__(self, in_dim, hidden_dim=32, num_layers=2, dropout=0.2, num_classes=2):
        super().__init__()
        self.dropout = dropout

        def make_mlp(in_f, out_f):
            return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU(), nn.Linear(out_f, out_f))

        self.layers = nn.ModuleList()
        last_dim = in_dim
        for _ in range(num_layers):
            conv = GINConv(make_mlp(last_dim, hidden_dim))
            self.layers.append(conv)
            last_dim = hidden_dim

        self.readout = global_mean_pool
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g_emb = self.readout(x, batch)
        out = self.fc(g_emb)
        return out

    def __call__(self, *args, **kwargs):
        # Allow both model(batch) and model(x, edge_index, batch)
        if len(args) == 1 and hasattr(args[0], "x"):
            batch = args[0]
            return super().__call__(batch.x, batch.edge_index, batch.batch)
        return super().__call__(*args, **kwargs)