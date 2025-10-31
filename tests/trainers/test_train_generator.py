import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.generator import GraphGenerator
from src.models.losses import SoftContrastiveEmbedLoss, EdgePenalty
from src.trainers.train_generator import GeneratorTrainer, GenTrainConfig, precompute_class_means
from src.utils.embeddings import compute_classwise_means


# ---------------------------------------------------------------------
#  Dummy Explainee GNN for testing
# ---------------------------------------------------------------------
class DummyExplainee(nn.Module):
    """Simple GCN-based classifier for tests."""
    def __init__(self, in_dim=8, hidden_dim=16, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch)
        return self.lin(out)


# ---------------------------------------------------------------------
#  Helper to make a small dataset
# ---------------------------------------------------------------------
def make_toy_dataset(num_graphs=6, num_nodes=5, in_dim=8):
    data_list = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes, in_dim)
        # fully-connected undirected graph
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data_list.append(Data(x=x, edge_index=edge_index))
    return data_list


# ---------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------
def test_compute_classwise_means_shapes():
    """Verify that mean embeddings are computed for each class and layer."""
    dataset = make_toy_dataset(num_graphs=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=Batch.from_data_list)

    model = DummyExplainee()
    device = torch.device("cpu")
    num_classes = 2

    means = compute_classwise_means(model, loader, device, num_classes)
    # expected keys include both conv layers and linear head
    assert any("conv1" in k for k in means)
    assert any("conv2" in k for k in means)
    for name, M in means.items():
        assert M.shape[0] == num_classes
        assert M.ndim == 2
        assert torch.isfinite(M).all()


def test_soft_contrastive_embed_loss_forward_backward():
    """Ensure SoftContrastiveEmbedLoss runs without error and gives finite grad."""
    model = DummyExplainee()
    device = torch.device("cpu")

    # fake means (2 classes, 16-dim) for both conv layers
    means = {
        "conv1": torch.randn(2, 16),
        "conv2": torch.randn(2, 16),
    }

    gen_out = {
        "adj": torch.sigmoid(torch.randn(2, 5, 5)),
        "cont_node": torch.randn(2, 5, 8),
    }

    loss_fn = SoftContrastiveEmbedLoss(
        explainee=model,
        classwise_means=means,
        target_class=0,
        thresh=0.5,
        layer_names=["conv1", "conv2"],
    )
    loss = loss_fn(gen_out)
    assert torch.isfinite(loss)
    loss.backward()


def test_generator_trainer_step_computes_gradients():
    """End-to-end: generator + trainer compute loss and grads."""
    model = DummyExplainee()
    dataset = make_toy_dataset(num_graphs=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=Batch.from_data_list)
    means = compute_classwise_means(model, loader, torch.device("cpu"), num_classes=2)

    trainer = GeneratorTrainer(
        explainee=model,
        num_classes=2,
        classwise_means=means,
        device=torch.device("cpu"),
        lambda_edge=0.1,
    )

    gen = GraphGenerator(max_nodes=5, num_cont_node_feats=8, batch_size=1)
    loss = trainer.step(gen, target_class=0)
    assert torch.isfinite(loss)
    # verify that generator parameters got grads
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert len(grads) > 0, "Expected gradients on generator parameters"


def test_edge_penalty_behavior():
    """EdgePenalty gives larger loss for denser adjacency."""
    pen = EdgePenalty(edge_budget=10.0)
    sparse = torch.zeros(1, 5, 5)
    dense = torch.ones(1, 5, 5)
    assert pen(dense) > pen(sparse)
