import torch
from src.models.generator import GraphGenerator


def test_cont_node_only_shapes_and_grads():
    """
    Step 1 sanity test:
      - can generate adjacency + continuous node features
      - shapes are correct (no batch dim when B==1)
      - values of adj in (0,1)
      - gradients flow
    """
    N, F = 7, 1
    gen = GraphGenerator(
        max_nodes=N,
        num_cont_node_feats=F,
        allow_self_loops=False,   # exercise mask
        symmetric_adj=True,       # exercise symmetry
        batch_size=1
    )

    out = gen()
    adj = out["adj"]
    X = out["cont_node"]

    # shapes
    assert adj.shape == (1, N, N)
    assert X.shape == (1, N, F)

    # range checks for adjacency
    assert torch.all(adj >= 0) and torch.all(adj <= 1)

    # self-loops masked for every graph in batch
    B, N, _ = adj.shape
    diag = adj[:, torch.arange(N), torch.arange(N)]  # (B, N)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    # symmetry respected
    assert torch.allclose(adj, adj.transpose(1, 2), atol=1e-6)

    # differentiability
    loss = adj.mean() + X.mean()
    loss.backward()

    # some gradients should exist
    grads = [gen.adj_logits.grad, gen.cont_node.grad]
    assert all(g is not None for g in grads)

def test_cont_node_only_shapes_batch_mode():
    """
    Step 1: Allow batching multiple independent graphs (B > 1):
      - adjacency shape [B, N, N]
      - cont_node shape [B, N, F]
      - gradients flow for all items
    """
    B, N, F = 5, 7, 1
    gen = GraphGenerator(
        max_nodes=N,
        num_cont_node_feats=F,
        allow_self_loops=True,   # allow diagonal now
        symmetric_adj=False,     # allow directed adj now
        batch_size=B
    )

    out = gen()
    adj = out["adj"]
    X = out["cont_node"]

    assert adj.shape == (B, N, N)
    assert X.shape == (B, N, F)

    # adjacency values in [0, 1]
    assert torch.all(adj >= 0) and torch.all(adj <= 1)

    # differentiability
    loss = adj.mean() + X.mean()
    loss.backward()

    assert gen.adj_logits.grad is not None
    assert gen.cont_node.grad is not None


def test_discrete_node_shapes_and_probabilities():
    N, C = 6, 4
    gen = GraphGenerator(
        max_nodes=N,
        dis_node_blocks=[C], 
        allow_self_loops=False,
        symmetric_adj=True,
        batch_size=1,
    )

    out = gen()
    dis = out["dis_node"]

    # shape check
    assert dis.shape == (1, N, C)

    # rows should softmax to 1
    assert torch.allclose(dis.sum(dim=(0, 2)), torch.ones(N), atol=1e-4)

    # differentiability check
    loss = dis.mean()
    loss.backward()
    assert any(p.grad is not None for p in gen.dis_node_logits)



def test_discrete_node_batch_mode():
    B, N, C = 5, 6, 3
    gen = GraphGenerator(
        max_nodes=N,
        #num_dis_node_classes=C,
        dis_node_blocks=[C], 
        batch_size=B
    )
    out = gen()
    dis = out["dis_node"]

    assert dis.shape == (B, N, C)
    assert torch.allclose(dis.sum(dim=-1), torch.ones(B, N), atol=1e-4)


def test_mixed_node_features():
    """
    Ensure generator can produce both continuous + discrete node features together.
    """
    N = 7
    C_disc = [5]  # 5 node classes
    C_cont = 3  # 3 real-valued features

    gen = GraphGenerator(
        max_nodes=N,
        num_cont_node_feats=C_cont,
        #num_dis_node_classes=C_disc,
        dis_node_blocks=C_disc, 
        batch_size=1,
        allow_self_loops=True,
    )

    out = gen()

    cont = out["cont_node"]    # [B, N, 3]
    dis = out["dis_node"]      # [B, N, 5]

    assert cont.shape == (1, N, C_cont)
    assert dis.shape == (1, N, sum(C_disc))

    # Check continuous feature differentiability
    loss = cont.mean() + dis.mean()
    loss.backward()

    # Ensure gradients exist for both param sets
    assert gen.cont_node.grad is not None
    assert any(p.grad is not None for p in gen.dis_node_logits)

    # Discrete rows should sum to 1
    assert torch.allclose(dis.sum(dim=(0, 2)), torch.ones(N), atol=1e-4)


def test_multi_block_discrete_nodes():
    N = 6
    blocks = [3, 4]

    gen = GraphGenerator(max_nodes=N, dis_node_blocks=blocks)
    out = gen()
    dis = out["dis_node"]  # (1, N, sum(blocks)) or (B, N, C)

    assert dis.shape == (1, N, sum(blocks))  # ✅ Updated expectation

    # Probabilities should sum to 1 per block
    cumsum = 0
    for Ck in blocks:
        block = dis[0, :, cumsum:cumsum + Ck]
        assert torch.allclose(block.sum(dim=1), torch.ones(N), atol=1e-4)
        cumsum += Ck

    # Backprop sanity
    (dis.mean()).backward()
    assert any(p.grad is not None for p in gen.dis_node_logits)


def test_discrete_edge_shapes_and_probabilities():
    """
    Ensure dis_edge:
    - has correct shape
    - rows softmax to 1 over class dimension
    - grads flow
    """
    B, N, D = 1, 5, 3
    gen = GraphGenerator(
        max_nodes=N,
        dis_edge_blocks=[D],
        allow_self_loops=False,
        symmetric_adj=False,
        batch_size=B,
    )

    out = gen()
    dis = out["dis_edge"]  # (B, N, N, D)

    assert dis.shape == (B, N, N, D)
    assert torch.all(dis >= 0) and torch.all(dis <= 1)

    # Per-edge categorical distribution sum to 1
    assert torch.allclose(dis.sum(dim=-1), torch.ones(B, N, N), atol=1e-4)

    # Grad check
    (dis.mean()).backward()
    assert any(p.grad is not None for p in gen.dis_edge_logits)


def test_discrete_edge_symmetry():
    """If symmetric_adj=True, dis_edge must be symmetric across nodes"""
    N, D = 6, 4
    gen = GraphGenerator(
        max_nodes=N,
        dis_edge_blocks=[D],
        symmetric_adj=True,
        batch_size=1,
    )
    out = gen()
    dis = out["dis_edge"]  # (1, N, N, D)

    # Symmetric across edge index dims (1<->2)
    assert torch.allclose(dis, dis.transpose(1, 2), atol=1e-6)


def test_continuous_edge_shapes_and_grads():
    """Ensure cont_edge outputs valid values and gradients flow"""
    B, N, F = 3, 7, 2
    gen = GraphGenerator(
        max_nodes=N,
        num_cont_edge_feats=F,
        symmetric_adj=False,
        batch_size=B,
    )
    out = gen()
    E = out["cont_edge"]  # (B, N, N, F)

    assert E.shape == (B, N, N, F)
    assert torch.is_tensor(E)

    (E.mean()).backward()
    assert gen.cont_edge.grad is not None


def test_continuous_edge_symmetry():
    """Validate symmetry enforcement for continuous edge features"""
    N, F = 6, 2
    gen = GraphGenerator(
        max_nodes=N,
        num_cont_edge_feats=F,
        symmetric_adj=True,
        batch_size=1,
    )
    out = gen()
    E = out["cont_edge"]

    assert torch.allclose(E, E.transpose(1, 2), atol=1e-6)


def test_mixed_edge_features():
    """
    Both continuous and discrete edge features enabled together.
    Ensure shapes, softmax, and gradients work.
    """
    N, F, D = 5, 2, 3
    gen = GraphGenerator(
        max_nodes=N,
        num_cont_edge_feats=F,
        dis_edge_blocks=[D],
        symmetric_adj=True,
        allow_self_loops=True,
        batch_size=1,
    )
    out = gen()
    cont = out["cont_edge"]
    dis = out["dis_edge"]

    assert cont.shape == (1, N, N, F)
    assert dis.shape == (1, N, N, D)

    # Discrete edge class probs
    assert torch.allclose(dis.sum(dim=-1), torch.ones(1, N, N), atol=1e-4)

    # Backprop both loss components
    (cont.mean() + dis.mean()).backward()
    assert gen.cont_edge.grad is not None
    assert any(p.grad is not None for p in gen.dis_edge_logits)


def test_multi_block_discrete_edges():
    """Multiple discrete edge one-hot blocks concatenated correctly"""
    N = 4
    blocks = [2, 3]  # => 5 total discrete edge channels

    gen = GraphGenerator(
        max_nodes=N,
        dis_edge_blocks=blocks,
        symmetric_adj=True,
        batch_size=1,
    )
    out = gen()
    dis = out["dis_edge"]

    assert dis.shape == (1, N, N, sum(blocks))

    # Validate softmax per block
    cumsum = 0
    for Dk in blocks:
        blk = dis[..., cumsum:cumsum + Dk]
        assert torch.allclose(blk.sum(dim=-1), torch.ones(1, N, N), atol=1e-4)
        cumsum += Dk

    (dis.mean()).backward()
    assert any(p.grad is not None for p in gen.dis_edge_logits)


def test_full_mixed_feature_generation():
    """
    Confirm generator supports:
      - continuous node feats
      - continuous edge feats
      - multiple discrete node blocks
      - multiple discrete edge blocks
    """
    B, N = 2, 8
    dis_node_blocks = [3, 5]
    dis_edge_blocks = [2, 4]

    gen = GraphGenerator(
        max_nodes=N,
        batch_size=B,
        num_cont_node_feats=2,
        num_cont_edge_feats=1,
        dis_node_blocks=dis_node_blocks,
        dis_edge_blocks=dis_edge_blocks,
        allow_self_loops=False,
        symmetric_adj=True,
    )

    out = gen()
    adj = out["adj"]
    Xc = out["cont_node"]
    Xe = out["cont_edge"]
    Xdn = out["dis_node"]
    Xde = out["dis_edge"]

    assert adj.shape == (B, N, N)
    assert Xc.shape == (B, N, 2)
    assert Xe.shape == (B, N, N, 1)
    assert Xdn.shape == (B, N, sum(dis_node_blocks))
    assert Xde.shape == (B, N, N, sum(dis_edge_blocks))

    # check symmetry
    assert torch.allclose(adj, adj.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(Xde, Xde.transpose(1, 2), atol=1e-6)

    # discrete blocks each softmax:
    csum = 0
    for Ck in dis_node_blocks:
        blk = Xdn[..., csum:csum + Ck]
        assert torch.allclose(blk.sum(dim=-1), torch.ones(B, N), atol=1e-4)
        csum += Ck

    csum = 0
    for Ck in dis_edge_blocks:
        blk = Xde[..., csum:csum + Ck]
        assert torch.allclose(blk.sum(dim=-1), torch.ones(B, N, N), atol=1e-4)
        csum += Ck

    # gradients
    (adj.mean() + Xc.mean() + Xdn.mean() + Xe.mean() + Xde.mean()).backward()
    assert any(p.grad is not None for p in gen.parameters())