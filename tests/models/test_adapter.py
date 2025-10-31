import torch
from torch_geometric.data import Data
#from torch_geometric.datasets import TUDataset
from src.models.adapter import GeneratorAdapter


def test_adapter_cont_and_discrete_nodes():
    """Mixed continuous + one-hot node blocks must be detected correctly"""
    N = 6

    X = torch.cat([
        torch.randn(N, 2),        # continuous (2)
        torch.eye(N)[:, :3],      # one-hot block (3)
        torch.randn(N, 1),        # continuous (1)
        torch.eye(N)[:, :4],      # one-hot block (4)
    ], dim=1)

    data = Data(x=X)

    adapter = GeneratorAdapter([data])

    # ✅ Should detect two blocks, sizes 3 and 4
    assert adapter.dis_node_blocks == [3, 4]
    # ✅ Continuous = 2 + 1 = 3
    assert adapter.num_cont_node_feats == 3
    # ✅ Max nodes
    assert adapter.max_nodes == N


def test_adapter_with_hints_splits_binary_run():
    # Make 4 nodes, 5 binary columns that are 'fake one-hot-ish':
    X = torch.tensor([
        [1,0, 0,1,0],
        [0,1, 1,0,0],
        [1,0, 0,1,0],
        [0,1, 0,0,1],
    ], dtype=torch.float)

    g = Data(x=X)  # no edge_attr needed here

    # Tell the adapter to split the single 5-col run into [2,3]
    adapter = GeneratorAdapter(
        [g],
        node_block_hints=[[2,3]],
    )

    # We expect:
    # - no continuous node features
    # - two discrete node blocks: 2 and 3
    assert adapter.num_cont_node_feats == 0
    assert adapter.dis_node_blocks == [2, 3]

    # Edge-related specs should still be well-formed defaults
    assert adapter.num_cont_edge_feats in (0, )
    assert adapter.dis_edge_blocks in ([], )


def test_adapter_dataset_scale():
    """Ensure max_nodes is selected over multiple graphs"""
    g1 = Data(x=torch.randn(4, 3))
    g2 = Data(x=torch.randn(10, 3))
    g3 = Data(x=torch.randn(6, 3))

    adapter = GeneratorAdapter([g1, g2, g3])
    assert adapter.max_nodes == 10


def test_adapter_mutag():
    from src.datasets.mutag import load_mutag
    dataset = load_mutag()
    adapter = GeneratorAdapter(dataset)

    assert adapter.max_nodes > 0
    assert adapter.num_cont_node_feats == 0
    assert adapter.num_cont_edge_feats == 0
    assert adapter.dis_node_blocks == [7]
    assert adapter.dis_edge_blocks == [4]
    assert isinstance(adapter.dis_node_blocks, list)


def test_adapter_protein():
    from src.datasets.proteins import load_proteins
    dataset = load_proteins()
    adapter = GeneratorAdapter(dataset)

    assert adapter.max_nodes > 0 
    assert adapter.num_cont_node_feats == 0
    assert adapter.num_cont_edge_feats == 0
    assert adapter.dis_node_blocks == [3]
    assert adapter.dis_edge_blocks == []