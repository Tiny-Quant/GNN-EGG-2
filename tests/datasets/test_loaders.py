from src.datasets.mutag import load_mutag
from src.datasets.proteins import load_proteins

def test_load_mutag_structure():
    dataset = load_mutag()
    assert len(dataset) > 0
    first = dataset[0]
    assert hasattr(first, "x")
    assert hasattr(first, "edge_index")

def test_load_protein_structure():
    dataset = load_proteins()
    assert len(dataset) > 0
    first = dataset[0]
    assert hasattr(first, "x")
    assert hasattr(first, "edge_index")
