import torch
from src.datasets.ged_dataset import GEDDataset, create_and_save_ged_dataset, collate_pairs
from src.datasets.mutag import load_mutag

def test_create_and_save_ged_dataset(tmp_path):
    dataset = load_mutag()[:3]
    out_path = tmp_path / "mutag_ged.pt"

    ged_ds = create_and_save_ged_dataset(dataset, str(out_path), n_perturbations=1)
    assert len(ged_ds) > 0
    assert out_path.exists()

    reloaded = GEDDataset.load(str(out_path))
    assert len(reloaded) == len(ged_ds)

def test_collate_pairs_shapes(tmp_path):
    dataset = load_mutag()[:2]
    out_path = tmp_path / "mutag_ged.pt"
    ged_ds = create_and_save_ged_dataset(dataset, str(out_path), n_perturbations=1)
    batch = [ged_ds[0], ged_ds[1]]
    b1, b2, labels = collate_pairs(batch)
    assert isinstance(labels, torch.Tensor)
    assert b1.x.size(0) > 0
    assert b2.edge_index.size(1) > 0
