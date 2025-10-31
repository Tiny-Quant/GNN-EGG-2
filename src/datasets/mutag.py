import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from typing import List

def load_mutag(root='/tmp/MUTAG') -> List[Data]: 
    """
    Load the MUTAG dataset as PyG Data objects.
    Ensures each node has a feature vector.
    """
    dataset = TUDataset(root=root, name='MUTAG')
    
    # Ensure node features exist
    for data in dataset:
        if not hasattr(data, 'x') or data.x is None:
            # Default feature: one-hot degree
            degrees = torch.tensor([data.edge_index[0].tolist().count(i) for i in range(data.num_nodes)])
            data.x = torch.nn.functional.one_hot(degrees, num_classes=int(degrees.max())+1).float()
    return dataset
