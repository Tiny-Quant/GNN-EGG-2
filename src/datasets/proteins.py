import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from typing import List

def load_proteins(root='/tmp/PROTEINS') -> List[Data]:
    """
    Load the PROTEINS dataset as PyG Data objects.
    Ensures each node has a feature vector.
    """
    dataset = TUDataset(root=root, name='PROTEINS')
    
    # Ensure node features exist
    for data in dataset:
        if not hasattr(data, 'x') or data.x is None:
            # Use one-hot labels if available
            if hasattr(data, 'node_labels') and data.node_labels is not None:
                data.x = torch.nn.functional.one_hot(data.node_labels, 
                                                     num_classes=int(data.node_labels.max())+1).float()
            else:
                data.x = torch.ones((data.num_nodes, 1))
    return dataset
