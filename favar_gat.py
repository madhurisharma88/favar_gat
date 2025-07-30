import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, top_k_accuracy_score

# ---- Data Utils ----
def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() == 0:
            return None
        atom_feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        edges = rdmolops.GetAdjacencyMatrix(mol)
        if len(atom_feats) == 0 or edges.sum() == 0:
            return None
        edge_index = torch.tensor(np.array(np.nonzero(edges)), dtype=torch.long)
        x = torch.tensor(atom_feats, dtype=torch.float).unsqueeze(1)
        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        print(f"[SMILES ERROR] {smiles[:20]}...: {str(e)}")
        return None
