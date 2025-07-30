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


class FAVAR_GAT(nn.Module):
    def __init__(self, prot_dim=128, gat_out_dim=64, lag_dim=5):
        super().__init__()
        self.prot_embed = nn.Embedding(25, prot_dim)
        self.gat = GATConv(1, gat_out_dim, heads=4)
        self.var_fc = nn.Linear(lag_dim, 64)
        self.out = nn.Sequential(
            nn.Linear(prot_dim + gat_out_dim * 4 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, prot_batch, lig_batch, favar_batch):
        lig_out = self.gat(lig_batch.x, lig_batch.edge_index)
        lig_emb = global_mean_pool(lig_out, lig_batch.batch)

        prot_embs = []
        for prot in prot_batch:
            prot_idx = torch.tensor([ord(c) % 25 for c in prot], dtype=torch.long, device=lig_batch.x.device)
            emb = self.prot_embed(prot_idx).mean(dim=0)
            prot_embs.append(emb)
        prot_embs = torch.stack(prot_embs)

        favar_embs = self.var_fc(favar_batch)
        x = torch.cat([prot_embs, lig_emb, favar_embs], dim=-1)
        return self.out(x).squeeze()
