import os
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from utils.graph_utils import mol_to_graph, seq_to_graph  # custom utilities

DATASET = 'pdbbind'  # change this to your dataset name
RAW_PATH = f'data/raw/{DATASET}.csv'
SAVE_DIR = f'data/processed/{DATASET}/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(RAW_PATH)

# Normalize affinity values
scaler = StandardScaler()
df['affinity'] = scaler.fit_transform(df[['Log Binding Affinity']])

data_list = []

for idx, row in df.iterrows():
    smiles = row['SMILES']
    sequence = row['Protein Sequence']
    affinity = row['affinity']
    time_index = row['Time'] if 'Time' in row else 0  # optional

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Convert to graph
        ligand_data = mol_to_graph(mol)
        protein_data = seq_to_graph(sequence)

        # Merge graphs or keep separate
        data = Data(
            x_ligand=ligand_data.x,
            edge_index_ligand=ligand_data.edge_index,
            x_protein=protein_data.x,
            edge_index_protein=protein_data.edge_index,
            y=torch.tensor([affinity], dtype=torch.float),
            time=torch.tensor([time_index], dtype=torch.long)
        )

        data_list.append(data)

    except Exception as e:
        print(f"Skipping index {idx} due to error: {e}")

# Save processed data
torch.save(data_list, os.path.join(SAVE_DIR, 'processed_data.pt'))
print(f"Saved {len(data_list)} samples to {SAVE_DIR}")
