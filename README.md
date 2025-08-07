# favar_gat
Factor-Augmented Vector Autoregression with Graph Attention Network (FAVAR-GAT) model, designed for drug–target interaction modeling or protein-ligand binding prediction. 

🧠 Overview
FAVAR-GAT is a novel deep learning framework that integrates:
Factor-Augmented Vector Autoregression (FAVAR) to model latent temporal dynamics of protein-ligand interactions.
Graph Attention Networks (GAT) to capture spatial/structural dependencies between proteins and ligands.
Designed for predicting drug–target binding affinity over time using biomedical datasets like Davis, BindingDB, PDBbind and KIBA.

📌 Key Features
Combines temporal modeling (FAVAR) with molecular graph attention (GAT).
Processes large-scale biological data (EHR, SMILES, PDB structures).
Supports training on datasets with sequence + graph inputs.
Evaluates using metrics like MSE, Rm², MSFE, CI, AUC-ROC.


