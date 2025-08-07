# 📊 FAVAR-GAT: A Hybrid Temporal–Structural Model for Drug–Target Binding Affinity Prediction

**FAVAR-GAT** (Factor-Augmented Vector Autoregression with Graph Attention Networks) is a novel deep learning framework that models the temporal evolution and structural dependencies in drug–target interaction data. This hybrid model integrates time-series factor dynamics (FAVAR) with relational graph learning (GAT) for more accurate binding affinity prediction.

---

## 🧠 Key Features

- 🔁 **Temporal modeling** using Factor-Augmented Vector Autoregression (FAVAR)
- 🧬 **Graph-based learning** with Graph Attention Networks (GAT)
- 🧪 Supports benchmark datasets: **PDBbind**, **Davis**, **KIBA**, **BindingDB**
- ⚙️ Modular design: easy to extend to other datasets and models

---

## 📂 Datasets

This project supports multiple benchmark DTI datasets:

### 🔗 PDBbind
- Structural protein–ligand complexes with binding affinities.
- Used: **PDBbind v2013 Core & Refined Sets**
- [https://www.pdbbind.org.cn/](https://www.pdbbind.org.cn/)

### 🔗 Davis
- 68 drugs × 442 kinases.
- Label: `−log(Kd)` in molar units.
- From: DeepDTA / GraphDTA repositories

### 🔗 KIBA
- 2,111 drugs × 229 kinases.
- Label: KIBA score (integrated bioactivity).
- Sparse matrix with ~118k interactions.

### 🔗 BindingDB
- Public experimental dataset with ~2M drug–target pairs.
- Includes Ki/Kd/IC50 values and full sequences.
- [https://www.bindingdb.org/](https://www.bindingdb.org/)

Each dataset is preprocessed into:
- SMILES strings (`smiles.txt`)
- Protein sequences (`seqs.txt`)
- Binding affinities matrix (`affinity.csv`)
- Graph structures (generated during runtime)

---

## 🧱 Model Architecture

### 1. **FAVAR Module**
- Captures **temporal latent factors** from interaction sequences.
- Implements VAR with dimensionality reduction via factor loading.

### 2. **Graph Attention Network**
- Drug and protein graphs constructed from SMILES and sequences.
- Applies multi-head attention for feature aggregation.

### 3. **Fusion & Prediction**
- FAVAR time embeddings and GAT node embeddings are fused.
- Final output layer predicts binding affinity (regression).

---

## **Evaluation Metrics**
- **MSE: Mean Squared Error**
- **R²: Coefficient of Determination**
- **CI: Concordance Index**
- **AUC-ROC: Area Under the Receiver Operating Characteristic**

---
## 🚀 **Installation**
```bash
git clone https://github.com/madhurisharma88/favar_gat.git
```

## Usage
➤ Train the model
```bash
python train.py --dataset {dataset_name} --epochs 100 --batch_size 64
```
➤ Evaluate
```bash
python evaluate.py --dataset {dataset_name} --model_path saved_models/{saveddataset_name}.pt
```
➤ Available arguments
- --dataset       &nbsp;&nbsp;&nbsp;&nbsp; Dataset name: pdbbind | davis | kiba | bindingdb
- --epochs        &nbsp;&nbsp;&nbsp;&nbsp; Number of training epochs
- --hidden_dim    &nbsp;&nbsp;&nbsp;&nbsp; Hidden dimensions
- --fusion        &nbsp;&nbsp;&nbsp;&nbsp; Fusion type: concat | attention | mlp
- --use_favar     &nbsp;&nbsp;&nbsp;&nbsp; Enable FAVAR temporal modeling

