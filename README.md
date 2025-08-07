# 📊 FAVAR-GAT: A Hybrid Temporal–Structural Model for Drug–Target Binding Affinity Prediction

**FAVAR-GAT** (Factor-Augmented Vector Autoregression with Graph Attention Networks) is a novel deep learning framework that models the temporal evolution and structural dependencies in drug–target interaction data. This hybrid model integrates time-series factor dynamics (FAVAR) with relational graph learning (GAT) for more accurate binding affinity prediction.

---

## 🧠 Key Features

- 🔁 **Temporal modeling** using Factor-Augmented Vector Autoregression (FAVAR)
- 🧬 **Graph-based learning** with Graph Attention Networks (GAT)
- 🧪 Supports benchmark datasets: **PDBbind**, **Davis**, **KIBA**, **BindingDB**
- ⚙️ Modular design: easy to extend to other datasets and models
- 🔐 Optionally supports privacy-preserving FL integration (e.g., CrypTen-FL)

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

## 🚀 Installation

```bash
git clone https://github.com/yourusername/FAVAR-GAT.git
cd FAVAR-GAT
pip install -r requirements.txt
