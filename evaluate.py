import torch
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from torch_geometric.loader import DataLoader
from models.favar_gat import FAVARGAT
import argparse

def rm2_score(y_true, y_pred):
    """RM² metric combining R² and Pearson R"""
    r, _ = pearsonr(y_true, y_pred)
    r2 = r**2
    y_mean = np.mean(y_true)
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    rm2 = r2 * (1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_mean) ** 2)))
    return max(0.0, rm2)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y_true.extend(batch.y.view(-1).cpu().numpy())
            y_pred.extend(out.view(-1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rm2 = rm2_score(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)

    # Binary AUC ROC (optional: define threshold or binary labels)
    try:
        # Assuming binary labels can be created (e.g., thresholding at mean or 7.0)
        binary_labels = (y_true >= np.median(y_true)).astype(int)
        auc_roc = roc_auc_score(binary_labels, y_pred)
    except Exception:
        auc_roc = float('nan')

    print(f"MSE: {mse:.4f}")
    print(f"RM²: {rm2:.4f}")
    print(f"CI: {ci:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pdbbind')
    parser.add_argument('--model-path', type=str, default='checkpoints/model_best.pth')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(f'data/processed/{args.dataset}/processed_data.pt')
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    model = FAVARGAT(
        gat_hidden_dim=128,
        var_order=2,
        num_factors=8
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    evaluate(model, loader, device)
