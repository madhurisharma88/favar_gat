import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models.favar_gat import FAVARGAT
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import argparse
from utils.train_utils import set_seed, split_dataset

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    data = torch.load(f'data/processed/{args.dataset}/processed_data.pt')
    train_set, val_set = split_dataset(data, train_ratio=0.8)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print("Building model...")
    model = FAVARGAT(
        gat_hidden_dim=args.gat_hidden_dim,
        var_order=args.var_order,
        num_factors=args.num_factors
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_rmse = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        mse, rm2, ci = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val MSE: {mse:.4f} | Rm2: {rm2:.4f} | CI: {CI:.4f}")
        
       torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pth'))
      print("✅ Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pdbbind')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gat-hidden-dim', type=int, default=128)
    parser.add_argument('--num-factors', type=int, default=8)
    parser.add_argument('--var-order', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--save-dir', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(args)
