import torch
import argparse
from torch_geometric.loader import DataLoader
from models.favar_gat import FAVARGAT
from models.ablation_models import GATOnly, FAVAR_GCN, MLPBaseline
from evaluate import evaluate

def load_data(dataset, batch_size):
    data = torch.load(f'data/processed/{dataset}/processed_data.pt')
    return DataLoader(data, batch_size=batch_size, shuffle=False)

def run_ablation(name, model, data_loader, device, model_path=None):
    print(f"\n🔍 Running ablation: {name}")
    model = model.to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    evaluate(model, data_loader, device)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = load_data(args.dataset, args.batch_size)

    ablation_configs = {
        'favar_gat': FAVARGAT(gat_hidden_dim=128, var_order=2, num_factors=8),
        'gat_only': GATOnly(hidden_dim=128),
        'favar_gcn': FAVAR_GCN(gcn_hidden_dim=128, var_order=2, num_factors=8),
        'mlp': MLPBaseline(input_dim=256)
    }

    for name, model in ablation_configs.items():
        model_path = f'checkpoints/{name}.pth' if args.use_checkpoints else None
        run_ablation(name, model, loader, device, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pdbbind')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-checkpoints', action='store_true')
    args = parser.parse_args()
    main(args)
