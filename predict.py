import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import Classifier, GNN


ROOT = Path(__file__).resolve().parent


def load_inputs(data_dir, device):
    edges = np.load(data_dir / 'STRING_PPI.npy', allow_pickle=False)
    genes = np.unique(edges[:, :2])
    features = np.load(
        data_dir / 'Feat.npy',
        allow_pickle=False,
    )
    if features.ndim != 2 or features.shape[0] != len(genes):
        raise ValueError('Feature rows must match the sorted STRING gene list')
    if not np.isfinite(features).all():
        raise ValueError('Features contain non-finite values')
    edge_index = torch.from_numpy(np.searchsorted(genes, edges[:, :2]).T.copy()).long()
    tensors = (
        torch.as_tensor(features, dtype=torch.float32),
        torch.zeros(len(genes), dtype=torch.long),
        torch.full((edge_index.shape[1],), 43, dtype=torch.long),
        edge_index,
        torch.ones(edge_index.shape[1], dtype=torch.long),
    )
    return genes, tuple(value.to(device) for value in tensors)


def load_model(model_path, device):
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=True)
    config = checkpoint['gnn_config']
    model = torch.nn.Sequential(
        GNN(**config),
        Classifier(config['n_hid'], checkpoint['n_out']),
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval()


def predict(model, tensors, genes):
    with torch.inference_mode():
        representations = model[0](*tensors)
        probabilities = torch.softmax(model[1](representations), dim=-1).cpu().numpy()
    if probabilities.shape != (len(genes), 2):
        raise ValueError('Expected two class probabilities for every gene')
    if not np.isfinite(probabilities).all():
        raise ValueError('Predictions contain non-finite values')
    return pd.DataFrame({
        'gene': genes,
        'ngtv_prob': probabilities[:, 0],
        'pstv_prob': probabilities[:, 1],
    }).sort_values('pstv_prob', ascending=False, kind='stable').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', choices=['PD', 'MDD', 'both'], default='both')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.top_k < 1 or args.threads < 1:
        parser.error('--top-k and --threads must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    genes, tensors = load_inputs(args.data_dir, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diseases = ['PD', 'MDD'] if args.disease == 'both' else [args.disease]
    rankings = {}
    for disease in diseases:
        model = load_model(args.model_dir / (disease + '_state_dict.pt'), device)
        rankings[disease] = predict(model, tensors, genes)
        output = args.output_dir / (disease + '_predictions.csv')
        rankings[disease].to_csv(output, index=False)
        rankings[disease].head(args.top_k).to_csv(
            args.output_dir / (disease + '_top_' + str(args.top_k) + '.csv'), index=False,
        )
        print('{}: {} genes -> {}'.format(disease, len(genes), output), flush=True)
        del model
    if len(rankings) == 2:
        common = np.intersect1d(
            rankings['PD'].head(args.top_k)['gene'],
            rankings['MDD'].head(args.top_k)['gene'],
        )
        pd.DataFrame({'gene': common}).to_csv(
            args.output_dir / ('PD_MDD_common_top_' + str(args.top_k) + '.csv'), index=False,
        )
        print('Shared top {} genes: {}'.format(args.top_k, len(common)), flush=True)


if __name__ == '__main__':
    main()
