import argparse
import json
from pathlib import Path

from src.config import mgf_path
from src.models.Transformer import Transformer
from src.data.data_loader import data_loader

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(args):

    print(f'Start the training pipeline with seed: {args.seed}')

    try:
        artifacts_dir = Path(args.artifacts_dir) / str(args.seed)

        with open(artifacts_dir / 'pipeline_config.json', 'r') as f:
            pipeline_config = json.load(f)

        max_num_peaks = pipeline_config['max_num_peaks']
        max_seq_len = pipeline_config['max_seq_len']
        mz_vocabs = pipeline_config['mz_vocabs']
        vocab_size = pipeline_config['vocab_size']

        loaders = data_loader(
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_spectra=args.num_spectra,
            mgf_path=args.mgf_path,
            max_num_peaks=max_num_peaks,
            mz_vocabs=mz_vocabs)

        model = Transformer(seed=args.seed,
                            max_seq_len=max_seq_len,
                            vocab_size=vocab_size,
                            morgan_default_dim=args.fingerprint_dim,
                            d_model=args.d_model,
                            n_head=args.n_head,
                            num_layers=args.num_layers,
                            dropout_rate=args.dropout_rate,
                            loss_func=args.loss,
                            pos_weight=args.pos_weight,
                            focal_gama=args.focal_gama,
                            focal_alpha=args.focal_alpha)

        model_fitted = model.fit(train_loader=loaders['train'], val_loader=loaders['val'], max_epochs=args.max_epochs,
                                 fast_dev_run=args.fast_dev_run)

    except Exception as e:
        print(f'Error found: {e}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training script")

    parser.add_argument('--seed', type=int, required=True, help='Seed to save the training outputs')
    parser.add_argument('--mgf_path', type=str, default=Path(mgf_path), help='Path to the .mgf file')
    parser.add_argument('--num_spectra', type=int, default=None, help='Number of spectra, all by default')
    parser.add_argument('--artifacts_dir', type=str, default=REPO_ROOT / 'src/data/artifacts',
                        help='Artifacts directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs')

    parser.add_argument('--fast_dev_run', action='store_true', help='Runs the pipeline with only one '
                        'training and validation batch ')

    parser.add_argument('--fingerprint_dim', type=int, default=2048, help='Dimension of the fingerprints')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)

    parser.add_argument('--loss', type=str, default='bce_logits', help='Loss to be used (bce, bce_logits, focal')
    parser.add_argument('--pos_weight', type=float, default=1, help='Only used if loss=bce_logits - '
                        'Weight for the positive class')
    parser.add_argument('--focal_gama', type=float, default=2, help='Only used if loss=focal - '
                        'Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Only used if loss=focal - '
                        'Weighting factor in range [0, 1] to balance positive vs negative examples or -1 for ignore')

    args = parser.parse_args()
    main(args)
