import argparse
import json
import optuna
from pathlib import Path

from src.config import mgf_path
from src.data.data_loader import data_loader
from src.models.components.hyperparameters_tuning import objective

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(args):

    artifacts_dir = Path(args.artifacts_dir) / str(args.seed)

    with open(artifacts_dir / 'pipeline_config.json', 'r') as f:
        pipeline_config = json.load(f)

    hyper_params = {
            'seed': args.seed,
            'max_seq_len': pipeline_config['max_seq_len'],
            'vocab_size': pipeline_config['vocab_size'],
            'morgan_default_dim': 2048
             }

    loaders = data_loader(seed=args.seed, mgf_path=args.mgf_path, batch_size=args.batch_size,
                          num_workers=args.num_workers, max_num_peaks=pipeline_config['max_num_peaks'],
                          mz_vocabs=pipeline_config['mz_vocabs'])

    func = lambda trial: objective(trial, hyper_params, loaders)
    study = optuna.create_study(direction='minimize')
    study.optimize(func, n_trials=args.trials)

    print(f'Best trial: {study.best_value}')
    print(f'Best params: {study.best_params}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')

    parser.add_argument('--seed', type=int, required=True, help='Seed')
    parser.add_argument('--mgf_path', type=str, default=Path(mgf_path), help='Path to the .mgf file')
    parser.add_argument('--artifacts_dir', type=str, default=REPO_ROOT / 'src/data/artifacts',
                        help='Artifacts directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')

    args = parser.parse_args()
    main(args)
