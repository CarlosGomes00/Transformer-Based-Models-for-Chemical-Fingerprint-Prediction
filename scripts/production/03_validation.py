import argparse
import json
from pathlib import Path

from src.models.Transformer import Transformer
from src.data.data_loader import data_loader
from src.config import mgf_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(args):

    print('Starting the model evaluation...')

    try:
        artifacts_dir = Path(args.artifacts_dir) / str(args.seed)

        with open(artifacts_dir / 'pipeline_config.json', 'r') as f:
            pipeline_config = json.load(f)

        max_num_peaks = pipeline_config['max_num_peaks']
        mz_vocabs = pipeline_config['mz_vocabs']

        print('Loading the data...')
        loaders = data_loader(
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_spectra=args.num_spectra,
            mgf_path=args.mgf_path,
            max_num_peaks=max_num_peaks,
            mz_vocabs=mz_vocabs)

        print(f'Loading model from: {args.checkpoint_path}...')
        model = Transformer.load_model(checkpoint_path=args.checkpoint_path, seed=args.seed)

        print('Performing the evaluation on the train set!')
        results_train = model.validate(data_loader=loaders['train'], split_name='train', threshold=args.threshold,
                                       save_results=args.save_results)
        print(json.dumps(results_train, indent=4))

        print('Performing the evaluation on the validation set!')
        results_val = model.validate(data_loader=loaders['val'], split_name='val', threshold=args.threshold,
                                     save_results=args.save_results)
        print(json.dumps(results_val, indent=4))


        print('Performing the evaluation on the test set!')
        results_test = model.validate(data_loader=loaders['test'], split_name='test', threshold=args.threshold,
                                      save_results=args.save_results)
        print(json.dumps(results_test, indent=4))

    except Exception as e:
        print(f'Error found: {e}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model evaluation script")

    parser.add_argument('--seed', type=int, required=True, help='Seed to save the training outputs '
                        '(use the same seed of the splits)')
    parser.add_argument('--mgf_path', type=str, default=Path(mgf_path), help='Path to the .mgf file')
    parser.add_argument('--num_spectra', type=int, default=None, help='Number of spectra, all by default')
    parser.add_argument('--artifacts_dir', type=str, default=REPO_ROOT / 'src/data/artifacts',
                        help='Artifacts directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model to be loaded')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to binning')
    parser.add_argument('--no_save', dest='save_results', action='store_false',
                        help='Deactivates the saving of the results')

    args = parser.parse_args()
    main(args)
