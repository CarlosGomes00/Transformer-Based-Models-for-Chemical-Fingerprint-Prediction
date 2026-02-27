import argparse
import numpy as np
from pathlib import Path

from src.data.split_prep_tools.data_splitting import preprocess_and_split
from src.config import mgf_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(args):

    try:
        preprocess_and_split(
            mgf_path=Path(args.mgf_path),
            seed=args.seed,
            output_dir=Path(args.output_dir),
            num_spectra=args.num_spectra,
            frac_valid=args.frac_valid,
            frac_test=args.frac_test,
            remove_train_duplicates=args.remove_duplicates,
            balance_dataset=args.balance_dataset,
            spectra_by_compound=args.spectra_by_compound
        )
    except Exception as e:
        print(f'Error found: {e}')
        raise


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data splitting and preprocessing wrapper')

    parser.add_argument('--mgf_path', type=str, default=Path(mgf_path), help='Path to the .mgf file')
    parser.add_argument('--seed', type=int, required=True, help='Seed to save the splits')
    parser.add_argument('--output_dir', type=str, default=Path(REPO_ROOT / "src/data/artifacts"),
                        help='Path to the output directory')
    parser.add_argument('--num_spectra', type=int, default=None, help='Number of spectra, all by default')
    parser.add_argument('--frac_valid', type=float, default=0.1, help='Fraction of validation data')
    parser.add_argument('--frac_test', type=float, default=0.1, help='Fraction of test data')
    parser.add_argument('--remove_duplicates', action='store_true', help= 'Dont enable data augmentation')
    parser.add_argument('--balance_dataset', action='store_true', help= 'Limits the number of spectra per compound')
    parser.add_argument('--spectra_by_compound', type=int, default=4, help='Maximum number of spectra by compound allowed. balance_dataset must be true to use this argument')

    args = parser.parse_args()
    main(args)
