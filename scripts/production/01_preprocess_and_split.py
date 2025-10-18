import argparse
import numpy as np
from pathlib import Path

from src.data.split_prep_tools.data_splitting import preprocess_and_split
from src.config import mgf_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(args):

    if not np.isclose(args.frac_train + args.frac_valid + args.frac_test, 1.0):
        raise ValueError('Fractions (Train, Validation and Test) must sum to 1.0')

    try:
        preprocess_and_split(
            mgf_path=Path(args.mgf_path),
            seed=args.seed,
            output_dir=Path(args.output_dir),
            num_spectra=args.num_spectra,
            frac_train=args.frac_train,
            frac_valid=args.frac_valid,
            frac_test=args.frac_test,
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
    parser.add_argument('--frac_train', type=float, default=0.8, help='Fraction of training data')
    parser.add_argument('--frac_valid', type=float, default=0.1, help='Fraction of validation data')
    parser.add_argument('--frac_test', type=float, default=0.1, help='Fraction of test data')
    parser.add_argument('--clean', action='store_false', help='Flag to clean the splits')

    args = parser.parse_args()
    main(args)
