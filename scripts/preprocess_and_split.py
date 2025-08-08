import os
from data.mgf_tools.mgf_get import mgf_get_spectra, mgf_get_smiles
from src.utils import mgf_deconvoluter
from deepmol.compound_featurization import MorganFingerprint
from deepmol.datasets import SmilesDataset
from src.data.stratified_split import make_split
from src.config import *
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]


def process_and_split(mgf_path, seed, output_dir=REPO_ROOT / "src/data/artifacts", num_spectra=None):

    os.makedirs(output_dir, exist_ok=True)

    print('\n1. Loading spectra and SMILES from MGF')
    try:
        mgf_spectra = mgf_get_spectra(mgf_path, num_spectra=num_spectra)
        print(f'Loaded {len(mgf_spectra)} spectra from MGF')

        smiles_df = mgf_get_smiles(mgf_spectra, as_dataframe=True)
        print(f'Loaded {len(smiles_df)} spectra')

    except Exception as e:
        print(f'Error loading the MGF file: {e}')
        return None

    print('\n2. Applying filtering and spectrum processing')
    try:
        processed_spectra = mgf_deconvoluter(
            mgf_data=mgf_spectra,
            mz_vocabs=mz_vocabs,
            min_num_peaks=min_num_peaks,
            max_num_peaks=max_num_peaks,
            noise_rmv_threshold=noise_rmv_threshold,
            mass_error=mass_error,
            log=True
        )

        spectrum_ids = [spectrum_id for spectrum_id, *_ in processed_spectra]
        print(f'Valid spectra after filtering: {len(spectrum_ids)}')

    except Exception as e:
        print(f'Error doing the processing: {e}')
        return None

    print('\n3. Generating fingerprints for valid spectra')
    try:
        filtered_smiles = smiles_df[smiles_df['spectrum_id'].isin(spectrum_ids)]

        smiles_list = filtered_smiles['smiles'].tolist()
        ids_list = filtered_smiles['spectrum_id'].tolist()

        dataset = SmilesDataset(smiles=smiles_list, ids=ids_list)
        dataset = MorganFingerprint().featurize(dataset)
        dataset._y = dataset.X

    except Exception as e:
        print(f"Error: {e}")
        return None

    print('\n4. Data spliting')
    try:
        splits = make_split(dataset, seed, output_dir)
        print(f"Split criado com seed={seed}")
        print(f"Train: {len(splits['train'])} samples ({len(splits['train']) / len(dataset) * 100:.1f}%)")
        print(f"Validation:   {len(splits['val'])} samples ({len(splits['val']) / len(dataset) * 100:.1f}%)")
        print(f"Test:  {len(splits['test'])} samples ({len(splits['test']) / len(dataset) * 100:.1f}%)")

        split_pkl = output_dir / 'split_ids.pkl'
        fingerprints_pkl = output_dir / 'fingerprints.pkl'

        if split_pkl.exists() and fingerprints_pkl.exists():
            print(f'Split IDs and Fingerprints saved')
        else:
            print('Files not found')

        return splits

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    seed = 0
    output_dir = REPO_ROOT / "src/data/artifacts"

    results = process_and_split(
        mgf_path=mgf_path,
        seed=seed,
        output_dir=output_dir,
        num_spectra=100
    )
