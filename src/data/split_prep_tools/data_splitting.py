import pickle
import numpy as np
import pandas as pd
from deepmol.splitters import MultiTaskStratifiedSplitter
from pathlib import Path
from deepmol.compound_featurization import MorganFingerprint
from deepmol.datasets import SmilesDataset
from src.data.mgf_tools.mgf_get import mgf_get_spectra, mgf_get_smiles
from src.utils import generate_data_stats, calculate_max_num_peaks, mgf_deconvoluter, calculate_mz_vocabs
from src.config import mgf_path, min_num_peaks, noise_rmv_threshold, mass_error
REPO_ROOT = Path(__file__).resolve().parents[2]


def make_split(dataset, seed, output_dir,
               frac_train: float = 0.8,
               frac_valid: float = 0.1,
               frac_test: float = 0.1):

    """
    Generates stratified splits

    Parameters:
        dataset
            Dataset to be split
        seed : int
            Seed for reproducibility
        output_dir : Path or str
            Base directory for storing results
        frac_train : float
            Fraction of data to use for training
        frac_valid : float
            Fraction of data to use for validation
        frac_test : float
            Fraction of data to use for testing


    Returns:
        dict
            Dictionary with split IDs by category
    """

    output_dir = Path(output_dir).resolve()
    seed_dir = output_dir / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    splitter = MultiTaskStratifiedSplitter()

    train_dataset, val_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed
    )

    splits = {"train": train_dataset.ids, "val": val_dataset.ids, "test": test_dataset.ids}

    print("Saving splits")
    print(f"Split created with seed = {seed}")
    print(f"Train: {len(train_dataset)} samples ({len(train_dataset) / len(dataset) * 100:.1f}%)")
    print(f"Validation:   {len(val_dataset)} samples ({len(val_dataset) / len(dataset) * 100:.1f}%)")
    print(f"Test:  {len(test_dataset)} samples ({len(test_dataset) / len(dataset) * 100:.1f}%)")

    split_pkl = seed_dir / 'split_ids.pkl'
    with split_pkl.open('wb') as f:
        pickle.dump(splits, f)

    print(f"Saving fingerprints cache")
    all_fingerprints = []
    all_ids = []

    all_fingerprints.append(train_dataset.X)
    all_ids.extend(train_dataset.ids)

    all_fingerprints.append(val_dataset.X)
    all_ids.extend(val_dataset.ids)

    all_fingerprints.append(test_dataset.X)
    all_ids.extend(test_dataset.ids)

    all_fp = np.vstack(all_fingerprints)

    fp_df = pd.DataFrame(all_fp, columns=[f'fp_{i}' for i in range(all_fp.shape[1])])
    fp_df['spectrum_id'] = all_ids

    fp_cache_path = seed_dir / 'fingerprints.pkl'
    fp_df.to_pickle(fp_cache_path)

    train_labels = train_dataset.y
    val_labels = val_dataset.y
    test_labels = test_dataset.y

    stats_df, table_styled = generate_data_stats(train_labels, test_labels, val_labels)
    stats_csv_path = seed_dir / 'split_statistics.csv'
    stats_html_path = seed_dir / 'split_statistics.html'

    with open(stats_html_path, 'w') as f:
        f.write(table_styled.to_html())

    stats_df.to_csv(stats_csv_path, index=False)

    print(f"Saved fingerprints cache: {fp_df.shape[0]} samples, {fp_df.shape[1] - 1} features")
    print(f"Files created:")
    print(f"  • {split_pkl}")
    print(f"  • {fp_cache_path}")
    print(f"  • {stats_csv_path}")
    print(f"  • {stats_html_path}")

    return splits


# LEGACY - This function was designed to perform cleaning in the val and test set, but it was discontinued
# due to its limitations.
def clean_splits(splits: dict, smiles_df: pd.DataFrame):

    train_smiles = set(smiles_df[smiles_df['spectrum_id'].isin(splits['train'])]['smiles'])

    val_df = smiles_df[smiles_df['spectrum_id'].isin(splits['val'])]
    test_df = smiles_df[smiles_df['spectrum_id'].isin(splits['test'])]

    val_df = val_df[~val_df['smiles'].isin(train_smiles)]
    test_df = test_df[~test_df['smiles'].isin(train_smiles)]

    val_smiles = set(val_df['smiles'])
    test_df = test_df[~test_df['smiles'].isin(val_smiles)]

    val_final_df = val_df.drop_duplicates(subset=['smiles'], keep='first')
    test_final_df = test_df.drop_duplicates(subset=['smiles'], keep='first')

    cleaned_splits = {
        'train': splits['train'],
        'val': val_final_df['spectrum_id'].tolist(),
        'test': test_final_df['spectrum_id'].tolist()
    }

    stats = {
        'original_val_count': len(splits['val']),
        'cleaned_val_count': len(cleaned_splits['val']),
        'removed_from_val': len(splits['val']) - len(cleaned_splits['val']),
        'original_test_count': len(splits['test']),
        'cleaned_test_count': len(cleaned_splits['test']),
        'removed_from_test': len(splits['test']) - len(cleaned_splits['test']),
    }

    return cleaned_splits, stats


def preprocess_and_split(mgf_path, seed, output_dir=REPO_ROOT / "src/data/artifacts", num_spectra=None,
                         frac_train: float = 0.8,
                         frac_valid: float = 0.1,
                         frac_test: float = 0.1):

    """
    Function that splits the data and calculates some of the essential parameters
    Both the splits and the parameters are saved (use the seed to keep track of them) for later use.

    Parameters:
        mgf_path : Path or str
            Path to the mgf file
        seed : int
            Seed for reproducibility
        output_dir : Path or str
            Directory for storing results
        num_spectra: int
            Number of spectra, if None reads all the spectra in the file
        frac_train : float
            Fraction of data to use for training
        frac_valid : float
            Fraction of data to use for validation
        frac_test : float
            Fraction of data to use for testing
    """

    output_dir = Path(output_dir)
    seed_dir = output_dir / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading spectra and SMILES from MGF')
    mgf_spectra = mgf_get_spectra(mgf_path, num_spectra=num_spectra)
    print(f'Loaded {len(mgf_spectra)} spectra from MGF')

    smiles_df = mgf_get_smiles(mgf_spectra, as_dataframe=True)
    print(f'Loaded {len(smiles_df)} spectra')

    print('\n2. Applying filtering and spectrum processing')
    max_num_peaks = calculate_max_num_peaks(mgf_spectra, percentile=95)
    mz_vocabs = calculate_mz_vocabs(mgf_spectra)

    max_seq_len = max_num_peaks + 1
    vocab_size = len(mz_vocabs)

    pipeline_config = {
        'max_num_peaks': max_num_peaks,
        'max_seq_len': max_seq_len,
        'mz_vocabs': mz_vocabs,
        'vocab_size': vocab_size
    }

    config_path = output_dir / str(seed) / 'pipeline_config.json'
    with open(config_path, 'w') as f:
        import json
        json.dump(pipeline_config, f, indent=4)

    processed_spectra = mgf_deconvoluter(
        mgf_data=mgf_spectra,
        mz_vocabs=mz_vocabs,
        min_num_peaks=min_num_peaks,
        max_num_peaks=max_num_peaks,
        noise_rmv_threshold=noise_rmv_threshold,
        mass_error=mass_error,
        log=False)

    spectrum_ids = [spectrum_id for spectrum_id, *_ in processed_spectra]

    print('\n3. Generating fingerprints for valid spectra')

    filtered_smiles = smiles_df[smiles_df['spectrum_id'].isin(spectrum_ids)]

    smiles_list = filtered_smiles['smiles'].tolist()
    ids_list = filtered_smiles['spectrum_id'].tolist()

    dataset = SmilesDataset(smiles=smiles_list, ids=ids_list)
    dataset = MorganFingerprint().featurize(dataset)
    dataset._y = dataset.X

    initial_count = len(dataset)
    dataset.remove_duplicates(inplace=True)
    count_after_cleaning = len(dataset)
    print(f"Dataset reduced from {initial_count} to {count_after_cleaning} unique samples.")

    print('\n4. Data Splitting')

    splits = make_split(dataset, seed, output_dir, frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test)

    split_pkl = output_dir / str(seed) / 'split_ids.pkl'
    fingerprints_pkl = output_dir / str(seed) / 'fingerprints.pkl'

    if split_pkl.exists() and fingerprints_pkl.exists():
        print(f'Split IDs and Fingerprints saved')
    else:
        print('Files not found')

    return splits
