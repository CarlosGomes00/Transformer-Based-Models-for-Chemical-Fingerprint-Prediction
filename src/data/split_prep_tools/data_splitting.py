import pickle
import numpy as np
import pandas as pd
from deepmol.splitters import MultiTaskStratifiedSplitter
from pathlib import Path
from deepmol.compound_featurization import MorganFingerprint
from deepmol.datasets import SmilesDataset
from src.data.mgf_tools.mgf_get import mgf_get_spectra, mgf_get_smiles
from src.utils import (generate_data_stats, calculate_max_num_peaks, mgf_deconvoluter, calculate_mz_vocabs,
                       canonicalize_smiles)
from src.config import mgf_path, min_num_peaks, noise_rmv_threshold, mass_error
REPO_ROOT = Path(__file__).resolve().parents[2]


def clean_splits(splits: dict, smiles_df: pd.DataFrame, remove_train_duplicates: bool):

    train_subset = smiles_df[smiles_df['spectrum_id'].isin(splits['train'])].copy()

    if remove_train_duplicates:
        print('Removing train duplicate smiles')
        train_final_df = train_subset.drop_duplicates(subset=['canon_smiles'], keep='first')
        train_ids = train_final_df['spectrum_id'].tolist()
        train_smiles = set(train_final_df['canon_smiles'].dropna())
    else:
        train_ids = splits['train']
        train_smiles = set(train_subset['canon_smiles'].dropna())

    val_df = smiles_df[smiles_df['spectrum_id'].isin(splits['val'])].copy()
    test_df = smiles_df[smiles_df['spectrum_id'].isin(splits['test'])].copy()

    val_clean = val_df[~val_df['canon_smiles'].isin(train_smiles)]
    test_clean = test_df[~test_df['canon_smiles'].isin(train_smiles)]

    val_smiles_final = set(val_clean['canon_smiles'].dropna())
    test_clean = test_clean[~test_clean['canon_smiles'].isin(val_smiles_final)]

    val_final_df = val_clean.drop_duplicates(subset=['canon_smiles'], keep='first')
    test_final_df = test_clean.drop_duplicates(subset=['canon_smiles'], keep='first')

    cleaned_splits = {
        'train': train_ids,
        'val': val_final_df['spectrum_id'].tolist(),
        'test': test_final_df['spectrum_id'].tolist()
    }

    stats = {
        'original_train_count': len(splits['train']),
        'cleaned_train_count': len(cleaned_splits['train']),
        'original_val_count': len(splits['val']),
        'cleaned_val_count': len(cleaned_splits['val']),
        'removed_from_val': len(splits['val']) - len(cleaned_splits['val']),
        'original_test_count': len(splits['test']),
        'cleaned_test_count': len(cleaned_splits['test']),
        'removed_from_test': len(splits['test']) - len(cleaned_splits['test']),
    }

    return cleaned_splits, stats


def make_split(dataset, seed, output_dir,
               frac_train: float = 0.8,
               frac_valid: float = 0.1,
               frac_test: float = 0.1):

    """
    Generates stratified splits

    Parameters:
        dataset
            Data to be splited
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


def preprocess_and_split(mgf_path, seed, output_dir=REPO_ROOT / "src/data/artifacts", num_spectra=None,
                         frac_valid: float = 0.1,
                         frac_test: float = 0.1,
                         remove_train_duplicates: bool = False):

    """
    Function that splits the data and calculates some of the essential parameters
    Cleans duplicate compounds after splitting, on the test and val set
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
        frac_valid : float
            Fraction of data to use for validation
        frac_test : float
            Fraction of data to use for testing
        remove_train_duplicates : bool
    """

    output_dir = Path(output_dir)
    seed_dir = output_dir / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading spectra and SMILES from MGF')
    mgf_spectra = mgf_get_spectra(mgf_path, num_spectra=num_spectra)
    print(f'Loaded {len(mgf_spectra)} spectra from MGF')

    smiles_df = mgf_get_smiles(mgf_spectra, as_dataframe=True)
    print(f'Loaded {len(smiles_df)} SMILES')

    print('\n2. Applying filtering and spectrum processing')

    smiles_df['canon_smiles'] = canonicalize_smiles(smiles_df['smiles'].tolist())
    smiles_df = smiles_df.dropna(subset=['canon_smiles'])

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

    filtered_smiles = smiles_df[smiles_df['spectrum_id'].isin(spectrum_ids)].copy()

    smiles_list = filtered_smiles['canon_smiles'].tolist()
    ids_list = filtered_smiles['spectrum_id'].tolist()

    dataset = SmilesDataset(smiles=smiles_list, ids=ids_list)
    dataset = MorganFingerprint().featurize(dataset)
    dataset._y = dataset.X

    print('\n4. Data Splitting')

    test_seed = 1
    val_seed = 2

    split = MultiTaskStratifiedSplitter()

    dataset_rest, _, test_dataset = split.train_valid_test_split(dataset, frac_train=1-frac_test, frac_valid=0,
                                                                 frac_test=frac_test, seed=test_seed)

    frac_value_adjusted = frac_valid / (1-frac_test)

    train_dataset, val_dataset, _ = split.train_valid_test_split(dataset_rest, frac_train=1 - frac_value_adjusted,
                                                                 frac_valid=frac_value_adjusted, frac_test=0, seed=val_seed)

    raw_splits = {'train': train_dataset.ids, 'val': val_dataset.ids, 'test': test_dataset.ids}

    clean_splits_dict, cleaning_stats = clean_splits(raw_splits, filtered_smiles, remove_train_duplicates)

    split_pkl = output_dir / str(seed) / 'split_ids.pkl'
    with split_pkl.open('wb') as f:
        pickle.dump(clean_splits_dict, f)

    fp_df = pd.DataFrame(dataset.X, columns=[f'fp_{i}' for i in range(dataset.X.shape[1])])
    fp_df['spectrum_id'] = dataset.ids
    fingerprints_pkl = output_dir / str(seed) / 'fingerprints.pkl'
    fp_df.to_pickle(fingerprints_pkl)

    print('\n5. Generating Split Statistics')

    id_to_label = dict(zip(dataset.ids, dataset._y))

    train_labels = np.array([id_to_label[spec_id] for spec_id in clean_splits_dict['train']])
    val_labels = np.array([id_to_label[spec_id] for spec_id in clean_splits_dict['val']])
    test_labels = np.array([id_to_label[spec_id] for spec_id in clean_splits_dict['test']])

    stats_df, table_styled = generate_data_stats(train_labels, test_labels, val_labels)

    stats_csv_path = seed_dir / 'split_statistics.csv'
    stats_html_path = seed_dir / 'split_statistics.html'

    with open(stats_html_path, 'w') as f:
        f.write(table_styled.to_html())
    stats_df.to_csv(stats_csv_path, index=False)

    final_train_count = len(clean_splits_dict['train'])
    final_val_count = len(clean_splits_dict['val'])
    final_test_count = len(clean_splits_dict['test'])

    total_final = final_train_count + final_val_count + final_test_count

    summary_data = {
        "mode": "unique_molecules" if remove_train_duplicates else "augmented_train",
        "split_seed": seed,
        "test_seed": test_seed,
        "val_seed": val_seed, # seed fixa para o val e test set

        "final_counts": {
            "train": final_train_count,
            "val": final_val_count,
            "test": final_test_count,
            "total": total_final
        },

        "final_fractions": {
            "train": round(final_train_count / total_final, 4),
            "val": round(final_val_count / total_final, 4),
            "test": round(final_test_count / total_final, 4)
        },

        "cleaning_impact": cleaning_stats
    }

    summary_path = seed_dir / 'split_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    if split_pkl.exists() and fingerprints_pkl.exists():
        print(f'Split IDs and Fingerprints saved')
    else:
        print('Files not found')

    return clean_splits_dict


def preprocess_and_split_2(mgf_path, seed, output_dir=REPO_ROOT / "src/data/artifacts", num_spectra=None,
                           frac_valid: float = 0.1,
                           frac_test: float = 0.1):

    """
    Function that splits the data and calculates some of the essential parameters
    Cleans duplicate compounds before splitting
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
    print(f'Loaded {len(smiles_df)} SMILES')

    print('\n2. Applying filtering and spectrum processing')

    smiles_df['canon_smiles'] = canonicalize_smiles(smiles_df['smiles'].tolist())
    smiles_df = smiles_df.dropna(subset=['canon_smiles'])

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

    filtered_smiles = smiles_df[smiles_df['spectrum_id'].isin(spectrum_ids)].copy()

    smiles_list = filtered_smiles['canon_smiles'].tolist()
    ids_list = filtered_smiles['spectrum_id'].tolist()

    dataset = SmilesDataset(smiles=smiles_list, ids=ids_list)
    dataset = MorganFingerprint().featurize(dataset)
    dataset._y = dataset.X

    dataset.remove_duplicates(inplace=True)

    print('\n4. Data Splitting')

    splitter = MultiTaskStratifiedSplitter()
    train_dataset, val_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=1.0 - frac_valid - frac_test,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed
    )

    splits = {'train': train_dataset.ids, 'val': val_dataset.ids, 'test': test_dataset.ids}

    split_pkl = seed_dir / 'split_ids.pkl'
    with split_pkl.open('wb') as f:
        pickle.dump(splits, f)

    final_train_count = len(splits['train'])
    final_val_count = len(splits['val'])
    final_test_count = len(splits['test'])
    total_final = final_train_count + final_val_count + final_test_count

    summary_data = {
        "mode": "Full compounds cleaning",
        "seed": seed,
        "final_counts": {
            "train": final_train_count,
            "val": final_val_count,
            "test": final_test_count,
            "total": total_final
        },
        "fractions": {
            "train": round(final_train_count / total_final, 4),
            "val": round(final_val_count / total_final, 4),
            "test": round(final_test_count / total_final, 4)
        }
    }

    with open(seed_dir / 'split_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=4)

    fp_df = pd.DataFrame(dataset.X, columns=[f'fp_{i}' for i in range(dataset.X.shape[1])])
    fp_df['spectrum_id'] = dataset.ids
    fingerprints_pkl = seed_dir / 'fingerprints.pkl'
    fp_df.to_pickle(fingerprints_pkl)

    id_to_label = dict(zip(dataset.ids, dataset._y))

    train_labels = np.array([id_to_label[spec_id] for spec_id in splits['train']])
    val_labels = np.array([id_to_label[spec_id] for spec_id in splits['val']])
    test_labels = np.array([id_to_label[spec_id] for spec_id in splits['test']])

    stats_df, table_styled = generate_data_stats(train_labels, test_labels, val_labels)

    stats_csv_path = seed_dir / 'split_statistics.csv'
    stats_html_path = seed_dir / 'split_statistics.html'

    with open(stats_html_path, 'w') as f:
        f.write(table_styled.to_html())
    stats_df.to_csv(stats_csv_path, index=False)

    if split_pkl.exists() and fingerprints_pkl.exists():
        print(f'Split IDs and Fingerprints saved')
    else:
        print('Files not found')

    return splits
