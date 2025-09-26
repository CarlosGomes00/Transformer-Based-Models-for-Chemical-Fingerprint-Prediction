import pickle
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.data.split_prep_tools.collate_fn import SpectraCollateFn
from data.mgf_tools.mgf_get import mgf_get_spectra
from src.utils import mgf_deconvoluter
from src.config import mgf_path, min_num_peaks, noise_rmv_threshold, mass_error

REPO_ROOT = Path(__file__).resolve().parents[2]


class SpectraDataset(Dataset):

    """
    Creates a PyTorch Dataset for processed mass spectra

    This class encapsulates a list of spectra already processed (tuples containing m/z, intensities, mask and ID)
    and provides a standard interface to the PyTorch DataLoader
    """

    def __init__(self, processed_spectra):
        self.data = processed_spectra

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def data_loader(seed, batch_size: int = 32, num_workers=4, num_spectra: int = None,
                mgf_path: str = mgf_path, max_num_peaks: int = None, mz_vocabs=None):

    mgf_spectra = mgf_get_spectra(mgf_path, num_spectra)

    if max_num_peaks is None or mz_vocabs is None:
        raise ValueError("max_num_peaks and mz_vocabs must be provided to ensure consistency")

    artifacts_dir = REPO_ROOT / "src/data/artifacts"
    split_pkl = artifacts_dir / str(seed) / 'split_ids.pkl'
    fingerprints_pkl = artifacts_dir / str(seed) / 'fingerprints.pkl'

    if not split_pkl.exists():
        raise FileNotFoundError("Split file not found")

    with open(split_pkl, 'rb') as f:
        splits = pickle.load(f)
        print(f"Loaded splits: Train({len(splits['train'])}), Val({len(splits['val'])}), Test({len(splits['test'])})")

    if not fingerprints_pkl.exists():
        raise FileNotFoundError("Fingerprints file not found")

    all_fingerprints = pd.read_pickle(fingerprints_pkl)

    loaders = {}

    max_seq_len = max_num_peaks + 1
    vocab_size = len(mz_vocabs)

    for split_name, split_ids in splits.items():
        split_ids_set = set(split_ids)

        filtered_spectra = []
        for spectrum in mgf_spectra:
            spectrum_id = spectrum['params'].get('spectrum_id', str(len(filtered_spectra)))
            if spectrum_id in split_ids_set:
                filtered_spectra.append(spectrum)

        print(f"Filtered to {len(filtered_spectra)} spectra for {split_name}")

        split_fingerprints = all_fingerprints[
            all_fingerprints['spectrum_id'].isin(split_ids_set)].copy()

        processed_spectra = mgf_deconvoluter(
            mgf_data=filtered_spectra,
            mz_vocabs=mz_vocabs,
            min_num_peaks=min_num_peaks,
            max_num_peaks=max_num_peaks,
            noise_rmv_threshold=noise_rmv_threshold,
            mass_error=mass_error,
            log=False
        )

        collate_fn = SpectraCollateFn(split_fingerprints, max_seq_len=max_seq_len, vocab_size=vocab_size)

        dataset = SpectraDataset(processed_spectra)

        is_train = split_name == 'train'

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=is_train,
            num_workers=num_workers,
            persistent_workers=True
        )

    print(f"{split_name} DataLoader ready: {len(dataset)} samples")

    return loaders

#TODO Alterar/adicionar método ao dataloader para ser compativel com o método .predict?