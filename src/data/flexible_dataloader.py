"""
The preprocessing logic is inside the dataloader_f function.

This file would be used for quick tests, debugging, and scenarios where you want to specify a small num_spectra directly
from the function call.

"""

from torch.utils.data import Dataset, DataLoader
from src.data.collate_fn import SpectraCollateFn
from src.mgf_tools.mgf_get import mgf_get_spectra
from src.utils import mgf_deconvoluter
from src.config import *

mgf_path = r"/Users/carla/PycharmProjects/Mestrado/Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/datasets/raw/cleaned_gnps_library.mgf"


class SpectraDataset(Dataset):

    def __init__(self, processed_spectra):
        self.data = processed_spectra

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


collate_fn = SpectraCollateFn()


def data_loader_f(batch_size: int = 4, shuffle: bool = True, num_workers: int = 0, num_spectra: int = None,
                  mgf_path: str = mgf_path):

    mgf_spectra = mgf_get_spectra(mgf_path, num_spectra)

    processed_spectra = mgf_deconvoluter(mgf_data=mgf_spectra, mz_vocabs=mz_vocabs, min_num_peaks=min_num_peaks,
                                         max_num_peaks=max_num_peaks, noise_rmv_threshold=noise_rmv_threshold,
                                         mass_error=mass_error, log=True)

    flexible_dataset = SpectraDataset(processed_spectra)

    return DataLoader(
        flexible_dataset,
        batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, num_workers=num_workers)
