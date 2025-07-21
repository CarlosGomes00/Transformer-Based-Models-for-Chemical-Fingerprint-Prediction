from torch.utils.data import Dataset, DataLoader
from src.data.collate_fn import SpectraCollateFn
from data.mgf_tools.mgf_get import mgf_get_spectra
from src.utils import mgf_deconvoluter
from src.config import *


mgf_path = r"/Users/carla/PycharmProjects/Mestrado/Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/datasets/raw/cleaned_gnps_library.mgf"

mgf_spectra = mgf_get_spectra(mgf_path, num_spectra=None)

processed_spectra = mgf_deconvoluter(mgf_data=mgf_spectra, mz_vocabs=mz_vocabs, min_num_peaks=min_num_peaks,
                                     max_num_peaks=max_num_peaks, noise_rmv_threshold=noise_rmv_threshold,
                                     mass_error=mass_error, log=True)


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


collate_fn = SpectraCollateFn()

spectra_dataset = SpectraDataset(processed_spectra)


def data_loader(batch_size: int = 4, shuffle: bool = True, num_workers: int = 0):

    """
    Returns a DataLoader

    Params:
        batch_size : int
            Number of samples in each batch
        shuffle : bool
            Whether to shuffle the data
        num_workers : int
            Number of subprocesses to use for data loading
    """

    return DataLoader(spectra_dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      shuffle=shuffle,
                      num_workers=num_workers)
