# Generic functions that can be reused

import os
import numpy as np
from pyteomics import mgf
from typing import Tuple


def path_check(mgf_data: str) -> bool:
    """
    Checks if the path to the dataset has been found

    Parameters:
        mgf_data : str
            Path to the dataset to be used

    Returns:
        bool
            True if file is found, False otherwise
    """

    if not os.path.exists(mgf_data):
        print(f"Error: File could not be found {os.path.abspath(mgf_data)}")
        return False
    else:
        print("File found!")


def check_spectrum_ids(mgf_data: str):
    spectra = mgf.read(mgf_data, index_by_scans=True)

    missing_ids = []

    for i, spectrum in enumerate(spectra):
        spectrum_id = spectrum['params'].get('spectrum_id', None)

        if not spectrum_id:
            missing_ids.append(i + 1)

    if missing_ids:
        raise ValueError(f"Error: Missing spectrum IDs in the following spectra: {', '.join(map(str, missing_ids))}")

    else:
        print("All spectra have valid IDs")


def check_mz_precursor(spectrum: dict, mz_vocabs: list[float] = [50, 2000]) -> Tuple[float | None, bool]:
    """
    Extracts and checks whether the PrecursorMZ of a spectrum is within the permitted range

    Parameters:
        spectrum : dict
            Dictionary containing the spectrum data
        mz_vocabs : list
            Sorted list of accepted lower and upper m/z limits

    Returns:
        Tuple: precursor_mz value or None, Is precursor_mz between the given range?
    """
    precursor_mz = spectrum['params'].get('precursor_mz', None)
    if isinstance(precursor_mz, (tuple, list)):
        precursor_mz = precursor_mz[0]
    if precursor_mz is None:
        return None, False

    try:
        precursor_mz = float(precursor_mz)
        in_range = mz_vocabs[0] <= precursor_mz <= mz_vocabs[-1]
        return precursor_mz, in_range
    except (ValueError, TypeError):
        return None, False


def check_mgf_data(spectra: list):

    """
    Analyzes an .MGF file and summarizes key statistics

    Parameters:
        spectra : list of dicts
            A list of dictionaries containing the spectra

    Returns:
        dict
            A dictionary containing:
                - 'Total compounds': Total number of spectra in the file
                - 'Unique compounds': Number of unique compound names identified
                - 'Unknown compounds': Number of spectra missing a compound name
                - 'Positive ionization mode': Number of spectra in positive ion mode
                - 'Negative ionization mode': Number of spectra in negative ion mode
                - 'Unknown ionization mode': Number of spectra where ion mode is unspecified or unrecognized
    """

    n_compounds = len(spectra)

    unique = set()
    unknown_compounds = 0
    pos_ion_mode = 0
    neg_ion_mode = 0
    unknown_ion_mode = 0

    for spectrum in spectra:
        params = spectrum['params']
        compound_name = params.get('compound_name', None)
        if compound_name and compound_name.strip():
            unique.add(compound_name)
        else:
            unknown_compounds += 1

        ion_mode = params.get('ionmode', None)
        if ion_mode == 'positive':
            pos_ion_mode += 1
        elif ion_mode == 'negative':
            neg_ion_mode += 1
        else:
            unknown_ion_mode += 1

    unique = len(unique)

    return {'Total compounds': n_compounds,
            'Unique compounds': unique,
            'Unknown compounds': unknown_compounds,
            'Positive ionization mode': pos_ion_mode,
            'Negative ionization mode': neg_ion_mode,
            'Unknown ionization mode': unknown_ion_mode}


def check_mgf_spectra(spectra: list, max_peak_threshold: int = 10000):

    """
    Analyze m/z values and peak counts from spectra

    Parameters:
            spectra : list of dict
                A list of dictionaries containing the spectra
            max_peak_threshold : int, optional
                Maximum number of peaks allowed in a spectrum to be considered valid

    Returns:
            dict
                A dictionary containing:
                    m/z range (min, max)
                    peak count statistics (min, max, mean, median)
    """

    mz_values = []
    n_peaks = []

    for spectrum in spectra:
        mz_array = spectrum.get("m/z array", [])
        if len(mz_array) == 0:
            continue

        if len(mz_array) > max_peak_threshold:
            continue

        mz_values.extend(mz_array)
        n_peaks.append(len(mz_array))

    stats = {
        'm/z range': (float(np.min(mz_values)), float(np.max(mz_values))),
        'peak count stats': {
            'min': int(np.min(n_peaks)),
            'max': int(np.max(n_peaks)),
            'mean': float(np.mean(n_peaks)),
            'median': float(np.median(n_peaks))
        }
    }

    return stats


def mgf_spectrum_deconvoluter(
        i: int,
        spectrum: dict,
        min_num_peaks: int,
        max_num_peaks: int,
        noise_rmv_threshold: float,
        allowed_spectral_entropy: float,
        mass_error: float,
        mz_vocab: list,
        log: bool,
        **kwargs):

    mz_array = spectrum.get('m/z array', [])
    if len(mz_array) < min_num_peaks:
        return None