# Generic functions that can be reused
import os
import numpy as np
from pyteomics import mgf
from typing import Tuple
import re
from collections import Counter


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
    spectra = mgf.read(mgf_data, use_index=False)

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


def check_mgf_spectra(spectra: list, max_peak_threshold: int = 10000, percentile: int = None):

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
            'median': float(np.median(n_peaks)),
            'percentile': {
                '25%': float(np.percentile(n_peaks, 25)),
                '75%': float(np.percentile(n_peaks, 75)),
                '90%': float(np.percentile(n_peaks, 90)),
                '95%': float(np.percentile(n_peaks, 95)),
                '99%': float(np.percentile(n_peaks, 99)),
                f'{percentile}': float(np.percentile(n_peaks, percentile))
            }
        }
    }

    return stats


def spectra_integrator(
        mz_array: np.ndarray,
        int_array: np.ndarray,
        mass_error: float) -> Tuple[np.ndarray, np.ndarray]:

    """
    Merge close m/z peaks within a given mass error tolerance

    Parameters:
        mz_array : np.ndarray
            Array of m/z values
        int_array : np.ndarray
            Array of intensity values
        mass_error : float
            Maximum allowed m/z distance between peaks to be merged

    Returns:
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
                Array of merged m/z values
                Array of corresponding summed intensities
    """

    order = np.argsort(int_array, kind='mergesort')[::-1]

    mz_result = np.zeros_like(mz_array)
    int_result = np.zeros_like(int_array)

    count = 0

    for idx in order:
        if mz_array[idx] != 0:
            close = np.where(np.abs(mz_array - mz_array[idx]) <= mass_error)[0]

            summed_intensity = np.sum(int_array[close])
            weighted_mz = np.sum(mz_array[close] * int_array[close]) / summed_intensity

            mz_result[count] = weighted_mz
            int_result[count] = summed_intensity

            mz_array[close] = 0
            int_array[close] = 0

            count += 1

    return mz_result[:count], int_result[:count]


def mgf_spectrum_deconvoluter(
        spectrum_obj: Tuple[int, dict],
        min_num_peaks: int,
        max_num_peaks: int,
        noise_rmv_threshold: float,
        mass_error: float,
        mz_vocabs: list,
        log: bool):

    """
    Applies a series of preprocessing steps including noise removal, peak count validation, precursor m/z validation and
    peak normalization

    Parameters:
        spectrum_obj : Tuple[int, dict]
            Tuple containing the index and a dictionary representing a spectrum
        min_num_peaks : int
            Minimum number of peaks required to consider the spectrum valid
        max_num_peaks : int
            Maximum number of peaks allowed in the spectrum
        noise_rmv_threshold : float
            Proportional threshold (between 0 and 1) to remove noise. Peaks with
            intensity below this fraction of the maximum intensity are discarded
        mass_error : float
            Tolerance for peak merging in m/z units during integration
        mz_vocabs : list
            List of reference m/z values used for tokenization
        log : bool
            If True, it prints any error messages that may be triggered when the function is used

    Returns:
        tuple or None
            Returns a tuple containing:
                Identifier of the spectrum
                Tokenized m/z peak indices
                Tokenized precursor m/z index
                Normalized intensity values

            If the spectrum fails any quality filter, returns None.
    """

    i, spectrum = spectrum_obj

    mz_array = spectrum.get('m/z array', [])
    spectrum_id = spectrum["params"].get("spectrum_id", f"Spectrum {i+1}")
    n_peaks = len(mz_array)

    # Verifica se o numero de picos é maior que o min e menor que o max
    if n_peaks < min_num_peaks or n_peaks > max_num_peaks:
        if log:
            print(f'[{i}] Rejected spectrum: {n_peaks}')
        return None

    # Verifica se o precursor está dentro do valor fornecido
    precursor_mz, in_range = check_mz_precursor(spectrum, mz_vocabs)
    if not in_range:
        if log:
            print(f'[{i}] Rejected m/z precursor ({precursor_mz})')
        return None

    # Extração do valor massa carga e intensidades
    mz_array = np.array(spectrum['m/z array'], dtype=float)
    int_array = np.array(spectrum['intensity array'], dtype=float)

    # Remove picos abaixo do limiar de ruído
    threshold = noise_rmv_threshold * np.max(int_array)
    keep = int_array >= threshold
    mz_array = mz_array[keep]
    int_array = int_array[keep]
    n_peaks = len(mz_array)

    if n_peaks < min_num_peaks:
        if log:
            print(f'[{i}] Rejected after noise filtering: {n_peaks} peaks left')
        return None

    max_int = np.max(int_array)
    if max_int == 0:
        if log:
            print(f'[{i}] Zero max intensity before integration')
        return None
    int_array = int_array / max_int

    mz_array, int_array = spectra_integrator(mz_array, int_array, mass_error)

    n_peaks = len(mz_array)
    if n_peaks < min_num_peaks:
        if log:
            print(f'[{i}] Rejected after integration: {n_peaks} peaks left')
        return None

    # Ordenar os picos por intensidade
    order = np.argsort(int_array)[::-1]
    if n_peaks > max_num_peaks:
        order = order[:max_num_peaks]
    mz_array = mz_array[order]
    int_array = int_array[order]

    # Filtra picos que estão dentro do intervalo definido por mz_vocabs
    mz_min, mz_max = mz_vocabs[0], mz_vocabs[-1]
    in_range = (mz_array >= mz_min) & (mz_array <= mz_max)
    mz_array = mz_array[in_range]
    int_array = int_array[in_range]
    n_filtered_peaks = len(mz_array)

    if n_filtered_peaks < min_num_peaks or n_filtered_peaks < 0.9 * len(order):
        if log:
            print(f'[{i}] Rejected after mz range filtering: {n_filtered_peaks} peaks left')
        return None

    # tokenizar os valores de m/z e do precursor
    tokenized_mz = [np.argmin(np.abs(mz - mz_vocabs)) for mz in mz_array]
    tokenized_precursor = np.argmin(np.abs(precursor_mz - mz_vocabs))

    int_sum = np.sum(int_array)
    if int_sum == 0:
        if log:
            print(f'[{i}] Zero intensities detected')
        return None
    int_array = int_array / int_sum

    training_tuple = (
        spectrum_id,
        tokenized_mz,
        tokenized_precursor,
        int_array,
    )

    return training_tuple


def validate_mgf_structure(mgf_path):

    with open(mgf_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        scan_ids = []
        spectrum_ids = []
        spectrum_count = 0
        current_block = []

        for line in lines:
            line = line.strip()

            if line == "BEGIN IONS":
                current_block = []
            elif line == "END IONS":
                spectrum_count += 1
                block_text = "\n".join(current_block)

                scan_match = re.search(r'SCANS=(.+)', block_text)
                id_match = re.search(r'SPECTRUM_ID=(.+)', block_text)

                scan_ids.append(scan_match.group(1).strip() if scan_match else "MISSING")
                spectrum_ids.append(id_match.group(1).strip() if id_match else "MISSING")
            else:
                current_block.append(line)

        scan_counter = Counter(scan_ids)
        spectrum_id_counter = Counter(spectrum_ids)

        duplicate_scans = [k for k, v in scan_counter.items() if v > 1 and k != "MISSING"]
        duplicate_specids = [k for k, v in spectrum_id_counter.items() if v > 1 and k != "MISSING"]

        print(f"\nTotal number of spectra found: {spectrum_count}")
        print(f"Missing SCANS: {scan_ids.count('MISSING')}")
        print(f"Missing SPECTRUM_ID: {spectrum_ids.count('MISSING')}")
        print(f"Duplicate SCANS: {len(duplicate_scans)} -> {duplicate_scans[:5]}...")
        print(f"Duplicate SPECTRUM_ID: {len(duplicate_specids)} -> {duplicate_specids[:5]}...")
