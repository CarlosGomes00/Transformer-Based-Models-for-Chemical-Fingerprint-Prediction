import numpy as np
from pyteomics import mgf
import re
from collections import Counter


def check_spectrum_ids(mgf_data: str):

    """
    Checks if all spectra in an MGF file contain a 'spectrum_id' field

    Parameters:
        mgf_data : str
            Path to the MGF file to be checked

    Raises:
        ValueError
            If one or more spectra are missing the 'spectrum_id' field
    """

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


def validate_mgf_structure(mgf_path):

    """
    Validates the structure of an .mgf file by checking the presence and uniqueness of
    SCANS and SPECTRUM_ID fields within spectra blocks

    Parameters:
        mgf_path : str
            Path to the MGF file to be validated.

    Returns:
        None

    Prints:
        - Total number of spectra found
        - Number of missing SCANS entries
        - Number of missing SPECTRUM_ID entries
        - Number and examples of duplicate SCANS
        - Number and examples of duplicate SPECTRUM_IDs
    """

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
