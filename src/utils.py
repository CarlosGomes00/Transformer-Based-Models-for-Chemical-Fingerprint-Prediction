# Generic functions that can be reused

import os
from pyteomics import mgf


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


def check_mgf_data(mgf_data: str):

    """
    Analyzes an .MGF file and summarizes key statistics

    Parameters:
        mgf_data : str
            Path to the dataset to be used

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

    spectra = list(mgf.read(mgf_data, index_by_scans=True))

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
