from src.mgf_tools.mgf_checks import *
from src.mgf_tools.mgf_readers import *
from src.mgf_tools.mgf_get import mgf_get_spectra
from src.utils import path_check


def validate_mgf_file(mgf_path: str):

    """
    Validates an MGF file by performing several checks:

    - Verifies that the path is valid.
    - Parses spectra and checks for missing or duplicate spectrum IDs.
    - Checks compound presence and structure.
    - Displays headers for the first compound as a sanity check.

    Parameters:
        mgf_path (str): Path to the .mgf file.
    """

    print("Checking path...")
    path_check(mgf_data=mgf_path)

    spectra = mgf_get_spectra(mgf_path)

    print("Checking IDs...")
    check_spectrum_ids(mgf_path)

    print("Checking Compounds...")
    check_mgf_data(spectra)

    print("Checking the first compound headers")
    mgf_read_headers(spectra, num_spectra=1)

    return "Check completed"
