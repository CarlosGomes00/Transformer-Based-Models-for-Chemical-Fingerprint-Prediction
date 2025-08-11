from data.mgf_tools.mgf_checks import *
from data.mgf_tools.mgf_readers import *
from data.mgf_tools.mgf_get import mgf_get_spectra
from src.utils import path_check
from data.mgf_tools.mgf_checks import check_mgf_data


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
    try:
        print("Checking path...")
        path_check(mgf_data=mgf_path)

        print("Checking file...")
        spectra = mgf_get_spectra(mgf_path)

        print("Checking IDs...")
        result_ids = check_spectrum_ids(mgf_path)

        print("Checking Compounds...")
        result_compounds = check_mgf_data(spectra)

        print("Checking the first compound headers")
        result_headers = mgf_read_headers(mgf_path, num_spectra=1)

        return result_ids, result_compounds, result_headers

    except FileNotFoundError:
        return f"The file {mgf_path} was not found"


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_validation.py <path/to/file.mgf>")
    else:
        mgf_path = sys.argv[1]
        validation_results = validate_mgf_file(mgf_path)
        if isinstance(validation_results, str):
            print(validation_results)
        else:
            result_ids, result_compounds, result_headers = validation_results
            print(f"\n--- Validation Summary ---")
            print(f"ID Check Result: {result_ids}")
            print(f"Compound Check Result: {result_compounds}")
            print(f"Header Check Result (First Compound): {result_headers}")
            print("Check completed successfully!")
