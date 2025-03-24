from pyteomics import mgf

def test_mgf_iter(mgf_data: str, num_spectra: int = 3):

    """
    Reads the headers of the first spectra in an MGF file
    The main purpose of this function is to see if the iteration over the spectra is taking place properly

    Parameters:
        mgf_data : str
            Path to the dataset to be used
        num_spectra : int
            Number of spectra info to be read, 3 by default

    Returns:
        Some of the parameters for each spectrum
    """

    try:
        spectra = mgf.read(mgf_data, index_by_scans=True)

        for i, spectrum in enumerate(spectra):
            if i >= num_spectra:
                break
            print(f"Spectrum {i + 1} - Parameters:")
            print(spectrum['params'])
            print("\n")
    except Exception as e:
        print(f"Error reading .MGF file: {e}")
