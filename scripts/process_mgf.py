from pyteomics import mgf



def mgf_read_headers(mgf_data: str, num_spectra: int = 1):

    """
    Reads the headers of the spectra in an .mgf file
    The main purpose of this function is to see if the iteration over the spectra is taking place properly

    Parameters:
        mgf_data : str
            Path to the dataset to be used
        num_spectra : int
            Number of spectra info to be read, 1 by default

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


def mgf_read_all(mgf_data: str, num_spectra: int = 1):

    """
    Read all the information about the spectra in a .mgf file
    The main purpose of this function is to see if the iteration over the spectra is taking place properly

    Parameters:
        mgf_data : str
            Path to the dataset to be used
        num_spectra : int
            Number of spectra info to be read, 1 by default

    Returns:
        Information about each spectrum
    """

    try:
        spectra = mgf.read(mgf_data, index_by_scans=True)

        for i, spectrum in enumerate(spectra):
            if i >= num_spectra:
                break
            print(f"Spectrum {i + 1} - Parameters:")
            print(spectrum['params'])
            print(spectrum['m/z array'])
            print(spectrum['intensity array'])
            print("\n")
    except Exception as e:
        print(f"Error reading .MGF file: {e}")



def mgf_get_spectra(mgf_data: str, num_spectra: int = None) -> dict:

    """
    Read all the information about the spectra in a .mgf file and return it as a dictionary
    It is mainly used as a reading base for the spectra plot function

    Parameters:
        mgf_data : str
            Path to the dataset to be used
        num_spectra : int
            Number of spectra info to be read. All by default

    Returns:
        Dictionary of each spectrum
    """

    spectra = list(mgf.read(mgf_data, index_by_scans=True))

    if len(spectra) == 0:
        print("Error reading .MGF file")
        return None

    if num_spectra is None:
        return spectra

    return spectra[:num_spectra] if num_spectra > 1 else spectra[0]



