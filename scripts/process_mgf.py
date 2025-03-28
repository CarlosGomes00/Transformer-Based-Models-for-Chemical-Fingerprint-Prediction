from pyteomics import mgf
import matplotlib as plt


def mgf_headers_info(mgf_data: str, num_spectra: int = 3):

    """
    Reads the headers of the spectra in an .mgf file
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


def mgf_all_info(mgf_data: str, num_spectra: int = 3):

    """
    Read all the information about the spectra in a .mgf file
    The main purpose of this function is to see if the iteration over the spectra is taking place properly

    Parameters:
        mgf_data : str
            Path to the dataset to be used
        num_spectra : int
            Number of spectra info to be read, 3 by default

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


def plot_spectrum(spectrum: dict, title: str = None):

    """
    Plots a spectrum based on the information in the .mgf file

    Parameters:
        spectrum : dict
            A dictionary containing spectrum data (m/z and intensity arrays)
        title : str
            Title of the plot
    """

    mz_values = spectrum['m/z array']
    intensity_values = spectrum['intensity array']

    if title is None:
        title = spectrum["params"].get("spectrum_id")

    plt.figure(figsize=(10, 5))
    plt.bar(mz_values, intensity_values, width=0.5, color='red')
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(f"Mass Spectrum - {title}")
    plt.show()

    return
