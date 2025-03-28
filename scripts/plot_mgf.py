import matplotlib.pyplot as plt

def plot_spectrum(spectrum: dict, title: str = None):

    """
    Plots a spectrum based on the information in the .mgf file

    Parameters:
        spectrum : dict
            A dictionary containing spectrum data (m/z and intensity arrays)
        title : str
            Title of the plot

    Return:
        MS/MS spectra
    """

    if not isinstance(spectrum, dict):
        raise TypeError("Spectrum must be a dictionary")

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

def plot_spectra(spectra: list, num_spectra: int):

    """"
    Recebe uma lista de dicionários e dá plot
    """

    return
