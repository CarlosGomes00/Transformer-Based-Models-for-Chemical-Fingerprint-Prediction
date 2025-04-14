import matplotlib.pyplot as plt
import os


def plot_spectrum(spectrum: dict, save: bool = False, save_path: str = r"/plots"):
    """
    Plots a spectrum based on the information in the .mgf file
    You need to call the function "mgf_get_spectra" in the target spectrum

    Parameters:
        spectrum : dict
            A dictionary containing spectrum data (m/z and intensity arrays)
        save : bool, optional
            If True, saves the plot.
        save_path : str, optional
            Path where to save the plot.

    Return:
        MS/MS spectra
    """

    if not isinstance(spectrum, dict):
        raise TypeError("Spectrum must be a dictionary")

    mz_values = spectrum['m/z array']
    intensity_values = spectrum['intensity array']
    spectrum_id = spectrum['params'].get("spectrum_id", "Unknown_ID")

    plt.figure(figsize=(10, 5))
    plt.bar(mz_values, intensity_values, width=0.5, color='red')
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(f"Mass Spectrum - {spectrum_id}")

    if save:
        if save_path is None:
            save_path = f"{spectrum_id}"
        elif os.path.isdir(save_path):
            save_path = os.path.join(save_path, f"{spectrum_id}")
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)
        print(f"Plot saved as: {save_path}")
        plt.close()

    else:
        plt.show()

    return


def plot_spectra(spectra: list, num_spectra: int = None, save: bool = False, save_path: str = r"/plots"):
    """
    Plots multiple spectra from a list of spectrum dictionaries and optionally saves them
    You need to call the function "mgf_get_spectra"


    Parameters:
        spectra : list of dict
            A list of dictionaries containing spectrum data (m/z and intensity arrays)
        num_spectra : int, optional
            The number of spectra to plot, all by default
        save : bool, optional
            If True, saves the plot.
        save_path : str, optional
            Path where plots will be saved, plots by default

    Returns:
        None
    """

    if not isinstance(spectra, list) or not all(isinstance(spectrum, dict) for spectrum in spectra):
        raise TypeError("Spectra must be a list of dictionaries")

    if num_spectra is None:
        num_spectra = len(spectra)
    num_spectra = min(num_spectra, len(spectra))

    if save:
        os.makedirs(save_path, exist_ok=True)

    for i, spectrum in enumerate(spectra[:num_spectra]):
        mz_values = spectrum['m/z array']
        intensity_values = spectrum['intensity array']
        title = spectrum["params"].get("spectrum_id", f"Spectrum {i+1}")

        if mz_values.size == 0 or intensity_values.size == 0:
            print(f"Spectrum {i+1} has no data to plot")
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(mz_values, intensity_values, width=0.5, color='red', label=title)
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title(f"Mass Spectrum - {title}")

        if save:
            filename = os.path.join(save_path, f"{title}.jpg")
            plt.savefig(filename, dpi=300)
            print(f"Plot saved as: {filename}")
            plt.close()

        else:
            plt.show()

    return
