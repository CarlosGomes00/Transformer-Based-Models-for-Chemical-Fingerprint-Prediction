import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm


def plot_spectrum(spectrum: dict, save: bool = False, save_path: str = "outputs"):
    """
    Plots a spectrum based on the information in the .mgf file
    You need to call the function "mgf_get_spectra" in the target spectrum

    Parameters:
        spectrum : dict
            A dictionary containing spectrum datasets (m/z and intensity arrays)
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


def plot_spectra(spectra: list, num_spectra: int = None, save: bool = False, save_path: str = "outputs"):
    """
    Plots multiple spectra from a list of spectrum dictionaries and optionally saves them
    You need to call the function "mgf_get_spectra"


    Parameters:
        spectra : list of dict
            A list of dictionaries containing spectrum datasets (m/z and intensity arrays)
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

    for i, spectrum in enumerate(tqdm(spectra[:num_spectra], desc="Plotting Spectra", unit="spectrum")):
        mz_values = spectrum['m/z array']
        intensity_values = spectrum['intensity array']
        title = spectrum["params"].get("spectrum_id", f"Spectrum {i+1}")

        if mz_values.size == 0 or intensity_values.size == 0:
            print(f"Spectrum {i+1} has no datasets to plot")
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


def plot_spectra_distribution(spectra: list, n_compounds: int = 20, top_percent: float = None):

    """
    Plot the distribution of spectra by compound

    Parameters:
        spectra : list of dict
            A list of dictionaries containing spectrum datasets (m/z and intensity arrays)
        n_compounds : int, optional
            Number of compounds to represent, the 20 most frequent by default
        top_percent : float, optional
            If defined, filters compounds to include only those contributing up to this percentage
            of total spectra
    """

    compound_frequency = {}

    for spectrum in spectra:
        params = spectrum["params"]
        compound = params.get("compound_name", None)

        if compound:
            if compound in compound_frequency:
                compound_frequency[compound] += 1
            else:
                compound_frequency[compound] = 1

    if top_percent is None and n_compounds is not None:
        sorted_compounds = sorted(compound_frequency.items(), key=lambda x: x[1], reverse=True)[:n_compounds]
    else:
        sorted_compounds = sorted(compound_frequency.items(), key=lambda x: x[1], reverse=True)

    if top_percent is not None:
        assert 0 < top_percent <= 99
        total = sum(freq for _, freq in sorted_compounds)
        cumulative = 0
        filtered = []
        for compound, freq in sorted_compounds:
            cumulative += freq
            filtered.append((compound, freq))
            if (cumulative / total) * 100 >= top_percent:
                break
        sorted_compounds = filtered

    compounds, frequencies = zip(*sorted_compounds)

    short_names = [name[:20] + '...' if len(name) > 20 else name for name in compounds]

    plt.figure(figsize=(10, 6))
    plt.bar(short_names, frequencies)
    plt.xticks(rotation=90)
    plt.xlabel("Compound")
    plt.ylabel("Frequency")
    plt.title("Distribution of compounds")
    plt.tight_layout()
    plt.show()



def plot_peak_distribution_comparison(raw_spectra: list, processed_spectra: list, max_peaks_plot=200):
    """
    Plots comparative histograms of peak counts (Before vs After processing).

    Parameters:
        raw_spectra: List of dicts (from pyteomics.mgf.read)
        processed_spectra: List of tuples (from deconvoluter)
        max_peaks_plot: Limit x-axis for better visualization (zoom in relevant area)
    """

    print("Extracting raw stats...")
    raw_counts = []
    for s in raw_spectra:
        mz = s.get('m/z array', [])
        raw_counts.append(len(mz))

    print("Extracting processed stats...")
    proc_counts = []
    for s in processed_spectra:
        mz_tokens = s[2]
        proc_counts.append(len(mz_tokens))

    plt.figure(figsize=(12, 6))

    bins = np.arange(0, 200, 2)

    plt.hist(raw_counts, bins=bins, color='red', alpha=0.3, label='Raw', density=True)

    plt.hist(proc_counts, bins=bins, color='blue', alpha=0.5, label='Processed', density=True)

    plt.xlim(0, 200)
    plt.title("Impact of Processing on Peak Count Distribution")
    plt.xlabel("Number of Peaks per Spectrum")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.2)

    raw_med = np.median(raw_counts)
    proc_med = np.median(proc_counts)

    plt.text(0.7, 0.8, f"Raw Median: {raw_med:.0f}\nProc Median: {proc_med:.0f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return raw_counts, proc_counts