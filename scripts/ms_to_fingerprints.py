from src.utils import *
import matplotlib.pyplot as plt


def mgf_deconvoluter(mgf_data, mz_vocabs, min_num_peaks, max_num_peaks, noise_rmv_threshold, mass_error, log,
                     plot=False):

    """
    Iterates through a list of MGF spectra, applying filtering and preprocessing steps via `mgf_spectrum_deconvoluter`

    Parameters:
        mgf_data : list of dict
            List of spectra, where each spectrum is represented as a dictionary
        mz_vocabs : list
            List of reference m/z values used for tokenization
        min_num_peaks : int
            Minimum number of peaks required to consider the spectrum valid
        max_num_peaks : int
            Maximum number of peaks allowed in the spectrum
        noise_rmv_threshold : float
            Proportional threshold to remove noise. Peaks below this fraction of
            the maximum intensity are discarded
        mass_error : float
            Tolerance for peak merging in m/z units during integration
        log : bool
            If True, it prints any error messages that may be triggered when the function is used
        plot : bool, optional
            If True, plots the tokenized spectra that pass the filtering

    """
    processed_spectra = []

    for i, spectrum in enumerate(mgf_data):
        result = mgf_spectrum_deconvoluter(
            spectrum_obj=(i, spectrum),
            min_num_peaks=min_num_peaks,
            max_num_peaks=max_num_peaks,
            noise_rmv_threshold=noise_rmv_threshold,
            mass_error=mass_error,
            mz_vocabs=mz_vocabs,
            log=log,
        )

        if result is not None:
            processed_spectra.append(result)

            if plot:
                spectrum_id, tokenized_mz, tokenized_precursor, intensities = result
                print(f"\nID: {spectrum_id}")
                print(f"Tokenized m/z: {tokenized_mz}")
                print(f"Tokenized precursor: {tokenized_precursor}")
                print(f"Normalized intensities: {intensities}")

                plt.figure(figsize=(10, 5))
                plt.bar(tokenized_mz, intensities, width=5, color='royalblue')
                plt.title(f"Mass Spectrum {spectrum_id}")
                plt.xlabel("Tokenized m/z")
                plt.ylabel("Normalized Intensity")
                plt.tight_layout()
                plt.show()

    return processed_spectra
