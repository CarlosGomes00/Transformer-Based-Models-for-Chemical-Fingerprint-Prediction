from src.utils import *
import matplotlib.pyplot as plt


def mgf_deconvoluter(mgf_data, mz_vocabs, min_num_peaks, max_num_peaks, noise_rmv_threshold, mass_error, log,
                     plot=False, **kwargs):

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
            **kwargs
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
