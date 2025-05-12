from src.utils import *


def mgf_deconvoluter(mgf_data, mz_vocabs, min_num_peaks, max_num_peaks, noise_rmv_threshold, mass_error, log, **kwargs):

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

    return processed_spectra
