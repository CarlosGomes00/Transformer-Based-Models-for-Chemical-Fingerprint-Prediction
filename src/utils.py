# Generic functions that can be reused
import os
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from src.config import mz_vocabs


def path_check(mgf_data: str) -> None:
    """
    Checks if the path to the datasets has been found

    Parameters:
        mgf_data : str
            Path to the datasets to be used

    Raises:
        FileNotFoundError
            If the file is not found at the specified path
    """

    if not os.path.exists(mgf_data):
        raise FileNotFoundError(f"Error: File could not be found {os.path.abspath(mgf_data)}")
    else:
        print("File found!")


def check_mz_precursor(spectrum: dict, mz_vocabs: list[float]) -> Tuple[float | None, bool]:
    """
    Extracts and checks whether the PrecursorMZ of a spectrum is within the permitted range

    Parameters:
        spectrum : dict
            Dictionary containing the spectrum datasets
        mz_vocabs : list
            List of reference m/z values used for tokenization

    Returns:
        Tuple: precursor_mz value or None, Is precursor_mz between the given range?
    """
    precursor_mz = spectrum['params'].get('precursor_mz', None)
    if isinstance(precursor_mz, (tuple, list)):
        precursor_mz = precursor_mz[0]
    if precursor_mz is None:
        return None, False

    try:
        precursor_mz = float(precursor_mz)
        in_range = mz_vocabs[0] <= precursor_mz <= mz_vocabs[-1]
        return precursor_mz, in_range
    except (ValueError, TypeError):
        return None, False


def spectra_integrator(
        mz_array: np.ndarray,
        int_array: np.ndarray,
        mass_error: float) -> Tuple[np.ndarray, np.ndarray]:

    """
    Merge close m/z peaks within a given mass error tolerance

    Parameters:
        mz_array : np.ndarray
            Array of m/z values
        int_array : np.ndarray
            Array of intensity values
        mass_error : float
            Maximum allowed m/z distance between peaks to be merged

    Returns:
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
                Array of merged m/z values
                Array of corresponding summed intensities
    """

    order = np.argsort(int_array, kind='mergesort')[::-1]

    mz_result = np.zeros_like(mz_array)
    int_result = np.zeros_like(int_array)

    count = 0

    for idx in order:
        if mz_array[idx] != 0:
            close = np.where(np.abs(mz_array - mz_array[idx]) <= mass_error)[0]

            summed_intensity = np.sum(int_array[close])
            weighted_mz = np.sum(mz_array[close] * int_array[close]) / summed_intensity

            mz_result[count] = weighted_mz
            int_result[count] = summed_intensity

            mz_array[close] = 0
            int_array[close] = 0

            count += 1

    return mz_result[:count], int_result[:count]


def mgf_spectrum_deconvoluter(
        spectrum_obj: Tuple[int, dict],
        min_num_peaks: int,
        max_num_peaks: int,
        noise_rmv_threshold: float,
        mass_error: float,
        mz_vocabs: list,
        log: bool):

    """
    Applies a series of preprocessing steps including noise removal, peak count validation, precursor m/z validation and
    peak normalization

    Parameters:
        spectrum_obj : Tuple[int, dict]
            Tuple containing the index and a dictionary representing a spectrum
        min_num_peaks : int
            Minimum number of peaks required to consider the spectrum valid
        max_num_peaks : int
            Maximum number of peaks allowed in the spectrum
        noise_rmv_threshold : float
            Proportional threshold (between 0 and 1) to remove noise. Peaks with
            intensity below this fraction of the maximum intensity are discarded
        mass_error : float
            Tolerance for peak merging in m/z units during integration
        mz_vocabs : list
            List of reference m/z values used for tokenization
        log : bool
            If True, it prints any error messages that may be triggered when the function is used

    Returns:
        tuple or None
            Returns a tuple containing:
                Identifier of the spectrum
                Tokenized m/z peak indices
                Tokenized precursor m/z index
                Normalized intensity values

            If the spectrum fails any quality filter, returns None.
    """

    i, spectrum = spectrum_obj

    mz_array = spectrum.get('m/z array', [])
    spectrum_id = spectrum["params"].get("spectrum_id", f"Spectrum {i+1}")
    n_peaks = len(mz_array)
    mz_vocabs = np.array(mz_vocabs, dtype=float)

    # Verifica se o numero de picos é maior que o min e menor que o max
    if n_peaks < min_num_peaks or n_peaks > max_num_peaks:
        if log:
            print(f'[{i}] Rejected spectrum: {n_peaks}')
        return None

    # Verifica se o precursor está dentro do valor fornecido
    precursor_mz, in_range = check_mz_precursor(spectrum, mz_vocabs)
    if not in_range:
        if log:
            print(f'[{i}] Rejected m/z precursor ({precursor_mz})')
        return None

    # Extração do valor massa carga e intensidades
    mz_array = np.array(spectrum['m/z array'], dtype=float)
    int_array = np.array(spectrum['intensity array'], dtype=float)

    # Remove picos abaixo do limiar de ruído
    threshold = noise_rmv_threshold * np.max(int_array)
    keep = int_array >= threshold
    mz_array = mz_array[keep]
    int_array = int_array[keep]
    n_peaks = len(mz_array)

    if n_peaks < min_num_peaks:
        if log:
            print(f'[{i}] Rejected after noise filtering: {n_peaks} peaks left')
        return None

    max_int = np.max(int_array)
    if max_int == 0:
        if log:
            print(f'[{i}] Zero max intensity before integration')
        return None
    int_array = int_array / max_int

    mz_array, int_array = spectra_integrator(mz_array, int_array, mass_error)

    n_peaks = len(mz_array)
    if n_peaks < min_num_peaks:
        if log:
            print(f'[{i}] Rejected after integration: {n_peaks} peaks left')
        return None

    # Ordenar os picos por intensidade
    order = np.argsort(int_array)[::-1]
    if n_peaks > max_num_peaks:
        order = order[:max_num_peaks]
    mz_array = mz_array[order]
    int_array = int_array[order]

    # Filtra picos que estão dentro do intervalo definido por mz_range_limits
    mz_min, mz_max = mz_vocabs[0], mz_vocabs[-1]
    in_range = (mz_array >= mz_min) & (mz_array <= mz_max)
    mz_array = mz_array[in_range]
    int_array = int_array[in_range]
    n_filtered_peaks = len(mz_array)

    if n_filtered_peaks < min_num_peaks or n_filtered_peaks < 0.9 * len(order):
        if log:
            print(f'[{i}] Rejected after mz range filtering: {n_filtered_peaks} peaks left')
        return None

    # voltar a ordenar os valores para o transformer conseguir pegar contexto
    order_by_mz = np.argsort(mz_array)
    mz_array = mz_array[order_by_mz]
    int_array = int_array[order_by_mz]

    # tokenizar os valores de m/z e do precursor
    tokenized_mz = [np.argmin(np.abs(mz - mz_vocabs)) for mz in mz_array]
    tokenized_precursor = np.argmin(np.abs(precursor_mz - mz_vocabs))

    int_sum = np.sum(int_array)
    if int_sum == 0:
        if log:
            print(f'[{i}] Zero intensities detected')
        return None
    int_array = int_array / int_sum

    training_tuple = (
        spectrum_id,
        tokenized_mz,
        tokenized_precursor,
        int_array,
    )

    return training_tuple


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
