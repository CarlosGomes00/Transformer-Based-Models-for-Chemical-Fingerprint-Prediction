# Generic functions that can be reused
import os
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
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
        log: bool,
        allowed_spectral_entropy: bool = True):

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
        allowed_spectral_entropy : bool
            If True, calculates spectral entropy as precursor intensity following
            If False, uses fixed value (2.0) as fallback
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

    if allowed_spectral_entropy:
        spectrum_pairs = np.column_stack((mz_array, int_array))

        spectral_entropy, processed_spectrum = spectral_entropy_calculator(spectrum_pairs)

        int_array = processed_spectrum[:, 1]
        precursor_int = spectral_entropy

    else:
        precursor_int = 2.0

    training_tuple = (
        spectrum_id,
        tokenized_precursor,
        tokenized_mz,
        precursor_int,
        int_array,
    )

    return training_tuple


def mgf_deconvoluter(mgf_data, mz_vocabs, min_num_peaks, max_num_peaks, noise_rmv_threshold, mass_error,
                     log, allowed_spectral_entropy=True, plot=False):

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
        allowed_spectral_entropy : bool
            If True, calculates spectral entropy as precursor intensity following
            If False, uses fixed value (2.0) as fallback
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
            allowed_spectral_entropy=allowed_spectral_entropy,
            log=log,
        )

        if result is not None:
            processed_spectra.append(result)

            if plot:
                spectrum_id, tokenized_precursor, tokenized_mz, precursor_int, intensities = result
                print(f"\nID: {spectrum_id}")
                print(f"Tokenized m/z: {tokenized_mz}")
                print(f"Tokenized precursor: {tokenized_precursor}")
                print(f"Precursor intensity (Entropy): {precursor_int}")
                print(f"Normalized intensities: {intensities}")

                plt.figure(figsize=(10, 5))
                plt.bar(tokenized_mz, intensities, width=5, color='royalblue')
                plt.title(f"Mass Spectrum {spectrum_id}")
                plt.xlabel("Tokenized m/z")
                plt.ylabel("Normalized Intensity")
                plt.tight_layout()
                plt.show()

    return processed_spectra


def spectral_entropy_calculator(spectra, allowed_weighted_spectral_entropy: bool = True):

    """
    Calculate spectral entropy for mass spectrometry data following IDSL_Mint methodology

    Parameters:
        spectra : array-like
            Input spectral data containing m/z and intensity pairs
        allowed_weighted_spectral_entropy : bool, default = True
            Whether to apply weighted spectral entropy transformation

    Returns:
        tuple[float, np.ndarray]
    """

    spectra = np.array(spectra).reshape(-1, 2)

    if np.sum(spectra[:, 1]) == 0:
        return 0.0, spectra

    spectra[:, 1] = spectra[:, 1] / np.sum(spectra[:, 1])

    spectral_entropy = -np.sum(spectra[:, 1] * np.log(spectra[:, 1]))

    if allowed_weighted_spectral_entropy:

        if spectral_entropy < 3:
            weights = 0.25 + spectral_entropy * 0.25
            spectra[:, 1] = np.power(spectra[:, 1], weights)

            spectra[:, 1] = spectra[:, 1]/np.sum(spectra[:, 1])

            spectral_entropy = -np.sum(spectra[:, 1] * np.log(spectra[:, 1]))

    return spectral_entropy, spectra


def tensor_to_bitvect(t: torch.Tensor) -> ExplicitBitVect:

    """
    Converts a 1-D tensor (0/1) to RDKit ExplicitBitVect to calculate Tanimoto Simularity
    """

    arr = t.cpu().numpy().astype(np.uint8)
    bv = ExplicitBitVect(len(arr))
    on_bits = arr.nonzero()[0].tolist()
    for i in on_bits:
        bv.SetBit(i)
    return bv


def generate_data_stats(y_train: np.ndarray, y_test: np.ndarray, y_val: np.ndarray = None):
    """
    Parameters
    ----------
    y_train : np.ndarray
        Labels of the train set
    y_test : np.ndarray
        Labels of the test set
    y_val : np.ndarray, optional
        Labels of the validation set, by default None

    Returns
    -------
    Tuple[pd.DataFrame, Any]
        DataFrame with the stats of the split, styled table
    """
    y_test_sum = np.sum(y_test, axis=0)
    y_train_sum = np.sum(y_train, axis=0)

    sum_of_all = pd.DataFrame([y_train_sum, y_test_sum], index=["train", "test"])

    if y_val is not None:
        y_val_sum = np.sum(y_val, axis=0)
        sum_of_all = pd.DataFrame([y_train_sum, y_test_sum, y_val_sum], index=["train", "test", "validation"])
        sum_of_all.loc['Validation relative split', :] = sum_of_all.loc['validation', :] / (
                    sum_of_all.loc['train', :] + sum_of_all.loc['test', :] + sum_of_all.loc['validation', :]) * 100
        sum_of_all.loc['Test relative split', :] = sum_of_all.loc['test', :] / (
                    sum_of_all.loc['train', :] + sum_of_all.loc['test', :] + sum_of_all.loc['validation', :]) * 100
        sum_of_all.loc['Train relative split', :] = sum_of_all.loc['train', :] / (
                    sum_of_all.loc['train', :] + sum_of_all.loc['test', :] + sum_of_all.loc['validation', :]) * 100

    else:
        sum_of_all.loc['Test relative split', :] = sum_of_all.loc['test', :] / (
                    sum_of_all.loc['train', :] + sum_of_all.loc['test', :]) * 100
        sum_of_all.loc['Train relative split', :] = sum_of_all.loc['train', :] / (
                    sum_of_all.loc['train', :] + sum_of_all.loc['test', :]) * 100

    df = pd.melt(sum_of_all.T.reset_index(), id_vars=['index']).rename(
        columns={'index': 'EC', 'value': 'Percentage of data'})
    if y_val is not None:
        df = df[(df["variable"] != "train") & (df["variable"] != "validation") & (df["variable"] != "test")]
    else:
        df = df[(df["variable"] != "train") & (df["variable"] != "test")]

    df1 = sum_of_all.loc['Test relative split', :].describe()
    df2 = sum_of_all.loc['Train relative split', :].describe()
    if y_val is not None:
        df3 = sum_of_all.loc['Validation relative split', :].describe()
        stats_table = pd.concat([df1, df2, df3], axis=1)
    else:
        stats_table = pd.concat([df1, df2], axis=1)

    stats_table.drop(['count'], inplace=True)
    table_styled = stats_table.style.background_gradient(cmap="YlGn")

    return df, table_styled


def save_as_png(df, filename):

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
