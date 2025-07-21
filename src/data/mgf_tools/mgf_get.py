from pyteomics import mgf
import pandas as pd


def mgf_get_spectra(mgf_data: str, num_spectra: int = None, spectrum_id: str = None):

    """
    Read all the information about the spectra in a .mgf file and return it as a dictionary
    It is mainly used as a reading base for the spectra plot function

    Parameters:
        mgf_data : str
            Path to the datasets to be used
        num_spectra : int
            Number of spectra info to be read. All by default
        spectrum_id : str
            Specific spectrum ID to fetch. Overrides num_spectra if provided.

    Returns:
        Dictionary of each spectrum
    """

    spectra = list(mgf.read(mgf_data, use_index=False))

    if len(spectra) == 0:
        print("Error reading .MGF file")
        return None

    if spectrum_id:
        for spec in spectra:
            if spec["params"].get("spectrum_id") == spectrum_id:
                return spec
        print(f"Spectrum ID '{spectrum_id}' not found.")
        return None

    if num_spectra is None:
        return spectra

    return spectra[:num_spectra] if num_spectra > 1 else spectra[0]


def mgf_get_smiles(spectra: list[dict], as_dataframe: bool = False):

    """
    Extract spectrum IDs and SMILES from a list of spectra.

    Parameters:
        spectra : list of dicts
           Output of mgf_get_spectra
        as_dataframe : bool
           If True, returns a pandas DataFrame instead of a dict

    Returns:
       dict or pd.DataFrame
           Dictionary with 'id' and 'smiles' lists, or a DataFrame
    """

    ids, smiles_list = [], []
    for spec in spectra:
        spec_id = spec["params"].get("spectrum_id")
        smiles = spec["params"].get("smiles")
        if spec_id and smiles:
            ids.append(spec_id)
            smiles_list.append(smiles)

    if as_dataframe:
        return pd.DataFrame({'spectrum_id': ids, 'smiles': smiles_list})
    else:
        return {'spectrum_id': ids, 'smiles': smiles_list}
