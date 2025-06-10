import numpy as np
import pandas as pd
import rdkit.Chem.rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from deepmol.datasets import SmilesDataset
from deepmol.compound_featurization import MorganFingerprint


import warnings
warnings.filterwarnings('ignore')

# LEGACY: Função antiga, utilizar a função que usa o DeepMol (smiles_to_fingerprint)
def smiles_to_fingerprint_(smiles: pd.DataFrame, radius: int = 1, nbits: int = 2048, save: bool = False,
                          save_path: str = "fingerprints.csv") -> pd.DataFrame:

    '''
    Converts SMILES strings into Morgan (ECFP-like) fingerprints using RDKit

    Parameters:
        smiles : pd.DataFrame
            DataFrame containing a 'smiles' column
        radius : int
            Radius for the Morgan fingerprint, 1 by default
        nbits : int
            Number of bits in the fingerprint, 2048 by default
        save : bool
            Whether to save the DataFrame with fingerprints to a CSV, False by default
        save_path : str
            Path where to save the DataFrame

    Returns:
        pd.DataFrame
            DataFrame with the original info + Morgan fingerprints bits
    '''


    if "smiles" not in smiles.columns:
        raise KeyError("The 'smiles' column was not found in DataFrame")
    if radius < 0:
        raise ValueError("Radius must be greater than or equal to 0")
    if nbits <= 0:
        raise ValueError("nBits must be greater than 0")

    PandasTools.AddMoleculeColumnToFrame(smiles, "smiles", "molecule")

    df_mf = []

    for mol in smiles["molecule"]:
        mf_bitvector = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)

        DataStructs.ConvertToNumpyArray(mf_bitvector, arr)
        df_mf.append(arr)

    fingerprints_df = pd.concat([smiles, pd.DataFrame(df_mf)], axis=1)

    if save:
        fingerprints_df.to_csv(save_path, index=False)
        print(f"Fingerprints saved in {save_path}")

    return fingerprints_df


def smiles_to_fingerprint(smiles_data, ids: list = None, n_jobs: int = 10, return_df: bool = False):

    """
    Generates Morgan fingerprints from SMILES using DeepMol

    Parameters:
        smiles_data : pd.DataFrame, list
            DataFrame with ‘smiles’ and ‘spectrum_id’ or list of SMILES
        ids : list
            List of matching IDs if ‘smiles_data’ is a list
        n_jobs : int
            Number of parallel processes to generate fingerprints
        return_df : bool
            If True, it also returns a DataFrame with the fingerprints

    Returns:
        datasets : SmilesDataset
            A DeepMol datasets with the generated fingerprints
        df : pd.Dataframe
             DataFrame with spectrum_id and fingerprint bits if return_df is True
    """

    if isinstance(smiles_data, pd.DataFrame):
        if "smiles" not in smiles_data.columns or "spectrum_id" not in smiles_data.columns:
            raise ValueError('DataFrame needs to have a smiles and spectrum_id column')
        smiles = smiles_data["smiles"].tolist()
        ids = smiles_data["spectrum_id"].tolist()

    elif isinstance(smiles_data, list):
        if ids is None:
            raise ValueError("If you use a list of SMILES, provide a list with the corresponding IDs as well")
        smiles = smiles_data

    else:
        raise TypeError("Provide a DataFrame or two lists, one with the SMILES and the other with their corresponding IDs")

    dataset = SmilesDataset(smiles, ids=ids)

    MorganFingerprint(n_jobs=n_jobs).featurize(dataset, inplace=True)

    if return_df:
        df = pd.DataFrame(dataset.X, dtype=int)
        df.insert(0, "spectrum_id", ids)
        return dataset, df

    return dataset

