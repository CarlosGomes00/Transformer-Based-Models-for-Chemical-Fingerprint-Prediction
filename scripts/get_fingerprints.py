import numpy as np
import pandas as pd
import rdkit.Chem.rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit import DataStructs

import warnings
warnings.filterwarnings('ignore')


def smiles_to_fingerprint(smiles: pd.DataFrame, radius: int = 1, nbits: int = 2048, save: bool = False,
                          save_path: str = "fingerprints.csv") -> pd.DataFrame:

    """
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

    """

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
