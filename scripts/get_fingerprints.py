import numpy as np
import pandas as pd
import rdkit.Chem.rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, PandasTools, MACCSkeys, AtomPairs, rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem.rdmolops import PatternFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
import warnings
warnings.filterwarnings('ignore')


def smiles_to_fingerprint(smiles: pd.DataFrame, radius: int = 1, nbits: int = 2048) -> pd.DataFrame:

    """
    Converts SMILES strings into Morgan (ECFP-like) fingerprints using RDKit

    Parameters:
        smiles : pd.DataFrame
            DataFrame containing a 'smiles' column
        radius : int
            Radius for the Morgan fingerprint, 1 by default
        nbits : int
            Number of bits in the fingerprint, 2048 by default

    Returns:
        pd.DataFrame
            DataFrame with the original info + Morgan fingerprints bits

    """

    PandasTools.AddMoleculeColumnToFrame(smiles, 'smiles', 'molecule')

    df_mf = []

    for mol in smiles['molecule']:
        mf_bitvector = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)

        DataStructs.ConvertToNumpyArray(mf_bitvector, arr)
        df_mf.append(arr)

    fingerprints_df = pd.concat([smiles, pd.DataFrame(df_mf)], axis=1)

    return fingerprints_df
