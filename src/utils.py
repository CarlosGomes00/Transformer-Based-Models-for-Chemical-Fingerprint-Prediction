# Generic functions that can be reused

import os

def path_check(mgf_data : str):
    """
    Checks if the path to the dataset has been found

    Parameters:
        mgf_data : str
            Path to the dataset to be used
    """

    if not os.path.exists(mgf_data):
        print(f"Erro: Não foi possivel encontrar o ficheiro {os.path.abspath(mgf_data)}")
    else:
        print("Ficheiro encontrado!")

    return
