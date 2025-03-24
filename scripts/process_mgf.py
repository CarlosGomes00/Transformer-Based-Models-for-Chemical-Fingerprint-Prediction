from pyteomics import mgf
import os

"""
O *pyteomics* é um package para a análise de dados de MS
"""


mgf_data = r"/Users/carla/PycharmProjects/Mestrado/Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/data/raw/cleaned_gnps_library.mgf"


if not os.path.exists(mgf_data):
    print(f"❌ Erro: Não foi possivel encontrar o ficheiro {os.path.abspath(mgf_data)}")
else:
    print("✅ Ficheiro encontrado!")



with mgf.read(mgf_data) as spectra:
    for i, spectrum in enumerate(spectra):
        # Primeiro irei testar apenas para 3 espetros
        if i >= 3:
            break
        print(spectrum["params"])
