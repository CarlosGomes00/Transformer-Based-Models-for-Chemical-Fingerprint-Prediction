from pyteomics import mgf

"""
O *pyteomics* é um package especializado para a análise de dados de MS
"""

# Primeiro,começamos por fazer load dos dados
mgf_data = "Data/cleaned_gnps_library.mgf"

with mgf.read(mgf_data) as spectra:
    for i, spectrum in enumerate(spectra):
        # Primeiro irei testar apenas para 3 espetros
        if i >= 3:
            break

        mz_values = spectrum["m/z array"]


