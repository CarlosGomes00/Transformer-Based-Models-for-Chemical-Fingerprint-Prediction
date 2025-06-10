# Transformer-Based-Models-for-Chemical-Fingerprint-Prediction

This repository contains the code and work developed for my master's thesis in Bioinformatics. 

## Repository structure
The structure of the repository has been organized in a modular way to make it easier to navigate and understand the code.

```    
└── Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/
    ├── env.yml
    ├── notebooks/
    │   ├── outputs/
    │   ├── fingerprints_test.ipynb
    │   └── mgf_test.ipynb
    ├── scripts/
    │   └── __init__.py
    └── src/
        ├── fingerprint_tools/
        │   ├── __init__.py
        │   └── fingerprint_generator.py
        ├── mgf_tools/
        │   ├── __init__.py
        │   ├── mgf_checks.py
        │   ├── mgf_get.py
        │   ├── mgf_plots.py
        │   └── mgf_readers.py
        ├── transformer/
        │   └── __init__.py
        └── utils.py
```

To ensure that all the project's dependencies are installed correctly, we recommend using a Conda environment. 
You can easily create the environment with the project's dependencies from the env.yml file included in this repository.

## Steps to set up the environment:

1. Make sure you have Conda installed. If you don't have Conda installed, download and install Miniconda or Anaconda.

2. Clone the repository to your local machine: If you haven't cloned the repository yet, use the following command to clone it:

```
git clone https://github.com/CarlosGomes00/Transformer-Based-Models-for-Chemical-Fingerprint-Prediction
```

3. Create the Conda environment from the env.yml file: Navigate to the cloned project directory and run the following
command to create the Conda environment with the required dependencies:

```
conda env create -f env.yml 
```

4. Activate the Conda environment: Once the environment is created, activate it with the following command:

```
conda activate Transformer-Based-Models-for-Chemical-Fingerprint-Prediction
```

5. Verify that the dependencies & environment were installed correctly.
You can check if the environment was created successfully and if all dependencies were installed by running:

```
conda list
conda env list
```

With the features provided in this repository, you can:
- Reading and parsing *.mgf* files
- Visualisation of individual and multiple spectra
- Features that allow an exploratory analysis of the .mgf file 
- SMILES extraction from *.mgf* files
- Generation of Morgan Fingerprints

To understand how, you can check out the mini tutorials that are available on the notebooks.


Much of the work that can be found in this repository has been adapted from IDSL_MINT (https://github.com/idslme/IDSL_MINT)