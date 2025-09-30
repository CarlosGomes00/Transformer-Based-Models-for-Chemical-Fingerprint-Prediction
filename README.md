# Transformer-Based-Models-for-Chemical-Fingerprint-Prediction

This repository contains the code developed for my master's thesis in Bioinformatics, focused on predicting untargeted molecular fingerprints from mass spectrometry data using Transformers-based models.

## Features
* **End-to-End Machine Learning Pipeline:** Modular scripts for pre-processing, training, and evaluation.
* **Spectral Data Processing:** Tools for reading, filtering, and processing mass spectrometry files (`.mgf`).
* **Fingerprint generation:** Generation of Morgan Fingerprints from SMILES.

## Repository structure
The structure of the repository has been organized in a modular way to make it easier to navigate and understand the code.

```    
└── Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/
    ├── env.yml
    │
    ├── notebooks/ # Notebooks for data exploration and tutorials
    │
    ├── outputs/ # Model logs, checkpoints and evaluations
    │   
    ├── scripts/
    │   └── production/ # Main pipeline scripts
    └── src/ # Project source code
        ├── data/ # Data manipulation modules
        ├── models/ # Definition of models and wrappers
        ├── training/
        ├── config.py # Global settings (e.g. paths)
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

## Main Pipeline

The pipeline is designed to be executed in three sequential steps from the command line. The `seed` is used as a key to link all steps, to guarantee consistency.

#### **1. Data Pre-processing and Splitting**

This script processes the `.mgf` file, generates the fingerprints, and creates the training, validation, and test splits, saving the artifacts in a specific folder according to the provided seed.

```
python -m scripts.production.01_preprocess_and_split --seed 1
```


#### **2. Model training**

This script loads the corresponding `seed` data, instantiates the model, and executes the training according to the given parameters.

```
python -m scripts.production.02_train_model --seed 1 --fast_dev_run
```


#### **3. Model evaluation**

This script loads the checkpoint from a training session and evaluates its performance on the test set.

```
python -m scripts.production.03_evaluation --seed 1 --checkpoint_path path.to.the.checkpoint
```

**All scripts have additional flags that can and should be changed for greater customisation.**


## Notebooks and API

* The `Transformer` class in `src/models/Transformer.py` serves as a high-level API for interacting with the model.
* If you prefer to use class methods instead of scripts, you can see how by looking at the examples in the notebooks. 


Much of the work that can be found in this repository was inspired by and adapted from the excellent repository (https://github.com/idslme/IDSL_MINT)