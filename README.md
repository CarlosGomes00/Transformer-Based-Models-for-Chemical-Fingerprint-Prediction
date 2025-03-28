# Transformer-Based-Models-for-Chemical-Fingerprint-Prediction

This repository contains the code and work developed for my master's thesis in Bioinformatics. 

## Repository structure
The structure of the repository has been organized in a modular way to make it easier to navigate and understand the code. Each component of the workflow is organized in the following folders:

```
└── Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/
    ├── data/                       # Stores raw and processed data (It may be hidden in .gitignore)
    │   ├── raw/                          # .mgf files
    │   ├── processed/                    # Extracted data (CSV, DataFrame, JPG, etc.)
    │   └── fingerprints/                 # Generated fingerprints
    ├── notebooks/                  # Jupyter notebooks for exploring and visualizing data/results
    ├── src/                        # The project's main source code
    │   └── utils.py                      # Auxiliary and reusable functions
    ├── results/                    # Results achieved
    ├── scripts/                    # Scripts for quick execution
    │   ├── process_mgf.py                # Features for extracting relevant information from spectra
    │   └── plot_mgf.py                   # Different functions to plot the spectra
    ├── README.md
    └── .gitignore
```
