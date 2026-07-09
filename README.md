# Transformer-Based-Models-for-Chemical-Fingerprint-Prediction

<!-- Tech Stack Badges (Estilo Shields.io) -->
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Logs%20%26%20Metrics-orange.svg)
![CLI](https://img.shields.io/badge/Interface-CLI%20%2F%20Automation-white.svg)

This repository contains the framework developed for my master's thesis in Bioinformatics at the University of Minho, focused on predicting untargeted molecular fingerprints from mass spectrometry data using Transformer-based models.

The purpose of this dissertation was to build and evaluate transformer models able to translate MS/MS spectra into useful molecular representations, namely ECFPs and MACCS fingerprints.

## 🧬 Framework & Pipeline Architecture

To achieve the project objectives, a data engineering workflow was implemented to process raw spectral data into tensors:

**Spectral Data Utilities:** A set of utilities was implemented to manipulate and extract information from .mgf files.

**Pre-processing Workflow:** Developed to filter, remove noise, and tokenise the raw spectra so that they could be processed by the models.


<img src="results/spectra_processing.png" alt="Overview of the data processing pipeline" width="500"/>


These sequences were then used to train two Transformer-based models, enabling the prediction of two different structural representations of compounds.

<img src="results/arquitetura.png" alt="Transformer Model Architecture" width="300"/>

## 📊 Experimental Results & Benchmarks

The experimental results demonstrated that the model trained to predict MACCS fingerprints achieved robust performance, obtaining an average Tanimoto score of 0.63.

In contrast, predicting ECFPs proved to be a far more complex challenge. Unlike MACCS fingerprints, which function as a fixed dictionary, ECFPs are generated through an iterative hashing process, in which specific bits do not always correspond to identical physical substructures in different compounds. However, despite the complexity of the task, the ECFPs model achieved a mean Tanimoto score of 0.33, a result that is highly competitive and at the level of current state-of-the-art for similar tasks.

<table>
<tr>
<td valign="top" width="50%">

### **MACCS Keys Model Performance**

| Metrics | Test Set |
| :--- | :---: |
| mPrecision | 0.46 |
| wPrecision | 0.70 |
| mRecall | 0.46 |
| wRecall | 0.83 |
| mF1 | 0.44 |
| wF1 | 0.75 |
| **Tanimoto (Jaccard Index)** | **0.63** |


</td>
<td valign="top" width="50%">

### **ECFPs Keys Model Performance**

| Metrics | Test Set |
| :--- |:--------:|
| mPrecision |   0.31   |
| wPrecision |   0.48   |
| mRecall |   0.19   |
| wRecall |   0.43   |
| mF1 |   0.22   |
| wF1 |   0.44   |
| **Tanimoto (Jaccard Index)** | **0.33** |

</td>
</tr>
</table>


## 📂 Repository Structure


The structure of the repository has been organized in a modular way to make it easier to navigate and understand the code.

```    
└── Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/
    ├── env.yml
    │
    ├── notebooks/     │
    │   
    ├── scripts/
    │   ├── production/ 
    │   └──development/
    │    
    └── src/ # Project source code
        ├── data/ 
        ├── models/ 
        ├── training/
        ├── config.py 
        └── utils.py
```



Much of the work that can be found in this repository was inspired by and adapted from the excellent repository (https://github.com/idslme/IDSL_MINT)