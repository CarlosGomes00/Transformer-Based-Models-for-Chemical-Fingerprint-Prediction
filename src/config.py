import numpy as np

# Document for storing global variables

# Paths
#mgf_path = r'/home/cgomes/Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/dataset/raw/cleaned_gnps_library.mgf'
mgf_path = r"/Users/carla/PycharmProjects/Mestrado/Transformer-Based-Models-for-Chemical-Fingerprint-Prediction/datasets/raw/cleaned_gnps_library.mgf"

# Fixed/default peaks related params
noise_rmv_threshold = 0.01
min_num_peaks = 5
max_num_peaks = 431  # Default value, usually overwritten
max_seq_len = 1 + max_num_peaks  # max_num_peaks + percursor
mass_error = 0.01


# M/z vocabs
mz_vocabs = np.arange(1.0, 5000.1, 0.1).tolist() # Default value, usually overwritten
vocab_size = len(mz_vocabs)  # Variável que serve como token para o padding


# Model related default values
d_model = 128
num_layers = 4
nhead = 4
dropout_rate = 0.1
learning_rate = 0.001
weight_decay = 1e-4   # Padrão para regularização L2
morgan_default_dim = 2048
