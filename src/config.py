import numpy as np

# Document for storing global variables

mz_vocabs = np.arange(1.0, 5000.1, 0.1).tolist()

vocab_size = len(mz_vocabs)

max_peaks_per_spectrum = 431

max_seq_len = 1 + max_peaks_per_spectrum  # Quantidade de picos (percentil 95%) + o percursor

d_model = 256

dropout_rate = 0.1
