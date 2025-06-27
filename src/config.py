import numpy as np

# Document for storing global variables

mz_vocabs = np.arange(1.0, 5000.1, 0.1).tolist()

vocab_size = len(mz_vocabs)  # Variável que serve como token para o padding

min_num_peaks = 5

max_num_peaks = 431

max_seq_len = 1 + max_num_peaks  # Quantidade de picos (percentil 95%) + o percursor

d_model = 256

dropout_rate = 0.1
