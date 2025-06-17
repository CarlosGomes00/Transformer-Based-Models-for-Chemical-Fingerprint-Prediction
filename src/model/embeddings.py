# File used to store the code for the creation of embeddigns

import torch
import torch.nn as nn
from src.config import mz_vocabs

vocab_size = len(mz_vocabs)
max_peaks_per_spectrum = 431
max_seq_len = 1 + max_peaks_per_spectrum  # Quantidade de picos (percentil 95%) + o percursor
d_model = 256
dropout_rate = 0.1


class PeakEmbedding(nn.Module):

    """
    Layer to create combined m/z and Intensity embeddings for each peak

    Each peak (m/z_token, intensity) is transformed into a vector of d_model dimensions
    """

    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.mz_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.linear_combine = nn.Linear(d_model + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tokenized_mz: torch.Tensor, intensities: torch.Tensor) -> torch.Tensor:

        mz_emb = self.mz_embedding(tokenized_mz)
        intensities_expanded = intensities.unsqueeze(-1)
        combined_features = torch.cat((mz_emb, intensities_expanded), dim=-1)
        combined_emb = self.linear_combine(combined_features)

        return self.dropout(combined_emb)


class PrecursorEmbedding(nn.Module):
    """
    Layer to create precursor embeddings

    Each peak (m/z_token, intensity) is transformed into a vector of d_model dimensions
    """
    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.precursor_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tokenized_precursor: torch.Tensor) -> torch.Tensor:
        precursor_emb = self.precursor_embedding(tokenized_precursor)

        return self.dropout(precursor_emb)
