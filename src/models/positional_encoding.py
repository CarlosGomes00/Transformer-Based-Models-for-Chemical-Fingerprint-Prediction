import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    """
    Adds sinusoidal positional information to the input embeddings

    Parameters:
        d_model : int
            The dimension of the input embeddings
        max_seq_len : int
            Size of the tokens in a spectrum (including the precursor)
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout_rate: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Add positional encoding to input embeddings

        Parameters:
            x : torch.Tensor
                Input embeddings of shape [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor
                Input embeddings with positional encoding added, followed by dropout
        """

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
