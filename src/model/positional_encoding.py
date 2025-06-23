# File used

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    """
    Adds positional information to the input embeddings
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
