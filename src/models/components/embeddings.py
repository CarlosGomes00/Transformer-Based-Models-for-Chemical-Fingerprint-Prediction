import torch
import torch.nn as nn
import math


class PeakEmbedding(nn.Module):

    def __init__(self, vocab_size:int, d_model:int, dropout_rate:float, max_norm:int=2):
        super().__init__()

        self.mz_embedding = nn.Embedding(vocab_size + 1, d_model, max_norm=max_norm)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model 

        idx1_local = torch.arange(start=1, end=self.d_model, step=2)
        idx2_local = torch.arange(start=0, end=self.d_model, step=2)

        odd = 10000.0 ** (2 * idx1_local / self.d_model)
        even = 10000.0 ** (2 * idx2_local / self.d_model)

        self.register_buffer('idx1', idx1_local)
        self.register_buffer('idx2', idx2_local)
        self.register_buffer('odd', odd)
        self.register_buffer('even', even)

    def forward(self, mz_batch: torch.Tensor, int_batch: torch.Tensor) -> torch.Tensor:

        mz_embed = self.mz_embedding(mz_batch) * math.sqrt(self.d_model)
        int_expanded = int_batch.unsqueeze(-1)
       
        phase_odd = (self.idx1 * int_expanded) / self.odd
        phase_even = (self.idx2 * int_expanded) / self.even

        pe = torch.zeros_like(mz_embed)
        
        pe[:, :, self.idx1] = torch.cos(phase_odd)
        pe[:, :, self.idx2] = torch.sin(phase_even)

        embeddings = mz_embed + pe

        return self.dropout(embeddings)


"""
class PeakEmbedding(nn.Module):


    Creates combined m/z and intensity embeddings for fragment peaks in mass spectra

    Each peak (m/z_token, intensity) is transformed into a vector of d_model dimensions

    Parameters:
        vocab_size : int
            Size of the m/z vocabulary for token embeddings
        d_model : int
            Dimension of the output embeddings
        dropout_rate : float
            Dropout probability applied after embedding combination
 

    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.mz_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.linear_combine = nn.Linear(d_model + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, mz_batch: torch.Tensor, int_batch: torch.Tensor) -> torch.Tensor:

 
        Forward pass to create combined peak embeddings

        Parameters:
            mz_batch : torch.Tensor
                Tokenized m/z values with shape [batch_size, seq_len]
            int_batch : torch.Tensor
                Normalized intensity values with shape [batch_size, seq_len]


        mz_emb = self.mz_embedding(mz_batch)
        intensities_expanded = int_batch.unsqueeze(-1)
        combined_features = torch.cat((mz_emb, intensities_expanded), dim=-1)
        combined_emb = self.linear_combine(combined_features)

        return self.dropout(combined_emb)


class PrecursorEmbeddingN(nn.Module):


    Creates embeddings for precursor ion

    Parameters:
        vocab_size : int
            Size of the m/z vocabulary for token embeddings
        d_model : int
            Dimension of the output embeddings
        dropout_rate : float
            Dropout probability applied after embedding lookup


    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.precursor_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, precursor_tokens: torch.Tensor) -> torch.Tensor:


        Forward pass to create precursor embeddings

        Parameters:
            precursor_tokens : torch.Tensor
                Tokenized precursor m/z values with shape [batch_size, 1]

        Returns:
            torch.Tensor
                Precursor embeddings with shape [batch_size, 1, d_model]


        precursor_emb = self.precursor_embedding(precursor_tokens)

        return self.dropout(precursor_emb)

"""
# LEGACY: Função antiga -> utilizar PrecursorEmbeddingN
class PrecursorEmbedding(nn.Module):

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.precursor_embedding = nn.Parameter(torch.randn(1, d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch_size: int) -> torch.Tensor:
        emb = self.precursor_embedding.expand(batch_size, -1, -1)
        return self.dropout(emb)
    