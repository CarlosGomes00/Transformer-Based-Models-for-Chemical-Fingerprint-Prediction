import torch
import torch.nn as nn


class PeakEmbedding(nn.Module):

    """
    Creates combined m/z and intensity embeddings for fragment peaks in mass spectra

    Each peak (m/z_token, intensity) is transformed into a vector of d_model dimensions

    Parameters:
        vocab_size : int
            Size of the m/z vocabulary for token embeddings
        d_model : int
            Dimension of the output embeddings
        dropout_rate : float
            Dropout probability applied after embedding combination
    """

    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.mz_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.linear_combine = nn.Linear(d_model + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, mz_batch: torch.Tensor, int_batch: torch.Tensor) -> torch.Tensor:

        """
        Forward pass to create combined peak embeddings

        Parameters:
            mz_batch : torch.Tensor
                Tokenized m/z values with shape [batch_size, seq_len]
            int_batch : torch.Tensor
                Normalized intensity values with shape [batch_size, seq_len]
        """

        mz_emb = self.mz_embedding(mz_batch)
        intensities_expanded = int_batch.unsqueeze(-1)
        combined_features = torch.cat((mz_emb, intensities_expanded), dim=-1)
        combined_emb = self.linear_combine(combined_features)

        return self.dropout(combined_emb)


class PrecursorEmbeddingN(nn.Module):

    """
    Creates embeddings for precursor ion

    Parameters:
        vocab_size : int
            Size of the m/z vocabulary for token embeddings
        d_model : int
            Dimension of the output embeddings
        dropout_rate : float
            Dropout probability applied after embedding lookup
    """

    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float):
        super().__init__()

        self.precursor_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, precursor_tokens: torch.Tensor) -> torch.Tensor:

        """
        Forward pass to create precursor embeddings

        Parameters:
            precursor_tokens : torch.Tensor
                Tokenized precursor m/z values with shape [batch_size, 1]

        Returns:
            torch.Tensor
                Precursor embeddings with shape [batch_size, 1, d_model]
        """

        precursor_emb = self.precursor_embedding(precursor_tokens)

        return self.dropout(precursor_emb)


# LEGACY: Função antiga -> utilizar PrecursorEmbeddingN
class PrecursorEmbedding(nn.Module):

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.precursor_embedding = nn.Parameter(torch.randn(1, d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch_size: int) -> torch.Tensor:
        emb = self.precursor_embedding.expand(batch_size, -1, -1)
        return self.dropout(emb)
    