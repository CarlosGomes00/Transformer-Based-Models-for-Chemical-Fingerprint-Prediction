import torch
import torch.nn as nn
from src.models.components.embeddings import PeakEmbedding, PrecursorEmbeddingN
from src.models.components.positional_encoding import PositionalEncoding
from src.models.components.pooling import mean_pooling
from src.models.components.fingerprint_head import FingerprintHead, FingerprintHeadLogits


class EncoderTransformer(nn.Module):

    """
    Transformer encoder models for predicting chemical fingerprints from mass spectra

    Parameters:
        vocab_size : int
            Size of the m/z vocabulary for token embeddings
        d_model : int
            Dimension of embeddings
        nhead : int
            Number of attention heads in each transformer layer
        num_layers : int
            Number of transformer encoder layers to stack
        dropout_rate : float
            Dropout probability applied throughout the models
        fingerprint_dim : int, 2048 by default
            Size of output fingerprint
        max_seq_len : int
            Maximum sequence length for positional encoding pre-computation
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout_rate, max_seq_len, fingerprint_dim=2048):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.fingerprint_dim = fingerprint_dim

        self.peak_embedding = PeakEmbedding(vocab_size, d_model, dropout_rate)
        self.precursor_embedding = PrecursorEmbeddingN(vocab_size, d_model, dropout_rate)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = mean_pooling
        self.fingerprint_head = FingerprintHeadLogits(d_model, fingerprint_dim)

    def forward(self, mz_batch, int_batch, attention_mask):

        """
        Forward pass through the transformer models

        Parameters:
            mz_batch : torch.Tensor
                Tokenized m/z values with shape [batch_size, max_seq_len]
            int_batch : torch.Tensor
                Normalized intensity values with shape [batch_size, max_seq_len]
            attention_mask : torch.Tensor
                Binary mask with shape [batch_size, max_seq_len]

        Returns:
            torch.Tensor
                Predicted fingerprint probabilities with shape [batch_size, fingerprint_dim].

        """

        precursor_tokens = mz_batch[:, 0:1]
        precursor_emb = self.precursor_embedding(precursor_tokens)

        peak_emb = self.peak_embedding(mz_batch[:, 1:], int_batch[:, 1:])

        x = torch.cat([precursor_emb, peak_emb], dim=1)

        x = self.positional_encoding(x)

        src_key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        pooled = self.pooling(x, attention_mask)

        output = self.fingerprint_head(pooled)

        return output
