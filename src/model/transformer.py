import torch
import torch.nn as nn
from src.model.embeddings import PeakEmbedding, PrecursorEmbeddingN
from src.model.positional_encoding import PositionalEncoding
from src.model.pooling import mean_pooling
from src.model.fingerprint_head import FingerprintHead


class EncoderTransformer(nn.Module):

    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout_rate, fingerprint_dim=2048, max_seq_len=432):
        super().__init__()

        self.peak_embedding = PeakEmbedding(vocab_size, d_model, dropout_rate)
        self.precursor_embedding = PrecursorEmbeddingN(vocab_size, d_model, dropout_rate)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = mean_pooling
        self.fingerprint_head = FingerprintHead(d_model, fingerprint_dim)

    def forward(self, mz_batch, int_batch, attention_mask):

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
