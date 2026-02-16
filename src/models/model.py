import torch
import torch.nn as nn
from src.models.components.embeddings import PeakEmbedding
from src.models.components.positional_encoding import PositionalEncoding
#from src.models.components.pooling import mean_pooling
from src.models.components.fingerprint_head import FingerprintHead, FingerprintHeadLogits


class EncoderTransformer(nn.Module):

    """
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

    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout_rate, max_seq_len, fingerprint_dim=2048,
                 head_type='logits', batch_norm=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.fingerprint_dim = fingerprint_dim
        self.head_type = head_type
        self.batch_norm = batch_norm

        self.peak_embedding = PeakEmbedding(vocab_size, d_model, dropout_rate, max_norm=2)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #self.pooling = mean_pooling

        if head_type == 'logits':
            self.fingerprint_head = FingerprintHeadLogits(d_model, fingerprint_dim, batch_norm)
        else:
            self.fingerprint_head = FingerprintHead(d_model, fingerprint_dim, batch_norm)

    def forward(self, mz_batch, int_batch, attention_mask):

        """
        Forward pass through the transformer model

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

        x = self.peak_embedding(mz_batch, int_batch)

        src_key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        cls_token = x[:, 0, :]

        output = self.fingerprint_head(cls_token)

        return output
