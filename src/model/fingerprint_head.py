import torch
import torch.nn as nn


class FingerprintHead(nn.Module):

    """
    Final layer that maps aggregated spectral features to chemical fingerprints
    This layer transforms the pooled representation from the transformer encoder into binary probability
    predictions for each bit of a fingerprint

    Parameters:
        d_model : int
            Dimension of input features from the pooling layer
        fingerprint_dim : int, 2048 by default
            Size of output fingerprint in bits
    """

    def __init__(self, d_model, fingerprint_dim=2048):
        super().__init__()
        self.fc = nn.Linear(d_model, fingerprint_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
