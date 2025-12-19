import torch
import torch.nn as nn


class BasicFingerprintHead(nn.Module):

    """
    Final layer that maps aggregated spectral features to chemical fingerprints
    This layer transforms the pooled representation from the transformer encoder into binary probability
    predictions for each bit of the fingerprint

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


class BasicFingerprintHeadLogits(nn.Module):

    """
    Final layer for use with nn.BCEWithLogitsLoss
    Returns the raw logits without a final activation function

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
        return self.fc(x)


class FingerprintHead(nn.Module):

    """
    Final layer that maps aggregated spectral features to chemical fingerprints
    This layer transforms the pooled representation from the transformer encoder into binary probability
    predictions for each bit of the fingerprint

    Parameters:
        d_model : int
            Dimension of input features from the pooling layer
        fingerprint_dim : int, 2048 by default
            Size of output fingerprint in bits
    """

    def __init__(self, d_model, fingerprint_dim=2048, batch_norm=True):
        super().__init__()

        middle_value = int(fingerprint_dim / 2)
        layers = []

        layers.append(nn.Linear(d_model, middle_value))

        if batch_norm:
            layers.append(nn.BatchNorm1d(middle_value))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(middle_value, fingerprint_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class FingerprintHeadLogits(nn.Module):

    """
    Final layer for use with nn.BCEWithLogitsLoss
    Returns the raw logits without a final activation function

    Parameters:
        d_model : int
            Dimension of input features from the pooling layer
        fingerprint_dim : int, 2048 by default
            Size of output fingerprint in bits
    """

    def __init__(self, d_model, fingerprint_dim=2048, batch_norm=True):
        super().__init__()

        middle_value = int(fingerprint_dim / 2)
        layers = []

        layers.append(nn.Linear(d_model, middle_value))

        if batch_norm:
            layers.append(nn.BatchNorm1d(middle_value))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(middle_value, fingerprint_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
