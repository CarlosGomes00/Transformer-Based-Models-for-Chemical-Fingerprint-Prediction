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
        target_type : str
            Type of fingerprint to predict ('ECFP4' or 'MACCS')
    """

    def __init__(self, d_model, target_type):
        super().__init__()
        
        if target_type == 'ECFP4':
            fingerprint_dim = 2048
        elif target_type == 'MACCS':
            fingerprint_dim = 167
        else:
            raise ValueError(f"Wrong target type '{target_type}' on the classification head.")
        
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
        target_type : str
            Type of fingerprint to predict ('ECFP4' or 'MACCS')
    """

    def __init__(self, d_model, target_type):
        super().__init__()

        if target_type == 'ECFP4':
            fingerprint_dim = 2048
        elif target_type == 'MACCS':
            fingerprint_dim = 167
        else:
            raise ValueError(f"Wrong target type '{target_type}' on the classification head.")

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
        target_type : str
            Type of fingerprint to predict ('ECFP4' or 'MACCS')
    """

    def __init__(self, d_model, target_type, batch_norm=True):
        super().__init__()

        if target_type == 'ECFP4':
            fingerprint_dim = 2048
            middle_value = int(fingerprint_dim / 2)
        elif target_type == 'MACCS':
            fingerprint_dim = 167
            middle_value = d_model
        else:
            raise ValueError(f"Wrong target type '{target_type}' on the classification head.")

        layers = []

        layers.append(nn.Linear(d_model, middle_value))

        if batch_norm:
            layers.append(nn.BatchNorm1d(middle_value))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
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
        target_type : str
            Type of fingerprint to predict ('ECFP4' or 'MACCS')s
    """

    def __init__(self, d_model, target_type, batch_norm=True):
        super().__init__()

        if target_type == 'ECFP4':
            fingerprint_dim = 2048
            middle_value = int(fingerprint_dim / 2)
        elif target_type == 'MACCS':
            fingerprint_dim = 167
            middle_value = d_model
        else:
            raise ValueError(f"Wrong target type '{target_type}' on the classification head.")
        layers = []

        layers.append(nn.Linear(d_model, middle_value))

        if batch_norm:
            layers.append(nn.BatchNorm1d(middle_value))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(middle_value, fingerprint_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
