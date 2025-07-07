import torch
import torch as nn


class FingerprintHead(nn.Module):

    """
    Class that applies the transfomer's final activation function, Sigmoid
    """

    def __init__(self, d_model, fingerprint_dim=2048):
        super().__init__()
        self.fc = nn.Linear(d_model, fingerprint_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
