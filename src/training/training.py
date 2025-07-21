import torch.nn as nn
import torch.optim as optim
from config import learning_rate, weight_decay


def training_setup(model):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    return criterion, optimizer


def train_step(model, batch, criterion, optimizer):

    mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch = batch

    outputs = model(mz_batch, int_batch, attention_mask_batch)

    loss = criterion(outputs, targets_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
