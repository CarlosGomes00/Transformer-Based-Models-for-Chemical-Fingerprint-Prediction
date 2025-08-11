import torch.nn as nn
import torch.optim as optim
from config import learning_rate, weight_decay


def training_setup(model):

    """
    Configures loss function and optimizer for transformer training.

    Parameters:
        model : EncoderTransformer
            The transformer model to be trained

    Returns:
        tuple[nn.BCELoss, optim.Adam]
            Loss function and optimizer ready for training loop
    """

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    return criterion, optimizer


def train_step(model, batch, criterion, optimizer):

    """
    Performs forward and backward pass on the training batch

    Parameters:
        model : EncoderTransformer
            The transformer model being trained. Must be in training mode
        batch : tuple
            Batch data from the DataLoader
        criterion : nn.BCELoss
            Loss function for binary classification
        optimizer : optim.Adam
            Optimizer for parameter updates

    Returns:
        float
            Loss value for the current batch
    """

    mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch = batch

    outputs = model(mz_batch, int_batch, attention_mask_batch)

    loss = criterion(outputs, targets_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_step_lightning(model, batch, criterion):
    """
    Lightning version of train_step without backward pass

    Parameters:
        model : EncoderTransformer
            The transformer model being trained
        batch : tuple
            Batch data from the DataLoader
        criterion : nn.BCELoss
            Loss function for binary classification

    Returns:
        torch.Tensor
            Loss tensor for Lightning
    """

    mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch = batch

    outputs = model(mz_batch, int_batch, attention_mask_batch)
    loss = criterion(outputs, targets_batch)

    return loss
