def mean_pooling(x, attention_mask):

    """
    Function that performs a mean pooling operation on an input tensor x, taking into account an attention_mask

    Params:
        x : torch.Tensor [batch, seq_len, d_model]
        attention_mask : torch.Tensor [batch, seq_len]

    Returns:
        torch.Tensor [batch, d_model]
    """

    mask = attention_mask.unsqueeze(-1)
    summed = (x * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    return summed / count
