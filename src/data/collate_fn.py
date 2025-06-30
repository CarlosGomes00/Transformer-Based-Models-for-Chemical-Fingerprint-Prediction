import torch
import numpy as np
from src.config import max_seq_len, vocab_size


class SpectraCollateFn:

    """
    A custom collation function for PyTorch DataLoader to prepare mass spectrometry data batches
    It combines precursor and peak tokens/intensities, applies padding, and generates attention masks
    for variable-length sequences, ensuring fixed-size tensors for model input
    """

    def __init__(self):
        self.max_length = max_seq_len
        self.padding_token_value = vocab_size

    def __call__(self, batch):

        """
        This method is called by the DataLoader
        Receives a list of training_tuple's from the Deconvoluter and returns tensors ready for the model
        """

        padded_mz_tensors = []
        padded_int_tensors = []
        batch_attention_mask = []
        batch_spectrum_ids = []

        for training_tuple in batch:

            spectrum_id, tokenized_mz, tokenized_precursor, int_array = training_tuple

            mz_tokens_list = [tokenized_precursor] + tokenized_mz
            int_tokens_list = [0.0] + int_array.tolist()

            num_padding_needed = self.max_length - len(mz_tokens_list)

            if num_padding_needed > 0:
                padding_mz_tokens = [self.padding_token_value] * num_padding_needed
                padding_int_tokens = [0.0] * num_padding_needed

                mz_tokens_list = mz_tokens_list + padding_mz_tokens
                int_tokens_list = int_tokens_list + padding_int_tokens

            mz_tokens_tensor = torch.tensor(mz_tokens_list, dtype=torch.int64)
            int_tokens_tensor = torch.tensor(int_tokens_list, dtype=torch.float32)

            new_attention_mask = (mz_tokens_tensor != self.padding_token_value)

            padded_mz_tensors.append(mz_tokens_tensor)
            padded_int_tensors.append(int_tokens_tensor)
            batch_attention_mask.append(new_attention_mask)
            batch_spectrum_ids.append(spectrum_id)

        mz_batch = torch.stack(padded_mz_tensors)
        int_batch = torch.stack(padded_int_tensors)
        attention_mask_batch = torch.stack(batch_attention_mask)

        return mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids
