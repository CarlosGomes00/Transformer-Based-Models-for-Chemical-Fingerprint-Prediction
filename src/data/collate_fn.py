import torch
import numpy as np
from src.config import max_seq_len, vocab_size
from src.data.fingerprints_tools.fingerprint_generator import smiles_to_fingerprint


class SpectraCollateFn:

    """
    A custom collation function for PyTorch DataLoader to prepare mass spectrometry data batches
    It combines precursor and peak tokens/intensities, applies padding, and generates attention masks
    for variable-length sequences, ensuring fixed-size tensors for models input, and includes real
    fingerprint targets for supervised training
    """

    def __init__(self, fingerprints_df):
        self.max_length = max_seq_len
        self.padding_token_value = vocab_size
        self.fingerprints_df = fingerprints_df.copy()

        self.fingerprints_cache = {}
        self._load_precomputed_fingerprints()

    def _load_precomputed_fingerprints(self):

        try:
            for _, row in self.fingerprints_df.iterrows():
                spectrum_id = str(row['spectrum_id'])

                fingerprint_cols = [col for col in row.index if col.startswith('fp_')]
                fingerprint_values = row[fingerprint_cols].values.astype(np.float32)

                self.fingerprints_cache[spectrum_id] = torch.from_numpy(fingerprint_values)

        except Exception as e:
            print(f'Error loading pre-computed fingerprints {e}')
            self.fingerprints_cache = {}

    def _get_fingerprint_for_id(self, spectrum_id: str) -> torch.Tensor:

        spectrum_id = str(spectrum_id)

        if spectrum_id in self.fingerprints_cache:
            return self.fingerprints_cache[spectrum_id]
        else:
            print(f'Fingerprint not found for {spectrum_id}, using zeros')
            return torch.zeros(2048, dtype=torch.float32)

    def __call__(self, batch):

        """
        This method is called by the DataLoader

        Receives a list of training_tuple's from the Deconvoluter and returns tensors ready for the models
        """

        padded_mz_tensors = []
        padded_int_tensors = []
        batch_attention_mask = []
        batch_spectrum_ids = []
        precursor_masks = []
        targets = []

        for training_tuple in batch:

            spectrum_id, tokenized_precursor, tokenized_mz, precursor_int, int_array = training_tuple

            mz_tokens_list = [tokenized_precursor] + tokenized_mz
            int_tokens_list = [precursor_int] + int_array.tolist()

            num_padding_needed = self.max_length - len(mz_tokens_list)

            if num_padding_needed > 0:
                padding_mz_tokens = [self.padding_token_value] * num_padding_needed
                padding_int_tokens = [0.0] * num_padding_needed

                mz_tokens_list = mz_tokens_list + padding_mz_tokens
                int_tokens_list = int_tokens_list + padding_int_tokens

            mz_tokens_tensor = torch.tensor(mz_tokens_list, dtype=torch.int64)
            int_tokens_tensor = torch.tensor(int_tokens_list, dtype=torch.float32)

            new_attention_mask = (mz_tokens_tensor != self.padding_token_value)
            precursor_mask = torch.zeros(self.max_length, dtype=torch.float32)
            precursor_mask[0] = 1

            target_fingerprint = self._get_fingerprint_for_id(spectrum_id)

            padded_mz_tensors.append(mz_tokens_tensor)
            padded_int_tensors.append(int_tokens_tensor)
            batch_attention_mask.append(new_attention_mask)
            batch_spectrum_ids.append(spectrum_id)
            precursor_masks.append(precursor_mask)
            targets.append(target_fingerprint)

        mz_batch = torch.stack(padded_mz_tensors)
        int_batch = torch.stack(padded_int_tensors)
        attention_mask_batch = torch.stack(batch_attention_mask)
        precursor_mask_batch = torch.stack(precursor_masks)
        targets_batch = torch.stack(targets)

        return mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch
