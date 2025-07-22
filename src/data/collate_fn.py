import torch
import numpy as np
from src.config import max_seq_len, vocab_size
from src.data.fingerprints_tools.fingerprint_generator import smiles_to_fingerprint
from src.data.mgf_tools.mgf_get import mgf_get_smiles


class SpectraCollateFn:

    """
    A custom collation function for PyTorch DataLoader to prepare mass spectrometry data batches
    It combines precursor and peak tokens/intensities, applies padding, and generates attention masks
    for variable-length sequences, ensuring fixed-size tensors for model input, and includes real
    fingerprint targets for supervised training
    """

    def __init__(self, smiles_df):
        self.max_length = max_seq_len
        self.padding_token_value = vocab_size
        self.smiles_df = smiles_df.copy()

        self.fingerprints_cache = {}
        self._precompute_fingerprints()

    def _precompute_fingerprints(self):

        try:
            datasets, fp_df = smiles_to_fingerprint(self.smiles_df, return_df=True)

            for _, row in fp_df.iterrows():
                spectrum_id = str(row['spectrum_id'])
                fingerprint_values = row.iloc[1:].values

                if fingerprint_values.dtype == 'object':
                    fingerprint_values = np.array(fingerprint_values, dtype=np.float32)
                else:
                    fingerprint_values = fingerprint_values.astype(np.float32)

                self.fingerprints_cache[spectrum_id] = torch.from_numpy(fingerprint_values)

            print(f'Cache created with {len(self.fingerprints_cache)} fingerprints')

        except Exception as e:
            print(f'Pre-computing fingerprints error {e}')
            self.fingerprints_cache = {}

    def _get_fingerprint_for_id(self, spectrum_id: str) -> torch.Tensor:

        if spectrum_id in self.fingerprints_cache:
            return self.fingerprints_cache[spectrum_id]

        spectrum_row = self.smiles_df[self.smiles_df['spectrum id'] == spectrum_id]

        if not spectrum_row.empty:
            try:
                dataset, fp_df = smiles_to_fingerprint(spectrum_row, return_df=True)
                fingerprint_values = fp_df.iloc[0, 1:].values
                fingerprint_tensor = torch.tensor(fingerprint_values, dtype=torch.float32)

                self.fingerprints_cache[spectrum_id] = fingerprint_tensor
                return fingerprint_tensor

            except Exception as e:
                print(f'Processing error {spectrum_id}: {e}')

        print(f'Using fingerprint zero for {spectrum_id}')
        return torch.zeros(2048, dtype=torch.float32)

    def __call__(self, batch):

        """
        This method is called by the DataLoader
        Receives a list of training_tuple's from the Deconvoluter and returns tensors ready for the model
        """

        padded_mz_tensors = []
        padded_int_tensors = []
        batch_attention_mask = []
        batch_spectrum_ids = []
        precursor_masks = []
        targets = []

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
