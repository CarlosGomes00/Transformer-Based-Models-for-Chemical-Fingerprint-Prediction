from src.data.flexible_dataloader import data_loader_f
from src.model.transformer import EncoderTransformer
from src.training.training import training_setup, train_step
from src.config import *


def training_validation():

    print('Loading Data')
    try:
        train_data = data_loader_f(batch_size=4,
                                   num_spectra=20,
                                   shuffle=True,
                                   mgf_path=mgf_path)

    except Exception as e:
        print(f'Error found: {e}')
        return False

    print('Loading Model')
    try:
        model = EncoderTransformer(vocab_size=vocab_size,
                                   d_model=d_model,
                                   nhead=nhead,
                                   num_layers=4,
                                   dropout_rate=dropout_rate,
                                   fingerprint_dim=morgan_default_dim,
                                   max_seq_len=max_seq_len)
    except Exception as e:
        print(f'Error found: {e}')
        return False

    print('Training Setup')
    try:
        criterion, optimizer = training_setup(model=model)
        print(f'Loss: {type(criterion).__name__}')
        print(f'Optimizer: {type(optimizer).__name__}')

    except Exception as e:
        print(f'Error found: {e}')
        return False

    print('Batch testing')
    try:
        for batch in train_data:
            mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets = batch

            print('Tensors shape')
            print(f'mz_batch: {mz_batch.shape}')
            print(f'int_batch: {int_batch.shape}')
            print(f'attention_mask_batch: {attention_mask_batch.shape}')
            print(f'batch_spectrum_ids (first 3): {batch_spectrum_ids[:3]}')
            print(f'precursor_mask_batch: {precursor_mask_batch.shape}')
            print(f'targets: {targets.shape}')

            outputs = model(mz_batch, int_batch, attention_mask_batch)
            loss = criterion(outputs, targets)

            active_bits = targets.sum(dim=1).tolist()
            if all(bits == 0 for bits in active_bits):
                print('Empty Targets')
                return False
            else:
                print('Targets aren´t empty')

    except Exception as e:
        print(f'Batch Error {e}')
        return False

    print('Validation Training')
    model.train()

    for epoch in range(3):
        epoch_loss = 0
        num_batches = 0

        print(f'Epoch {epoch +1 }/3')

        for batch_idx, batch in enumerate(train_data):
            try:
                loss = train_step(model, batch, criterion, optimizer)
                epoch_loss += loss
                num_batches += 1

                print(f'Batch {batch_idx}: Loss = {loss:.4f}')

            except Exception as e:
                print(f'Error found: {e}')
                return False

    avg_loss = epoch_loss / num_batches
    print(f'Avg Loss: {avg_loss:.4f}')


if __name__ == '__main__':
    print('Transformer Pipeline validation')