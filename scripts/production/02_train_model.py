from src.data.data_loader import data_loader
from src.models.transformer_lightning import TransformerLightning
from src.config import mgf_path, vocab_size, morgan_default_dim
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything


def train_model(seed:int,
                mgf_path,
                batch_size=32,
                nhead=4,
                num_layers=4,
                dropout_rate=0.1,
                fast_dev_run=False,
                max_epochs=100):

    seed_everything(seed, workers=True)

    loaders = data_loader(batch_size=batch_size, num_workers=4, shuffle=True, mgf_path=mgf_path)

    model = TransformerLightning(vocab_size, d_model=128, nhead=nhead, num_layers=num_layers, dropout_rate=dropout_rate,
                                 fingerprint_dim=morgan_default_dim, max_seq_len=432)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4),
        ModelCheckpoint(monitor='val_loss',
                        mode='min',
                        save_top_k=3,
                        dirpath=f'outputs/checkpoints/{seed}',
                        filename='transformer-{epoch:02d}-{val_loss:.4f}')
    ]

    logger = TensorBoardLogger(save_dir='outputs/logs', name=f'{seed}_train_logs')

    trainer = Trainer(accelerator='auto', benchmark=True, deterministic=True, fast_dev_run=fast_dev_run,
                      max_epochs=max_epochs, callbacks=callbacks, logger=logger)

    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])

    return trainer.checkpoint_callback.best_model_path


if __name__ == '__main__':
    MGF_PATH = mgf_path
    SEED = 0
    BATCH_SIZE = 32
    NHEAD = 4
    NUM_LAYERS = 4
    DROPOUT_RATE = 0.1

    FAST_DEV_RUN = False
    MAX_EPOCHS = 10

    best_model = train_model(seed=SEED,
                             mgf_path=MGF_PATH,
                             batch_size=BATCH_SIZE,
                             nhead=NHEAD,
                             num_layers=NUM_LAYERS,
                             dropout_rate=DROPOUT_RATE,
                             fast_dev_run=FAST_DEV_RUN,
                             max_epochs=MAX_EPOCHS
                             )