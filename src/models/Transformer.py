import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json
from pathlib import Path
from rdkit import DataStructs


class Transformer:

    def __init__(self,
                 seed,
                 max_seq_len,
                 vocab_size,
                 morgan_default_dim,
                 d_model,
                 n_head,
                 num_layers,
                 dropout_rate):

        self.seed = seed
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.morgan_default_dim = morgan_default_dim
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.model = None
        self.trainer = None
        self.best_model_path = None
        self.is_fitted = False

# ATENÇÃO: Ver onde colocar o data_loader, é preciso para dar a info ao .fit

    def fit(self, train_loader, val_loader, max_epochs=100, fast_dev_run=False):
        """
        Train the transformer model

        Parameters:
            train_loader :
                Train data loader
            val_loader :
                Validation data loader
            max_epochs : int
                Maximum number of epochs to train the model
            fast_dev_run : bool
                If True, runs 1 batch to ensure code will execute without errors (Debugging purposes)
        """

        from src.models.model_lightning import TransformerLightning

        pl.seed_everything(seed=self.seed, workers=True)

        self.model = TransformerLightning(vocab_size=self.vocab_size,
                                          max_seq_len=self.max_seq_len,
                                          d_model=self.d_model,
                                          nhead=self.n_head,
                                          num_layers=self.num_layers,
                                          dropout_rate=self.dropout_rate,
                                          fingerprint_dim=self.morgan_default_dim)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4),
            ModelCheckpoint(monitor='val_loss',
                            mode='min',
                            save_top_k=3,
                            dirpath=f'outputs/checkpoints/{self.seed}',
                            filename='transformer-{epoch:02d}-{val_loss:.4f}')
        ]

        logger = TensorBoardLogger(save_dir='outputs/logs', name=f'{self.seed}_train_logs')

        self.trainer = pl.Trainer(accelerator='auto', benchmark=True, deterministic=True, fast_dev_run=fast_dev_run,
                                  max_epochs=max_epochs, callbacks=callbacks, logger=logger)

        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self.best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.is_fitted = True

        return self.best_model_path

    def score(self, test_loader, y_true):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

            # Calcula as métricas
        return

    def predict(self, test_loader):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # Fazer as previsões

        return

    def load_model(self, checkpoint_path):
        """
        Load from a checkpoint

        Parameters:
            checkpoint_path : Path
                Path to the checkpoint
        """

        from src.models.model_lightning import TransformerLightning

        self.model = TransformerLightning.load_from_checkpoint(checkpoint_path)
        self.best_model_path = checkpoint_path
        self.is_fitted = True
        print(f'Model loaded from: {checkpoint_path}')

        return
