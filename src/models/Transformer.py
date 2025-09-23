import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json
import pandas as pd
from pathlib import Path
from rdkit import DataStructs
from src.models.model_lightning import TransformerLightning
from src.utils import tensor_to_bitvect


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

    def fit(self, train_loader, val_loader, max_epochs=100, fast_dev_run=False):
        """
        Train the transformer model

        Parameters:
            train_loader : pytorch DataLoader
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

    def eval(self, test_loader, threshold=0.5, save_results=True):

        """
        Calculates evaluation metrics on the test set

        Parameters:
            test_loader : pytorch DataLoader
                Test data loader
            threshold : float
                Threshold to binning
            save_results : bool
                If true, saves the results
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        model = TransformerLightning.load_from_checkpoint(self.best_model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.eval()
        model.to(device)

        preds, targets = [], []

        with (torch.no_grad()):
            for (mz_batch,
                 int_batch,
                 attention_mask_batch,
                 batch_spectrum_ids,
                 precursor_mask_batch,
                 targets_batch) in test_loader:
                # Passar os batches para device, seja ele cuda ou cpu
                mz_batch = mz_batch.to(device)
                int_batch = int_batch.to(device)
                attention_mask_batch = attention_mask_batch.to(device)
                # precursor_mask_batch = precursor_mask_batch.to(device)

                logits = model(mz_batch, int_batch, attention_mask_batch)
                preds.append(logits.cpu())
                targets.append(targets_batch)

        pred_float = torch.cat(preds)
        targets = torch.cat(targets)
        pred_bins = (pred_float > threshold).int()

        y_true = targets.numpy().ravel()
        y_pred = pred_bins.numpy().ravel()

        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')

        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        true_bvs = [tensor_to_bitvect(fp) for fp in targets]
        pred_bvs = [tensor_to_bitvect(fp) for fp in pred_bins]

        tanimoto_values = [DataStructs.TanimotoSimilarity(a, b) for a, b in zip(true_bvs, pred_bvs)]
        mean_tanimoto = float(np.mean(tanimoto_values))

        eval_results = {'n_samples': int(targets.shape[0]),
                        'precision_macro': float(precision_macro),
                        'precision_weighted': float(precision_weighted),
                        'recall_macro': float(recall_macro),
                        'recall_weighted': float(recall_weighted),
                        'f1_macro': float(f1_macro),
                        'f1_weighted': float(f1_weighted),
                        'mean_tanimoto_similarity_predicted_vs_true_morganfingerprints': mean_tanimoto}

        if save_results:
            eval_dir = Path('outputs/eval') / str(self.seed)
            eval_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = eval_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(eval_results, f, indent=2)

            print(f'Metrics saved to {metrics_path}')

        return eval_results

    def predict(self, data_loader, return_probabilities=False, threshold=0.5, save_results=True):

        """
        Make predictions on new data

        Important: Model must be fitted to the new data

        Parameters:
            data_loader
                Dataloader with new data
            return_probabilities : bool
                If true, return raw probabilities
            threshold : float
                Threshold to binning

        Returns:
            predictions : numpy.ndarray
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        model = TransformerLightning.load_from_checkpoint(self.best_model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.eval()
        model.to(device)

        preds = []
        n_samples = 0

        print('Making predictions...')
        print(f'Device: {device}')
        print(f'Return raw probabilities: {return_probabilities}')
        if not return_probabilities:
            print(f'Binary threshold: {threshold}')

        with (torch.no_grad()):
            for batch_data in data_loader:
                mz_batch, int_batch, attention_mask_batch = batch_data[:3]

                mz_batch = mz_batch.to(device)
                int_batch = int_batch.to(device)
                attention_mask_batch = attention_mask_batch.to(device)

                logits = model(mz_batch, int_batch, attention_mask_batch)
                probabilities = torch.sigmoid(logits)
                preds.append(probabilities.cpu())
                n_samples += probabilities.shape[0]

        pred_probabilities = torch.cat(preds, dim=0)

        print(f'Predictions made for {n_samples} samples')

        if return_probabilities:
            result = pred_probabilities.numpy()
            print(f"   Probability range: [{result.min():.4f}, {result.max():.4f}]")

            if save_results:
                pd.DataFrame(result).to_csv('predictions.csv', index=False)
            return result

        else:
            pred_binary = (pred_probabilities > threshold).int()
            result = pred_binary.numpy()

            if save_results:
                pd.DataFrame(result).to_csv('predictions_bin.csv', index=False)
            return result

    def score(self, y_true):
        """
        Analisar as previsões contra os targets reais
        """
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
