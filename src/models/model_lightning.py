import torch
import lightning.pytorch as pl
from torchmetrics import F1Score

from src.training.training import training_setup, training_setup_weighted, train_step_lightning
from torchvision.ops import sigmoid_focal_loss
from src.models.model import EncoderTransformer


class TransformerLightning(pl.LightningModule):

    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout_rate, fingerprint_dim, max_seq_len,
                 pos_weight,
                 learning_rate,
                 weight_decay,
                 loss_func: str = 'bce_logits',
                 focal_gamma: float = 2,
                 focal_alpha: float = 0.25):

        super().__init__()

        self.save_hyperparameters()

        self.model = EncoderTransformer(vocab_size=vocab_size,
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_layers=num_layers,
                                        dropout_rate=dropout_rate,
                                        fingerprint_dim=fingerprint_dim,
                                        max_seq_len=max_seq_len,
                                        head_type='logits' if loss_func in ('bce_logits', 'focal') else 'sigmoid')

        self.criterion = None
        if loss_func == 'bce':
            self.criterion, _ = training_setup(self.model, learning_rate=self.hparams.learning_rate,
                                               weight_decay=self.hparams.weight_decay)

        elif loss_func == 'bce_logits':
            self.criterion, _ = training_setup_weighted(self.model, self.hparams.pos_weight,
                                                        learning_rate=self.hparams.learning_rate,
                                                        weight_decay=self.hparams.weight_decay)

    def forward(self, mz_batch, int_batch, attention_mask):
        return self.model(mz_batch, int_batch, attention_mask)

    def training_step(self, batch, batch_idx):

        mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch = batch

        outputs = self.forward(mz_batch, int_batch, attention_mask_batch)

        if self.hparams.loss_func == 'focal':
            loss = sigmoid_focal_loss(outputs, targets=targets_batch,
                                      gamma=self.hparams.focal_gamma,
                                      alpha=self.hparams.focal_alpha,
                                      reduction='mean')
        else:
            loss = self.criterion(outputs, targets_batch)

        self.log('Loss/Train', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch = batch

        outputs = self.forward(mz_batch, int_batch, attention_mask_batch)

        if self.hparams.loss_func == 'focal':
            loss = sigmoid_focal_loss(outputs, targets=targets_batch,
                                      gamma=self.hparams.focal_gamma,
                                      alpha=self.hparams.focal_alpha,
                                      reduction='mean')
        else:
            loss = self.criterion(outputs, targets_batch)

        self.log('Loss/Val', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):

        _, optimizer = training_setup(self.model, learning_rate=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': 'Loss/Val',
                    'frequency': 1}}
