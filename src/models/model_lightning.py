import torch
import pytorch_lightning as pl
from src.training.training import training_setup, training_setup_weighted, train_step_lightning
from src.models.model import EncoderTransformer


class TransformerLightning(pl.LightningModule):

    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout_rate, fingerprint_dim,
                 max_seq_len):

        super().__init__()

        self.save_hyperparameters()

        self.model = EncoderTransformer(vocab_size=vocab_size,
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_layers=num_layers,
                                        dropout_rate=dropout_rate,
                                        fingerprint_dim=fingerprint_dim,
                                        max_seq_len=max_seq_len)

        self.criterion, _ = training_setup(self.model)
        # self.criterion, _ = training_setup_weighted(self.model) Teste para diferentes loss functions
        # (quando ativar esta, remover a sigmoid na ultima layer)

    def forward(self, mz_batch, int_batch, attention_mask):
        return self.model(mz_batch, int_batch, attention_mask)

    def training_step(self, batch, batch_idx):
        loss = train_step_lightning(self.model, batch, self.criterion)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        mz_batch, int_batch, attention_mask_batch, batch_spectrum_ids, precursor_mask_batch, targets_batch = batch

        outputs = self.forward(mz_batch, int_batch, attention_mask_batch)
        loss = self.criterion(outputs, targets_batch)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):

        _, optimizer = training_setup(self.model)

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
                    'monitor': 'val_loss',
                    'frequency': 1}}
