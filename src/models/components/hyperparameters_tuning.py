import os
import numpy as np
import optuna
from sklearn.metrics import f1_score
from src.utils import tensor_to_bitvect
from rdkit import DataStructs

from src.models.Transformer import Transformer
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial: optuna.Trial, hyper_params: dict, loaders: dict, target_type:str):

    # d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    # valid_n_heads = [n for n in [2, 4, 8, 16] if d_model % n == 0]
    # n_head = trial.suggest_categorical('n_head', valid_n_heads)
    # num_layers = trial.suggest_int('num_layers', 2, 6, step=1)

    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)

    learning_rate = trial.suggest_float('learning_rate', 3e-5, 1e-3, log=True)

    pos_weight = trial.suggest_float('pos_weight', 1, 3)
    # focal_alpha = trial.suggest_float('focal_alpha', 0.7, 0.95)
    # focal_gamma = trial.suggest_float('focal_gamma', 0.5, 3)

    model = Transformer(seed=hyper_params['seed'],
                        max_seq_len=hyper_params['max_seq_len'],
                        vocab_size=hyper_params['vocab_size'],
                        target_type=target_type,
                        d_model=256,
                        n_head=4,
                        num_layers=4,
                        dropout_rate=dropout_rate,
                        focal_alpha=0.25,
                        focal_gamma=2,
                        pos_weight=pos_weight,
                        loss_func='bce_logits',
                        weight_decay=weight_decay,
                        learning_rate=learning_rate,
                        batch_norm=True)

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="Loss/Val")

    model_fitted = model.fit(train_loader=loaders['train'],
                             val_loader=loaders['val'],
                             max_epochs=50,
                             callbacks=[pruning_callback])

    final_model = model.load_model(checkpoint_path=model_fitted.best_model_path, seed=hyper_params['seed'])

    val_metrics = final_model.validate(loaders["val"], save_results=False)

    tanimoto_score = val_metrics[f'mean_tanimoto_similarity_predicted_vs_true_{target_type}']

    os.remove(model_fitted.best_model_path)

    return tanimoto_score
