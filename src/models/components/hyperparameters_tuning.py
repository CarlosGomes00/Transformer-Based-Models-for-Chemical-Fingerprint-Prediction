import optuna
from src.models.Transformer import Transformer
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial: optuna.Trial, hyper_params: dict, loaders: dict):

    d_model = trial.suggest_categorical('d_model', [128, 256, 512])

    valid_n_heads = [n for n in [2, 4, 8, 16] if d_model % n == 0]
    n_head = trial.suggest_categorical('n_head', valid_n_heads)

    num_layers = trial.suggest_int('num_layers', 2, 6, step=1)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # pos_weight = trial.suggest_int('pos_weight', 1, 100, log=True)
    focal_alpha = trial.suggest_float('focal_alpha', 0.5, 0.95)
    focal_gamma = trial.suggest_float('focal_gamma', 0.5, 3)

    model = Transformer(seed=hyper_params['seed'],
                        max_seq_len=hyper_params['max_seq_len'],
                        vocab_size=hyper_params['vocab_size'],
                        morgan_default_dim=hyper_params['morgan_default_dim'],
                        d_model=d_model,
                        n_head=n_head,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate,
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        pos_weight=1,
                        loss_func='focal',
                        weight_decay=weight_decay,
                        learning_rate=learning_rate
                        )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_f1_macro")

    model_fitted = model.fit(train_loader=loaders['train'],
                             val_loader=loaders['val'],
                             max_epochs=50,
                             callbacks=[pruning_callback],
                             trial=True)

    return model.trainer.callback_metrics["val_f1_macro"].item()
