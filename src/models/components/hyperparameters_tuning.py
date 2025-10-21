import optuna
from src.models.Transformer import Transformer


def objective(trial: optuna.Trial, hyper_params: dict, loaders: dict):

    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    n_head = trial.suggest_int('n_head', 4, 8, log=True)
    num_layers = trial.suggest_int('num_layers', 2, 8, step=1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, log=True)
    #learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    #weight_decay = trial.suggest_float(
    #pos_weight Adicionar também
    #TODO Adicionar o learning rate

    model = Transformer(seed=hyper_params['seed'], max_seq_len=hyper_params['max_seq_len'], vocab_size=hyper_params['vocab_size'],
                        morgan_default_dim=hyper_params['morgan_default_dim'], d_model=d_model, n_head=n_head,
                        num_layers=num_layers, dropout_rate=dropout_rate)

    model_fitted = model.fit(train_loader=loaders['train'],
                             val_loader=loaders['val'],
                             max_epochs=50)

    return model.trainer.callback_metrics["val_loss"].item()
