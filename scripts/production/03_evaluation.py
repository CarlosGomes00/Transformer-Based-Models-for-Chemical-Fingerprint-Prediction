from rdkit import DataStructs

from src.models.transformer_lightning import TransformerLightning
from src.data.data_loader import data_loader
from src.config import mgf_path
from utils import tensor_to_bitvect
import os
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import json


def evaluate_model(model_checkpoint_path: str,
                   mgf_path: str,
                   batch_size: int = None,
                   threshold: float = 0.5,
                   save_fp: bool = False) -> dict:

    os.makedirs("outputs/eval", exist_ok=True)

    # Dar load do modelo
    model = TransformerLightning.load_from_checkpoint(model_checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Dar load dos dados de teste
    loaders = data_loader(
        batch_size=batch_size or 32,
        num_workers=4,
        shuffle=False,
        mgf_path=mgf_path)

    test_loader = loaders["test"]

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
            #precursor_mask_batch = precursor_mask_batch.to(device)

            logits = model(mz_batch, int_batch, attention_mask_batch)
            preds.append(logits.cpu())
            targets.append(targets_batch)

    pred_float = torch.cat(preds)
    targets = torch.cat(targets)
    pred_bins = (pred_float > threshold).int()

    y_true = targets.numpy().ravel()
    y_pred = pred_bins.numpy().ravel()

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    true_bvs = [tensor_to_bitvect(fp) for fp in targets]
    pred_bvs = [tensor_to_bitvect(fp) for fp in pred_bins]

    tanimoto_values = [DataStructs.TanimotoSimilarity(a, b) for a, b in zip(true_bvs, pred_bvs)]
    mean_tanimoto = float(np.mean(tanimoto_values))

    results = {'n_samples': int(targets.shape[0]),
               'precision': float(precision),
               'recall': float(recall),
               'f1': float(f1),
               'mean_tanimoto': mean_tanimoto}

    with open("outputs/eval/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    if save_fp:
        torch.save({"pred_float": pred_float, "pred_bins": pred_bins}, "outputs/eval/fingerprints.pt")

    return results


if __name__ == "__main__":

    evaluate_model(model_checkpoint_path="outputs/checkpoints/transformer-epoch=07-val_loss=0.2365.ckpt",
                   mgf_path=mgf_path,
                   batch_size=None,
                   threshold=0.5,
                   save_fp=True)
