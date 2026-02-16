import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from rdkit import DataStructs

from src.models.Transformer import Transformer
from src.models.model_lightning import TransformerLightning
from src.utils import *
from src.data.data_loader import *


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_predictions_and_targets(model, dataloader, device):
    """
    Get raw predictions (logits/probs) and targets from dataloader.
    Runs inference only ONCE.

    Returns:
        pred_probs : torch.Tensor [N, 2048]
            Probabilities for each bit
        targets : torch.Tensor [N, 2048]
            Ground truth targets
    """

    model.eval()
    model.to(device)

    all_logits = []
    all_targets = []

    print("Generating predictions (running model inference once)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            (mz_batch,
             int_batch,
             attention_mask_batch,
             batch_spectrum_ids,
             targets_batch) = batch

            mz_batch = mz_batch.to(device)
            int_batch = int_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            logits = model(mz_batch, int_batch, attention_mask_batch)

            all_logits.append(logits.cpu())
            all_targets.append(targets_batch)

    pred_logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    loss_func = model.hparams.loss_func

    if loss_func in ('bce_logits', 'focal'):
        pred_probs = torch.sigmoid(pred_logits)
    elif loss_func == 'bce':
        pred_probs = pred_logits
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    print(f"✓ Predictions generated: {pred_probs.shape}")

    return pred_probs, targets


def compute_metrics_for_threshold(pred_probs, targets, threshold):
    """
    Compute all metrics for a given threshold.

    Parameters:
        pred_probs : torch.Tensor [N, 2048]
            Probabilities
        targets : torch.Tensor [N, 2048]
            Ground truth
        threshold : float
            Threshold to apply

    Returns:
        dict : Metrics
    """

    # Binarize predictions
    pred_bins = (pred_probs > threshold).int()

    # Convert to numpy
    y_true = targets.numpy()
    y_pred = pred_bins.numpy()

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    true_bvs = [tensor_to_bitvect(fp) for fp in targets]
    pred_bvs = [tensor_to_bitvect(fp) for fp in pred_bins]
    tanimoto_values = [DataStructs.TanimotoSimilarity(a, b) for a, b in zip(true_bvs, pred_bvs)]
    mean_tanimoto = float(np.mean(tanimoto_values))

    return {
        'threshold': float(threshold),
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'tanimoto': float(mean_tanimoto)
    }


def sweep_thresholds(pred_probs, targets, thresholds):
    """
    Sweep multiple thresholds and compute metrics for each.

    Returns:
        list of dict : Metrics for each threshold
    """

    results = []

    print(f"\nSweeping {len(thresholds)} thresholds...")

    for t in tqdm(thresholds, desc="Thresholds"):
        metrics = compute_metrics_for_threshold(pred_probs, targets, t)
        results.append(metrics)

    return results


def plot_threshold_sweep(results_df, output_dir):
    """
    Plot metrics vs threshold and save figure.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1 scores
    axes[0, 0].plot(results_df['threshold'], results_df['f1_macro'], 'o-', label='F1-macro', linewidth=2)
    axes[0, 0].plot(results_df['threshold'], results_df['f1_weighted'], 's-', label='F1-weighted', linewidth=2)
    axes[0, 0].set_xlabel('Threshold', fontsize=12)
    axes[0, 0].set_ylabel('F1 Score', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('F1 Scores vs Threshold', fontsize=13, fontweight='bold')

    # Precision
    axes[0, 1].plot(results_df['threshold'], results_df['precision_macro'], 'o-', label='Precision-macro', linewidth=2)
    axes[0, 1].plot(results_df['threshold'], results_df['precision_weighted'], 's-', label='Precision-weighted',
                    linewidth=2)
    axes[0, 1].set_xlabel('Threshold', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Precision vs Threshold', fontsize=13, fontweight='bold')

    # Recall
    axes[1, 0].plot(results_df['threshold'], results_df['recall_macro'], 'o-', label='Recall-macro', linewidth=2)
    axes[1, 0].plot(results_df['threshold'], results_df['recall_weighted'], 's-', label='Recall-weighted', linewidth=2)
    axes[1, 0].set_xlabel('Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Recall', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Recall vs Threshold', fontsize=13, fontweight='bold')

    # Tanimoto
    axes[1, 1].plot(results_df['threshold'], results_df['tanimoto'], 'o-', color='green', linewidth=2)
    axes[1, 1].set_xlabel('Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Tanimoto Similarity', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Tanimoto vs Threshold', fontsize=13, fontweight='bold')

    plt.tight_layout()

    plot_path = output_dir / 'threshold_sweep.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved: {plot_path}")


def main(args):
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # 1. Load model
    print("\n1. Loading model...")
    model_wrapper = Transformer.load_model(
        checkpoint_path=Path(args.checkpoint_path),
        seed=args.seed
    )

    pl_model = model_wrapper.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"✓ Model loaded from: {args.checkpoint_path}")
    print(f"✓ Loss function: {pl_model.hparams.loss_func}")
    print(f"✓ pos_weight: {pl_model.hparams.pos_weight}")
    print(f"✓ Device: {device}")

    try:
        artifacts_dir = Path(args.artifacts_dir) / str(args.seed)

        with open(artifacts_dir / 'pipeline_config.json', 'r') as f:
            pipeline_config = json.load(f)

        max_num_peaks = pipeline_config['max_num_peaks']
        mz_vocabs = pipeline_config['mz_vocabs']

    except Exception as e:
        print(f'Error found: {e}')
        raise

    # 2. Load validation data
    print("\n2. Loading validation data...")
    loaders = data_loader(
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_spectra=args.num_spectra,
        mgf_path=args.mgf_path,
        max_num_peaks=max_num_peaks,
        mz_vocabs=mz_vocabs)

    val_loader = loaders['val']
    print(f"✓ Validation set loaded: {len(val_loader.dataset)} samples")

    # 3. Get predictions (ONCE)
    print("\n3. Generating predictions...")
    pred_probs, targets = get_predictions_and_targets(pl_model, val_loader, device)

    # 4. Sweep thresholds
    print("\n4. Sweeping thresholds...")
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    thresholds = [round(t, 2) for t in thresholds]  # Round to avoid floating point issues

    print(f"✓ Testing {len(thresholds)} thresholds: {thresholds[0]} to {thresholds[-1]}")

    results = sweep_thresholds(pred_probs, targets, thresholds)
    results_df = pd.DataFrame(results)

    # 5. Find best thresholds
    print("\n5. Finding optimal thresholds...")

    best_f1_macro_idx = results_df['f1_macro'].idxmax()
    best_f1_weighted_idx = results_df['f1_weighted'].idxmax()
    best_tanimoto_idx = results_df['tanimoto'].idxmax()

    best_f1_macro = results_df.loc[best_f1_macro_idx]
    best_f1_weighted = results_df.loc[best_f1_weighted_idx]
    best_tanimoto = results_df.loc[best_tanimoto_idx]

    print("\n" + "=" * 70)
    print("BEST THRESHOLDS")
    print("=" * 70)

    print(f"\n🔥 Best F1-macro: threshold = {best_f1_macro['threshold']:.2f}")
    print(f"   F1-macro:          {best_f1_macro['f1_macro']:.4f}")
    print(f"   Precision-macro:   {best_f1_macro['precision_macro']:.4f}")
    print(f"   Recall-macro:      {best_f1_macro['recall_macro']:.4f}")

    print(f"\n🔥 Best F1-weighted: threshold = {best_f1_weighted['threshold']:.2f}")
    print(f"   F1-weighted:       {best_f1_weighted['f1_weighted']:.4f}")
    print(f"   Precision-weighted:{best_f1_weighted['precision_weighted']:.4f}")
    print(f"   Recall-weighted:   {best_f1_weighted['recall_weighted']:.4f}")

    print(f"\n🔥 Best Tanimoto: threshold = {best_tanimoto['threshold']:.2f}")
    print(f"   Tanimoto:          {best_tanimoto['tanimoto']:.4f}")

    # 6. Save results
    output_dir = REPO_ROOT / 'outputs' / 'threshold_tuning' / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all results
    csv_path = output_dir / 'threshold_sweep_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Full results saved: {csv_path}")

    # Save best thresholds JSON
    best_thresholds = {
        'checkpoint_path': str(args.checkpoint_path),
        'loss_func': pl_model.hparams.loss_func,
        'pos_weight': float(pl_model.hparams.pos_weight) if pl_model.hparams.pos_weight else None,
        'best_threshold_f1_macro': float(best_f1_macro['threshold']),
        'best_f1_macro_value': float(best_f1_macro['f1_macro']),
        'best_threshold_f1_weighted': float(best_f1_weighted['threshold']),
        'best_f1_weighted_value': float(best_f1_weighted['f1_weighted']),
        'best_threshold_tanimoto': float(best_tanimoto['threshold']),
        'best_tanimoto_value': float(best_tanimoto['tanimoto']),
        'default_threshold_0.5_metrics': results_df[results_df['threshold'] == 0.5].to_dict('records')[
            0] if 0.5 in thresholds else None
    }

    json_path = output_dir / 'best_thresholds.json'
    with open(json_path, 'w') as f:
        json.dump(best_thresholds, f, indent=2)

    print(f"✓ Best thresholds saved: {json_path}")

    # 7. Plot
    print("\n6. Generating plots...")
    plot_threshold_sweep(results_df, output_dir)

    print("\n" + "=" * 70)
    print("✓ THRESHOLD OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nRecommended threshold for evaluation: {best_f1_macro['threshold']:.2f}")
    print(f"(Optimized for F1-macro)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal threshold for trained model"
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--mgf_path',
        type=str,
        required=True,
        help='Path to MGF file'
    )

    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Seed used for training'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for inference'
    )

    parser.add_argument(
        '--min_threshold',
        type=float,
        default=0.1,
        help='Minimum threshold to test'
    )

    parser.add_argument(
        '--max_threshold',
        type=float,
        default=0.9,
        help='Maximum threshold to test'
    )

    parser.add_argument(
        '--step',
        type=float,
        default=0.05,
        help='Step size for threshold sweep'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of DataLoader workers (use 0 for safety)'
    )

    parser.add_argument(
        '--num_spectra',
        type=int,
        default=None,
        help='Number of spectra to use (None = all)'
    )

    parser.add_argument('--artifacts_dir', type=str, default=REPO_ROOT / 'src/data/artifacts',
                        help='Artifacts directory')

    args = parser.parse_args()
    main(args)