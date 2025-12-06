import pandas as pd
import numpy as np
import pickle
from pathlib import Path


SEED = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
artifacts_dir = REPO_ROOT / "src/data/artifacts" / str(SEED)


print("A carregar fingerprints e splits...")
fp_df = pd.read_pickle(artifacts_dir / "fingerprints.pkl")


with open(artifacts_dir / "split_ids.pkl", "rb") as f:
    splits = pickle.load(f)

train_ids = splits['train']


train_fps = fp_df[fp_df['spectrum_id'].isin(train_ids)]


feature_cols = [c for c in train_fps.columns if c.startswith('fp_')]
X_train = train_fps[feature_cols].values

print(f"Dimensões do Treino: {X_train.shape}")


n_total = X_train.size
n_ones = np.count_nonzero(X_train)
n_zeros = n_total - n_ones

print(n_ones)
print(n_total)
print(n_zeros)

pos_weight = n_zeros / n_ones

print("-" * 30)
print(f"POS_WEIGHT EXATO: {pos_weight:.4f}")
print("-" * 30)
