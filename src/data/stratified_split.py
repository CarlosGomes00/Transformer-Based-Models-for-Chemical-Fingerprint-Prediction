import pickle
import numpy as np
import pandas as pd
from deepmol.splitters import MultiTaskStratifiedSplitter
from pathlib import Path


def make_split(dataset, seed, output_dir):

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splitter = MultiTaskStratifiedSplitter()

    train_dataset, val_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=seed
    )

    splits = {"train": train_dataset.ids, "val": val_dataset.ids, "test": test_dataset.ids}

    split_pkl = output_dir / 'split_ids.pkl'
    with split_pkl.open('wb') as f:
        pickle.dump(splits, f)

    print(f"Saving fingerprints cache")
    all_fingerprints = []
    all_ids = []

    all_fingerprints.append(train_dataset.X)
    all_ids.extend(train_dataset.ids)

    all_fingerprints.append(val_dataset.X)
    all_ids.extend(val_dataset.ids)

    all_fingerprints.append(test_dataset.X)
    all_ids.extend(test_dataset.ids)

    all_fp = np.vstack(all_fingerprints)

    fp_df = pd.DataFrame(all_fp, columns=[f'fp_{i}' for i in range(all_fp.shape[1])])
    fp_df['spectrum_id'] = all_ids

    fp_cache_path = output_dir / 'fingerprints.pkl'
    fp_df.to_pickle(fp_cache_path)

    print(f"Saved splits to {output_dir}"
          f"(train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})")

    print(f"Saved fingerprints cache: {fp_df.shape[0]} samples, {fp_df.shape[1] - 1} features")
    print(f"Files created:")
    print(f"  • {split_pkl}")
    print(f"  • {fp_cache_path}")

    return splits
