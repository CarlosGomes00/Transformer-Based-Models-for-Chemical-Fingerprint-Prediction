import os
import pickle
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

    print(f"Saved splits to {output_dir}"
          f"(train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})")

    return splits
