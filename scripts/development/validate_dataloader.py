import sys
from pathlib import Path
from src.data.data_loader import data_loader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def test_dataloader():

    print("Testing DataLoader with splits...")

    loaders = data_loader(batch_size=8, num_spectra=100)

    print(f"Keys created: {list(loaders.keys())}")

    for split_name, loader in loaders.items():
        print(f"\n Testing {split_name} split:")
        try:
            batch = next(iter(loader))
            print(f" Batch size: {len(batch)}")
            print(f" Tensor shapes: {[t.shape for t in batch if hasattr(t, 'shape')]}")
            print(f" Spectrum IDs: {batch[3][:3]}...")
            print(f" {split_name} DataLoader working correctly!")
        except Exception as e:
            print(f" Error in {split_name}: {e}")

    print(f"\n DataLoader validation completed!")


if __name__ == '__main__':
    test_dataloader()
