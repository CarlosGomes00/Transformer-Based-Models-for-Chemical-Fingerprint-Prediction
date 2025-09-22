from src.data.data_splitting import preprocess_and_split
from src.config import mgf_path
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]


if __name__ == '__main__':
    seed = 0
    output_dir = REPO_ROOT / "src/data/artifacts"

    results = preprocess_and_split(
        mgf_path=mgf_path,
        seed=seed,
        output_dir=output_dir,
        num_spectra=10
    )
