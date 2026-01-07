import pandas as pd
import requests
import time
import argparse
from pathlib import Path
from urllib.parse import quote
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_npclassifier_predictions(smiles, retries=3):

    encoded_smiles = quote(smiles)
    url = f"https://npclassifier.ucsd.edu/classify?smiles={encoded_smiles}"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                pathway = data.get('pathway_results', ['Unknown'])[0] if data.get('pathway_results') else 'Unknown'
                superclass = data.get('superclass_results', ['Unknown'])[0] if data.get(
                    'superclass_results') else 'Unknown'
                class_val = data.get('class_results', ['Unknown'])[0] if data.get('class_results') else 'Unknown'
                return pathway, superclass, class_val
            elif response.status_code == 500:
                return "Server_Error", "Server_Error", "Server_Error"

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(1 * (attempt + 1))
            continue
        except Exception as e:
            return "Error", "Error", "Error"

    return "Failed", "Failed", "Failed"

def main():
    parser = argparse.ArgumentParser(description="Fetch NPClassifier classes via API with checkpoints")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV with SMILES")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--batch_size", type=int, default=100, help="Save checkpoint every N molecules")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel requests")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    start_idx = 0
    results = {'npc_pathway': [], 'npc_superclass': [], 'npc_class': []}

    if output_path.exists():
        df_checkpoint = pd.read_csv(output_path)

        if len(df_checkpoint) < len(df):
            start_idx = len(df_checkpoint)
            print(f"Retomando checkpoint encontrado! Começando no índice {start_idx}")

            results['npc_pathway'] = df_checkpoint['npc_pathway'].tolist()
            results['npc_superclass'] = df_checkpoint['npc_superclass'].tolist()
            results['npc_class'] = df_checkpoint['npc_class'].tolist()
        else:
            print("Ficheiro de output já está completo! Nada a fazer.")
            return

        print(f"Processando {len(df) - start_idx} moléculas com {args.workers} threads...")

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:

                pbar = tqdm(total=len(df) - start_idx, initial=0)

                for i in range(start_idx, len(df), args.batch_size):
                    end_idx = min(i + args.batch_size, len(df))
                    batch_smiles = df.iloc[i:end_idx]['smiles'].tolist()

                    batch_results = list(executor.map(get_npclassifier_predictions, batch_smiles))

                    for res in batch_results:
                        p, s, c = res
                        results['npc_pathway'].append(p)
                        results['npc_superclass'].append(s)
                        results['npc_class'].append(c)

                    current_df = df.iloc[:end_idx].copy()
                    current_df['npc_pathway'] = results['npc_pathway']
                    current_df['npc_superclass'] = results['npc_superclass']
                    current_df['npc_class'] = results['npc_class']
                    current_df.to_csv(output_path, index=False)

                    pbar.update(len(batch_smiles))

                    time.sleep(0.5)

                pbar.close()

        except KeyboardInterrupt:
            print("\nParado pelo user. Checkpoint salvo.")

        print("Feito!")


if __name__ == "__main__":
    main()