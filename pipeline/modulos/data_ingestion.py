import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# === Configuration ===
BASE_PATH = Path("../dragon_pi")
OUTPUT_FOLDER = Path("../training")
RESULTS_FOLDER = Path("../results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Windowing parameters
WINDOW_SIZE = 1024
HOP_SIZE = 512

# Map dataset names to folders
DATASET_FOLDERS = {
    "bruteforce": [
        BASE_PATH / "dragon" / "dragon_bruteforce_large",
        BASE_PATH / "pi" / "pi_bruteforce_large",
    ],
    "dos": [
        BASE_PATH / "dragon" / "dragon_dos_large",
        BASE_PATH / "pi" / "pi_dos_large",
    ],
    "ctf": [
        BASE_PATH / "dragon" / "dragon_ctfs_large",
        BASE_PATH / "pi" / "pi_ctf_large",
    ],
    "portscan": [
        BASE_PATH / "dragon" / "dragon_portscan_large",
        BASE_PATH / "pi" / "pi_portscan_large",
    ],
}

# Derived/combined datasets
DATASET_FOLDERS["bruteforce_dos"] = DATASET_FOLDERS["bruteforce"] + DATASET_FOLDERS["dos"]
DATASET_FOLDERS["all_attacks"] = (
    DATASET_FOLDERS["bruteforce"]
    + DATASET_FOLDERS["dos"]
    + DATASET_FOLDERS["ctf"]
    + DATASET_FOLDERS["portscan"]
)


def list_csv_files(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.glob("*.csv") if "_legend" not in p.name.lower()])


def stream_csv_chunks(file_path: Path, chunksize: int = 200_000):
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            yield chunk
    except Exception as exc:
        print(f"[ERROR] Failed to read {file_path.name}: {exc}")


def process_dataset(dataset_name: str, folders: list[Path]):
    start_time = time.perf_counter()
    print(f"\n[START] Dataset '{dataset_name}'")
    print(f"[INFO] Window size: {WINDOW_SIZE}, Hop size: {HOP_SIZE}")
    print(f"[INFO] Folders to scan ({len(folders)}):")
    for f in folders:
        print(f"  - {f}")

    all_files = []
    for folder in folders:
        files_in_folder = list_csv_files(folder)
        if not files_in_folder:
            print(f"[WARNING] No CSV files found in {folder}")
        else:
            print(f"[INFO] Found {len(files_in_folder)} CSV file(s) in {folder.name}")
            all_files.extend(files_in_folder)

    seen = set()
    unique_files = []
    for f in all_files:
        if str(f.resolve()) not in seen:
            seen.add(str(f.resolve()))
            unique_files.append(f)

    print(f"[INFO] Total unique CSV files to process for '{dataset_name}': {len(unique_files)}")

    rows = []
    files_processed = 0
    chunks_processed = 0
    windows_generated = 0
    files_skipped_missing_cols = 0

    for file_path in unique_files:
        files_processed += 1
        print(f"\n[INFO] Reading file {files_processed}/{len(unique_files)}: {file_path.name}")
        for chunk in tqdm(stream_csv_chunks(file_path), desc=f"Chunks {file_path.name}", leave=False):
            chunks_processed += 1
            chunk.columns = [c.strip().lower() for c in chunk.columns]

            if "anno_type" not in chunk.columns or "current" not in chunk.columns:
                print(f"[WARNING] Skipping chunk from {file_path.name} due to missing columns: {list(chunk.columns)}")
                files_skipped_missing_cols += 1
                continue

            chunk["is_attack"] = (chunk["anno_type"].astype(str).str.lower() != "normal").astype(int)

            n_rows = len(chunk)
            if n_rows < WINDOW_SIZE:
                continue

            for start in range(0, n_rows - WINDOW_SIZE + 1, HOP_SIZE):
                end = start + WINDOW_SIZE
                window = chunk.iloc[start:end]
                rows.append({
                    "Current_mean": window["current"].mean(),
                    "Current_std": window["current"].std(),
                    "Current_min": window["current"].min(),
                    "Current_max": window["current"].max(),
                    "label": 1 if window["is_attack"].mean() > 0.5 else 0
                })
                windows_generated += 1

    if not rows:
        print(f"[WARNING] No windows generated for dataset '{dataset_name}'. No file will be saved.")
        elapsed = time.perf_counter() - start_time
        print(f"[END] Dataset '{dataset_name}' finished in {elapsed:.1f}s — windows: 0")
        return None

    df = pd.DataFrame(rows)
    output_name = f"df_windows_{dataset_name}.csv"
    training_path = OUTPUT_FOLDER / output_name
    results_path = RESULTS_FOLDER / output_name

    df.to_csv(training_path, index=False)
    df.to_csv(results_path, index=False)

    elapsed = time.perf_counter() - start_time
    print(f"\n[END] Dataset '{dataset_name}' processed.")
    print(f"[STATS] files_checked={len(unique_files)}, files_skipped_missing_cols={files_skipped_missing_cols}, chunks={chunks_processed}, windows_generated={windows_generated}")
    print(f"[INFO] Label distribution for '{dataset_name}':\n{df['label'].value_counts(dropna=False)}")
    print(f"[SUCCESS] Saved {len(df)} windows to {training_path} and {results_path} (time: {elapsed:.1f}s)")

    return df


def main():
    print("[INFO] Multi-dataset windowing pipeline started.")
    overall_start = time.perf_counter()

    processed = {}
    for dataset_name, folders in DATASET_FOLDERS.items():
        df = process_dataset(dataset_name, folders)
        processed[dataset_name] = df

    overall_elapsed = time.perf_counter() - overall_start
    created = [k for k, v in processed.items() if v is not None]
    print(f"\n[SUMMARY] Completed. Datasets created: {len(created)} -> {created}")
    print(f"[SUMMARY] Total pipeline time: {overall_elapsed:.1f}s")
    print("[INFO] Files saved in:", OUTPUT_FOLDER.resolve())
    print("[INFO] Results also saved in:", RESULTS_FOLDER.resolve())


if __name__ == "__main__":
    main()
