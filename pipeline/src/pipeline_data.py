#!/usr/bin/env python3
"""
pipeline_data.py

Generates windowed datasets (mean/std/min/max/current, label) for ML training.

Features:
- Builds windowed datasets from raw CSVs (attack or normal)
- Creates both individual attack datasets and combined ones
- Skips already generated datasets unless --force is passed
- Designed for large-scale efficient streaming (chunked CSV reading)

Usage:
    python pipeline_data.py                 # process only missing datasets
    python pipeline_data.py --force         # recreate all datasets
"""

import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# === Configuration ===
BASE_PATH = Path("../dragon_pi")
OUTPUT_FOLDER = Path("../training")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Windowing parameters
WINDOW_SIZE = 1024
HOP_SIZE = 512

# === Dataset definitions ===
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
    "normal": [
        BASE_PATH / "dragon" / "dragon_norm_large",
        BASE_PATH / "pi" / "pi_norm_large",
    ]
}

# === Derived datasets ===
DATASET_FOLDERS["dragon_all"] = [
    BASE_PATH / "dragon" / "dragon_bruteforce_large",
    BASE_PATH / "dragon" / "dragon_dos_large",
    BASE_PATH / "dragon" / "dragon_ctfs_large",
    BASE_PATH / "dragon" / "dragon_portscan_large",
]

DATASET_FOLDERS["pi_all"] = [
    BASE_PATH / "pi" / "pi_bruteforce_large",
    BASE_PATH / "pi" / "pi_dos_large",
    BASE_PATH / "pi" / "pi_ctf_large",
    BASE_PATH / "pi" / "pi_portscan_large",
]

DATASET_FOLDERS["dragon_pi_all"] = (
    DATASET_FOLDERS["dragon_all"] + DATASET_FOLDERS["pi_all"]
)

# === Helpers ===
def list_csv_files(folder: Path):
    """Return all CSV files in folder, skipping legend files."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.glob("*.csv") if "_legend" not in p.name.lower()])


def stream_csv_chunks(file_path: Path, chunksize: int = 200_000):
    """Yield pandas chunks from CSV file, handling read errors gracefully."""
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            yield chunk
    except Exception as exc:
        print(f"[ERROR] Failed to read {file_path.name}: {exc}")


def process_dataset(dataset_name: str, folders: list[Path]):
    """Process multiple CSV files into a windowed dataset saved to OUTPUT_FOLDER."""
    start_time = time.perf_counter()
    print(f"\n[START] Dataset '{dataset_name}'")
    print(f"[INFO] Window size={WINDOW_SIZE}, hop={HOP_SIZE}")
    print(f"[INFO] Scanning {len(folders)} folders...")

    # Collect and deduplicate CSV paths
    all_files = []
    for folder in folders:
        files = list_csv_files(folder)
        if not files:
            print(f"[WARNING] No CSVs found in {folder}")
        else:
            print(f"[INFO] {len(files)} CSV(s) found in {folder.name}")
            all_files.extend(files)

    seen = set()
    unique_files = []
    for f in all_files:
        resolved = str(f.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(f)

    print(f"[INFO] Total unique CSVs for '{dataset_name}': {len(unique_files)}")

    rows = []
    chunks_processed = windows_generated = skipped_missing_cols = 0

    for file_idx, file_path in enumerate(unique_files, 1):
        print(f"\n[INFO] Reading file {file_idx}/{len(unique_files)}: {file_path.name}")
        for chunk in tqdm(stream_csv_chunks(file_path), desc=f"Chunks {file_path.name}", leave=False):
            chunks_processed += 1
            chunk.columns = [c.strip().lower() for c in chunk.columns]

            # Required columns
            if "anno_type" not in chunk.columns or "current" not in chunk.columns:
                print(f"[WARNING] Skipping chunk (missing cols): {list(chunk.columns)}")
                skipped_missing_cols += 1
                continue

            chunk["is_attack"] = (chunk["anno_type"].astype(str).str.lower() != "normal").astype(int)
            n_rows = len(chunk)
            if n_rows < WINDOW_SIZE:
                continue

            # Sliding windows
            for start in range(0, n_rows - WINDOW_SIZE + 1, HOP_SIZE):
                end = start + WINDOW_SIZE
                window = chunk.iloc[start:end]
                time_mean = (
                    window["time"].mean()
                    if "time" in window.columns
                    else float(start)
                )
                rows.append({
                    "Time_mean": time_mean,
                    "Current_mean": window["current"].mean(),
                    "Current_std": window["current"].std(),
                    "Current_min": window["current"].min(),
                    "Current_max": window["current"].max(),
                    "label": 1 if window["is_attack"].mean() > 0.5 else 0
                })
                windows_generated += 1

    # Save or skip
    if not rows:
        print(f"[WARNING] No windows generated for '{dataset_name}'. Skipping save.")
        elapsed = time.perf_counter() - start_time
        print(f"[END] {dataset_name} finished in {elapsed:.1f}s — no data.")
        return None

    df = pd.DataFrame(rows)
    output_path = OUTPUT_FOLDER / f"df_windows_{dataset_name}.csv"
    df.to_csv(output_path, index=False)

    elapsed = time.perf_counter() - start_time
    print(f"\n[END] Dataset '{dataset_name}' done in {elapsed:.1f}s")
    print(f"[STATS] chunks={chunks_processed}, windows={windows_generated}, skipped_chunks={skipped_missing_cols}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts(dropna=False)}")
    print(f"[SUCCESS] Saved {len(df)} windows to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Windowing pipeline for Dragon/Pi datasets.")
    parser.add_argument("--force", action="store_true", help="Recreate all datasets even if they exist.")
    args = parser.parse_args()

    print("[INFO] Multi-dataset windowing started.")
    overall_start = time.perf_counter()

    processed = {}
    for dataset_name, folders in DATASET_FOLDERS.items():
        output_path = OUTPUT_FOLDER / f"df_windows_{dataset_name}.csv"
        if output_path.exists() and not args.force:
            print(f"[SKIP] {dataset_name} already exists. Use --force to overwrite.")
            processed[dataset_name] = None
            continue

        df = process_dataset(dataset_name, folders)
        processed[dataset_name] = df

    elapsed = time.perf_counter() - overall_start
    created = [k for k, v in processed.items() if v is not None]
    print(f"\n[SUMMARY] Created {len(created)} dataset(s): {created}")
    print(f"[SUMMARY] Total time: {elapsed:.1f}s")
    print(f"[INFO] Saved in: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
