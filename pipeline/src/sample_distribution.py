import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_PATH = Path("../dragon_pi") 

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
        BASE_PATH / "pi" / "pi_ctfs_large",
    ],
    "portscan": [
        BASE_PATH / "dragon" / "dragon_portscan_large",
        BASE_PATH / "pi" / "pi_portscan_large",
    ],
    "normal": [
        BASE_PATH / "dragon" / "dragon_norm_large",
        BASE_PATH / "pi" / "pi_norm_large",
    ],
}

def list_csv_files(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.glob("*.csv") if "_legend" not in p.name.lower()])

counts = {"Anomaly": 0, "Normal": 0}
total_files = sum(len(list_csv_files(f)) for lst in DATASET_FOLDERS.values() for f in lst)
pbar = tqdm(total=total_files, desc="Processing CSV files", ncols=100)

for attack_type, folders in DATASET_FOLDERS.items():
    for folder in folders:
        for file in list_csv_files(folder):
            try:
                df = pd.read_csv(file)
                if "anno_string" not in df.columns:
                    pbar.update(1)
                    continue
                counts["Anomaly"] += (df["anno_string"] != "Normal").sum()
                counts["Normal"] += (df["anno_string"] == "Normal").sum()
            except Exception as e:
                print(f"[WARN] Error reading {file}: {e}")
            pbar.update(1)

pbar.close()

total = counts["Anomaly"] + counts["Normal"]
if total == 0:
    print("Nenhum registro encontrado. Verifique BASE_PATH e coluna 'anno_string'.")
    exit()

table = pd.DataFrame({
    "Type": ["Anomaly", "Normal", "Total"],
    "Count": [counts["Anomaly"], counts["Normal"], total],
    "Percentage": [
        round(100 * counts["Anomaly"] / total, 1),
        round(100 * counts["Normal"] / total, 1),
        100.0
    ]
})

print("\n--- Distribution Summary ---")
print(table.to_string(index=False))