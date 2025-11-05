import os
import gc
import glob
import argparse
import polars as pl
import pandas as pd
from imblearn.over_sampling import SMOTE

# =========================
# Funções auxiliares
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def prepare_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cria colunas padronizadas a partir de anno_string:
    - attack_label__most_frequent
    - attack_label_enc__most_frequent
    - is_attack__most_frequent
    """
    if "anno_string" not in df.columns:
        raise ValueError("df_windows.csv precisa conter a coluna 'anno_string'.")

    df = df.with_columns([
        pl.col("anno_string").alias("attack_label__most_frequent"),
        pl.when(pl.col("anno_string").str.to_lowercase() == "normal")
          .then(0).otherwise(1).alias("is_attack__most_frequent")
    ])

    classes = sorted(df["anno_string"].unique().to_list())
    mapping = {v: i for i, v in enumerate(classes)}
    df = df.with_columns(
        pl.col("anno_string").map_elements(lambda x: mapping[x]).alias("attack_label_enc__most_frequent")
    )
    return df

def feature_columns(df: pl.DataFrame) -> list:
    ignore = {
        "index", "anno_string", "anno_type", "anno_specific",
        "attack_label__most_frequent", "attack_label_enc__most_frequent",
        "is_attack__most_frequent"
    }
    return [c for c in df.columns if c not in ignore and df[c].dtype.is_numeric()]

def create_splits(df: pl.DataFrame, save_dir: str, prefix: str, seed: int):
    ensure_dir(save_dir)
    n = df.height
    df = df.with_columns(pl.Series("index", range(n)))

    df_test = df.sample(n=int(n * 0.2), seed=seed)
    df_rest = df.join(df_test, on="index", how="anti")

    df_valid = df_rest.sample(n=int(n * 0.2), seed=seed)
    df_rest2 = df_rest.join(df_valid, on="index", how="anti")

    df_fs = df_rest2.sample(n=int(n * 0.05), seed=seed)
    df_train = df_rest2.join(df_fs, on="index", how="anti")

    paths = {
        "train": os.path.join(save_dir, f"{prefix}_train_preprocessed_seed{seed}.csv"),
        "valid": os.path.join(save_dir, f"{prefix}_valid_preprocessed_seed{seed}.csv"),
        "test": os.path.join(save_dir, f"{prefix}_test_preprocessed_seed{seed}.csv"),
        "fs": os.path.join(save_dir, f"{prefix}_fs_preprocessed_seed{seed}.csv"),
    }

    df_train.write_csv(paths["train"])
    df_valid.write_csv(paths["valid"])
    df_test.write_csv(paths["test"])
    df_fs.write_csv(paths["fs"])
    del df_train, df_valid, df_test, df_fs
    gc.collect()
    return paths

def balance_with_smote(train_path: str, prefix: str, seed: int):
    df = pl.read_csv(train_path)
    feats = feature_columns(df)
    pdf = df.to_pandas()
    X = pdf[feats]
    y = pdf["attack_label_enc__most_frequent"]

    X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
    out = pd.DataFrame(X_res, columns=feats)
    out["attack_label_enc__most_frequent"] = y_res

    mapping = pdf[["attack_label_enc__most_frequent", "attack_label__most_frequent"]].drop_duplicates().set_index("attack_label_enc__most_frequent")["attack_label__most_frequent"].to_dict()
    out["attack_label__most_frequent"] = out["attack_label_enc__most_frequent"].map(mapping)
    out["is_attack__most_frequent"] = (out["attack_label__most_frequent"].str.lower() != "normal").astype(int)

    out_path = os.path.join(os.path.dirname(train_path), f"{prefix}_train_preprocessed_smote_seed{seed}.csv")
    out.to_csv(out_path, index=False)
    return out_path

# =========================
# Pipelines
# =========================
def process_file(input_csv: str, save_dir: str, prefix: str, seed: int, apply_smote: bool):
    df = pl.read_csv(input_csv)
    df = prepare_labels(df)
    paths = create_splits(df, save_dir, prefix, seed)
    if apply_smote:
        balance_with_smote(paths["train"], prefix, seed)

def process_all(training_dir: str, seed: int, apply_smote: bool):
    """
    Lê:
      training/df_windows.csv          → global
      training/temp/*.csv              → individuais
    """
    global_path = os.path.join(training_dir, "df_windows.csv")
    temp_dir = os.path.join(training_dir, "temp")
    save_root = os.path.join(training_dir, "splits")
    ensure_dir(save_root)

    if os.path.exists(global_path):
        print(f"[INFO] Processando global: {global_path}")
        process_file(global_path, os.path.join(save_root, "GLOBAL"), "DRAGONPI_ALL", seed, apply_smote)

    temp_files = glob.glob(os.path.join(temp_dir, "*.csv"))
    for f in sorted(temp_files):
        attack_name = os.path.splitext(os.path.basename(f))[0].upper()
        print(f"[INFO] Processando ataque: {attack_name}")
        out_dir = os.path.join(save_root, attack_name)
        process_file(f, out_dir, attack_name, seed, apply_smote)

# =========================
# Execução CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processamento Dragon_Pi completo ou arquivo único")
    parser.add_argument("--training_dir", required=True, help="Diretório com df_windows.csv e temp/*.csv")
    parser.add_argument("--file", type=str, default=None, help="Caminho de um CSV específico (opcional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_smote", action="store_true")
    args = parser.parse_args()

    if args.file:
        print(f"[INFO] Processando arquivo único: {args.file}")
        save_dir = os.path.join(args.training_dir, "splits", "SINGLE")
        process_file(args.file, save_dir, "CUSTOM", args.seed, apply_smote=not args.no_smote)
    else:
        process_all(args.training_dir, args.seed, apply_smote=not args.no_smote)

    print("[DONE] Pré-processamento concluído.")
