import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from multistage_pipeline.utils import log_info, ensure_dirs


def apply_feature_selection(df: pd.DataFrame, method="mutual_info", k=10):
    if "label" not in df.columns:
        raise ValueError("O dataset precisa conter a coluna 'label' (0/1).")

    X = df.drop(columns=["label", "AttackType", "Filename"], errors="ignore")
    y = df["label"].astype(int)

    if X.empty:
        raise ValueError("Nenhuma coluna numérica encontrada para seleção de features.")

    if method == "anova":
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=k)

    selector.fit(X, y)
    selected_cols = X.columns[selector.get_support()]

    log_info(f"Selecionadas {len(selected_cols)} features: {list(selected_cols)[:5]}...")
    df_selected = df[selected_cols.tolist() + ["label"]]
    return df_selected


def main(args):
    attack = args.attack.lower()
    method = args.method.lower()
    k = args.k
    seed = args.seed

    input_path = Path("training") / f"df_windows_{attack}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {input_path.resolve()}")

    log_info(f"Lendo dataset: {input_path}")
    df = pd.read_csv(input_path)
    log_info(f"Dimensões originais: {df.shape}")

    df_selected = apply_feature_selection(df, method=method, k=k)

    output_dir = Path("results/feature_selection/features")
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"selected_features_{attack}_{method}_k{k}_seed{seed}.csv"

    df_selected.to_csv(output_file, index=False)
    log_info(f"Dataset salvo em: {output_file.resolve()}")
    log_info(f"Dimensões após seleção: {df_selected.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seleção de features simples para datasets Dragon_Pi")
    parser.add_argument("--attack", required=True, help="Tipo de ataque (ex: bruteforce, dos, portscan)")
    parser.add_argument("--method", default="mutual_info", choices=["mutual_info", "anova"], help="Método de seleção")
    parser.add_argument("--k", type=int, default=10, help="Número de features a selecionar")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória")
    args = parser.parse_args()

    main(args)
