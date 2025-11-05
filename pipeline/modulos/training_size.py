import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from multistage_pipeline.utils import log_info


def cross_val_accuracy(clf, X, y, folds=3):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accs = []
    for tr_idx, val_idx in skf.split(X, y):
        clf.fit(X[tr_idx], y[tr_idx])
        preds = clf.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
    return accs


def estimate_training_size(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("[ERRO] O dataframe está vazio.")
    if "label" not in df.columns:
        raise ValueError("[ERRO] O dataset não contém a coluna 'label'.")

    X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
    y = df["label"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if X.empty:
        raise ValueError("[ERRO] Nenhuma coluna numérica encontrada.")

    proportions = np.linspace(0.1, 0.9, 9)
    results = []

    log_info("Iniciando estimativa de tamanho de treino...")
    unique_classes = np.unique(y)
    use_stratify = len(unique_classes) > 1

    for p in tqdm(proportions, desc="Estimando tamanho de treino", unit="proporção"):
        try:
            X_train, _, y_train, _ = train_test_split(
                X, y, train_size=p, random_state=seed,
                stratify=y if use_stratify else None
            )

            X_train_s = StandardScaler().fit_transform(X_train)
            clf = RandomForestClassifier(n_estimators=50, random_state=seed)
            start = time.perf_counter()
            clf.fit(X_train_s, y_train)
            elapsed = time.perf_counter() - start

            acc = np.mean(cross_val_accuracy(clf, X_train_s, y_train))
            results.append({
                "TrainSize(%)": round(p * 100, 1),
                "Accuracy": round(acc, 4),
                "Time(s)": round(elapsed, 2)
            })
            tqdm.write(f"{p*100:.0f}% concluído → Acurácia={acc:.3f} Tempo={elapsed:.2f}s")

        except Exception as e:
            tqdm.write(f"[WARN] Falha em {p:.2f}: {type(e).__name__}: {e}")

    df_res = pd.DataFrame(results)
    print("\n=== Resultados finais ===")
    print(df_res.to_string(index=False))

    out_dir = Path("results/training_size")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_size_results.csv"
    df_res.to_csv(out_path, index=False)

    log_info(f"Resultados salvos em {out_path}")
    return df_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimativa de tamanho de treino ideal.")
    parser.add_argument("--data", type=str, default="training/df_windows.csv", help="Caminho para o dataset df_windows.csv")
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatória")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Arquivo não encontrado: {args.data}")

    log_info(f"Lendo dataset de {args.data}")
    df = pd.read_csv(args.data)
    estimate_training_size(df, seed=args.seed)
