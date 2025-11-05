import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from multistage_pipeline.utils import record_metrics, append_results, plot_comparison, log_info, ensure_dirs


def train_models(df, selected_models, attack_type, results_path="./results/evaluation/results.csv"):
    if df.empty:
        raise ValueError("O DataFrame fornecido está vazio. Verifique o arquivo de entrada.")

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results/evaluation", exist_ok=True)

    X = df.drop(columns=["label", "AttackType", "Filename"], errors="ignore")
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classical_models = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "NB": GaussianNB(),
        "XGB": XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        ),
    }

    results = []

    for name, model in classical_models.items():
        if "ALL" not in selected_models and name not in selected_models:
            continue
        log_info(f"Treinando {name} para {attack_type}...")
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - start
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }
        df_res = record_metrics(name, metrics, attack_type, elapsed)
        results.append(df_res)
        joblib.dump(model, f"./models/{attack_type}_{name}.pkl")

    if "ALL" in selected_models or "LSTM" in selected_models:
        log_info(f"Treinando LSTM para {attack_type}...")
        X_train_lstm = np.expand_dims(X_train, 2)
        X_test_lstm = np.expand_dims(X_test, 2)
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
        model = Sequential([
            LSTM(64, input_shape=(X_train_lstm.shape[1], 1)),
            Dense(32, activation="relu"),
            Dense(y_train_cat.shape[1], activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        start = time.time()
        model.fit(X_train_lstm, y_train_cat, epochs=10, batch_size=32, verbose=0)
        _, acc = model.evaluate(X_test_lstm, y_test_cat, verbose=0)
        elapsed = time.time() - start
        df_res = record_metrics("LSTM", {"accuracy": acc}, attack_type, elapsed)
        results.append(df_res)
        model.save(f"./models/{attack_type}_LSTM.h5")

    df_all = pd.concat(results, ignore_index=True)
    append_results(results_path, df_all)
    plot_comparison(df_all, attack_type)
    return df_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treinamento de modelos para detecção de ataques.")
    parser.add_argument("--attack", required=True, help="Tipo de ataque (ex: bruteforce, dos, portscan)")
    parser.add_argument("--model", default="ALL", help="Modelo específico ou 'ALL' para todos.")
    args = parser.parse_args()

    ensure_dirs()
    attack = args.attack.lower()
    model = args.model.upper()

    input_path = f"training/df_windows_{attack}.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    log_info(f"Lendo dataset: {input_path}")
    df = pd.read_csv(input_path)
    log_info(f"Dimensões do dataset: {df.shape}")

    selected_models = [model] if model != "ALL" else ["ALL"]
    df_results = train_models(df, selected_models, attack)
    log_info(f"Resultados salvos em ./results/evaluation/")
