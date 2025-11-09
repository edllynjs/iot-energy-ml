#!/usr/bin/env python3
"""
model_training.py

Train multiple ML models (classical + LSTM) over one or more windowed datasets.

Features:
- Discovers or selects dataset files under ../training (pattern: df_windows_*.csv)
- For each dataset:
    * Trains specified model(s) (or ALL)
    * Saves models, scalers, and detailed classification reports
    * Writes per-dataset results CSV immediately after each training

Usage:
    python model_training.py                      # trains ALL models on ALL datasets
    python model_training.py --model RandomForest # trains only RandomForest on ALL datasets
    python model_training.py --attack dos         # trains ALL models on df_windows_dos.csv
    python model_training.py --model LSTM --attack bruteforce # specific combo
"""

import os, argparse, joblib, time
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ===========================
# Configuration
# ===========================

TRAINING_DIR = Path("../training")
MODEL_DIR = Path("../models")
RESULTS_DIR = Path("../results")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOT_DIR = RESULTS_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_GLOB = "df_windows_*.csv"
TRAIN_SIZE = 0.7
RANDOM_STATE = 42
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 32

MODEL_NAMES = ["LogisticRegression", "KNN", "DecisionTree", "RandomForest", "NaiveBayes", "XGBoost"]

# ===========================
# Model Hyperparameters by Device and Attack
# ===========================
# Device-level defaults
MODEL_PARAMS_DEVICE = {
    "dragon": {
        "RandomForest": {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5},
        "KNN": {"n_neighbors": 3},
        "XGBoost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8},
        "DecisionTree": {"max_depth": None},
        "LogisticRegression": {"max_iter": 1000},
        "NaiveBayes": {}
    },
    "pi": {
        "RandomForest": {"n_estimators": 150, "max_depth": 15},
        "KNN": {"n_neighbors": 7},
        "XGBoost": {"n_estimators": 250, "learning_rate": 0.08, "max_depth": 6},
        "DecisionTree": {"max_depth": None},
        "LogisticRegression": {"max_iter": 1000},
        "NaiveBayes": {}
    },
    "dragon_pi": {
        "RandomForest": {"n_estimators": 400, "max_depth": 25},
        "KNN": {"n_neighbors": 5},
        "XGBoost": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 10},
        "DecisionTree": {"max_depth": None},
        "LogisticRegression": {"max_iter": 1000},
        "NaiveBayes": {}
    },
    "default": {
        "RandomForest": {"n_estimators": 100},
        "KNN": {"n_neighbors": 5},
        "XGBoost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
        "DecisionTree": {"max_depth": None},
        "LogisticRegression": {"max_iter": 1000},
        "NaiveBayes": {}
    }
}

# Attack-level overrides (these apply on top of device-level params when present)
MODEL_PARAMS_ATTACK = {
    "dos": {
        "RandomForest": {"n_estimators": 300, "max_depth": 30},
        "KNN": {"n_neighbors": 3},
    },
    "bruteforce": {
        "RandomForest": {"n_estimators": 250, "max_depth": 18},
        "KNN": {"n_neighbors": 3},
    },
    "portscan": {
        "RandomForest": {"n_estimators": 200, "max_depth": 15},
        "KNN": {"n_neighbors": 5},
    },
    "ctf": {
        "RandomForest": {"n_estimators": 150, "max_depth": 12},
    },
    # 'norm' and 'all' will fallback to device defaults (no overrides)
}

def parse_dataset_keys(dataset_name: str):
    """
    Infer device_key and attack_key from dataset_name.
    Examples:
      'dragon_bruteforce_large' -> device='dragon', attack='bruteforce'
      'pi_all' -> device='pi', attack='all'
      'dragon_pi_all' -> device='dragon_pi', attack='all'
    """
    tokens = dataset_name.lower().split('_')
    device = "default"
    if "dragon" in tokens and "pi" in tokens:
        device = "dragon_pi"
    elif tokens and tokens[0] == "dragon":
        device = "dragon"
    elif tokens and tokens[0] == "pi":
        device = "pi"
    else:
        # fallback heuristic: if any token matches device names
        if "dragon" in tokens:
            device = "dragon"
        elif "pi" in tokens:
            device = "pi"

    # detect attack token
    attacks_known = list(MODEL_PARAMS_ATTACK.keys()) + ["norm", "all"]
    attack = next((t for t in tokens if t in attacks_known), "all")
    return device, attack

def merge_params(device: str, attack: str, model_name: str):
    """
    Merge device-level params with attack-level overrides for a given model.
    Attack overrides take precedence.
    """
    device_params = MODEL_PARAMS_DEVICE.get(device, MODEL_PARAMS_DEVICE["default"]).get(model_name, {})
    attack_params = MODEL_PARAMS_ATTACK.get(attack, {}).get(model_name, {})
    merged = {}
    merged.update(device_params or {})
    merged.update(attack_params or {})
    return merged

def get_model_instance(model_name, dataset_name):
    device_key, attack_key = parse_dataset_keys(dataset_name)
    params = merge_params(device_key, attack_key, model_name)

    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params)
    elif model_name == "XGBoost":
        # ensure eval_metric present for compatibility
        if "eval_metric" not in params:
            params = dict(params)
            params["eval_metric"] = "logloss"
        return XGBClassifier(random_state=RANDOM_STATE, **params)
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    elif model_name == "LogisticRegression":
        # use device/attack params only for max_iter if provided; keep default otherwise
        lr_params = {}
        if "max_iter" in params:
            lr_params["max_iter"] = params["max_iter"]
        return LogisticRegression(random_state=RANDOM_STATE, **lr_params)
    elif model_name == "NaiveBayes":
        return GaussianNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===========================
# Utility Functions
# ===========================

def discover_datasets(training_dir: Path, pattern: str = DATASET_GLOB):
    """Return list of (dataset_name, path) for files matching df_windows_*.csv."""
    datasets = []
    for f in sorted(training_dir.glob(pattern)):
        if f.name.startswith("df_windows_") and f.suffix == ".csv":
            dataset_name = f.stem.replace("df_windows_", "")
            datasets.append((dataset_name, f))
    return datasets


def append_or_create_csv(path: Path, df: pd.DataFrame):
    """Append DataFrame to CSV, or create it if not exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            prev = pd.read_csv(path)
            combined = pd.concat([prev, df], ignore_index=True)
            combined.to_csv(path, index=False)
        except Exception as e:
            print(f"[WARNING] Could not append to {path}, overwriting. Reason: {e}")
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def safe_write_text(path: Path, text: str):
    """Safely write plain text to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)

# ===========================
#  Detection metrics 
# ===========================

def detection_metrics(y_true, y_pred, timestamps=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    delay = None
    if timestamps is not None and np.any(y_true == 1):
        try:
            attack_indices = np.where(y_true == 1)[0]
            first_attack_idx = attack_indices[0]
            detection_idx = next((i for i in range(first_attack_idx, len(y_pred)) if y_pred[i] == 1), None)
            if detection_idx is not None:
                delay = float(timestamps[detection_idx] - timestamps[first_attack_idx])
        except Exception:
            delay = None
    return dr, far, delay

# ===========================
# Training Functions
# ===========================

def train_classical_model(model, X_train, y_train, X_test, y_test):
    """Train a classical ML model and return accuracy, report, and training duration."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    duration = time.perf_counter() - start
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=3, zero_division=0)
    timestamps = None
    if "Time_mean" in X_test.columns:
        timestamps = X_test["Time_mean"].values
    dr, far, delay = detection_metrics(y_test, preds, timestamps)
    return acc, dr, far, delay, report, duration


def train_lstm_model(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train LSTM model, returning accuracy, report, duration, and trained model."""
    X_train_lstm = np.expand_dims(X_train_scaled, axis=2)
    X_test_lstm = np.expand_dims(X_test_scaled, axis=2)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], 1)),
        Dense(32, activation='relu'),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.perf_counter()
    model.fit(X_train_lstm, y_train_cat, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=1)
    duration = time.perf_counter() - start

    loss, acc = model.evaluate(X_test_lstm, y_test_cat, verbose=0)
    preds = np.argmax(model.predict(X_test_lstm, verbose=0), axis=1)
    report = classification_report(y_test, preds, digits=4)

    return acc, report, duration, model


# ===========================
# Core Process Function
# ===========================

def process_dataset_file(dataset_name: str, csv_path: Path, selected_model_key: str = "ALL"):
    """
    Train selected model(s) on a dataset.
    Saves results immediately after training.
    """
    print(f"\n[DATASET START] {dataset_name} — {csv_path}")
    start_total = time.perf_counter()

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Dataset '{dataset_name}' missing 'label' column.")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        shuffle=False,
    )

    print(f"[INFO] Split: {len(y_train)} train / {len(y_test)} test")

    orig_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"[INFO] Original class distribution (train): {orig_dist}")

    split_info = pd.DataFrame({
        "Phase": ["Train", "Test"],
        "Count": [len(y_train), len(y_test)]
    })
    split_info.to_csv(PLOT_DIR / f"split_{dataset_name}.csv", index=False)

    orig_dist_df = pd.DataFrame({
        "Class": list(orig_dist.keys()),
        "Count": list(orig_dist.values())
    })
    orig_dist_df.to_csv(PLOT_DIR / f"SMOTE_{dataset_name}_origdist.csv", index=False)

    # SMOTE (class balancing)
    if len(np.unique(y_train)) > 1:
        sm = SMOTE(random_state=RANDOM_STATE)

        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        X_train, y_train = X_train_res, y_train_res
        print(f"[INFO] SMOTE applied. New class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

        smote_info = pd.DataFrame({
            "Class": np.unique(y_train),
            "Count": np.bincount(y_train)
        })
        smote_info.to_csv(PLOT_DIR / f"SMOTE_{dataset_name}_balanced.csv", index=False)
    else:
        print("[WARNING] Only one class in training set — skipping SMOTE.")
        

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / f"scaler_{dataset_name}.pkl")

    results_rows = []

    # Select models
    should_run_lstm = selected_model_key in ("ALL", "LSTM")
    classical_keys = (
        MODEL_NAMES if selected_model_key == "ALL"
        else [selected_model_key] if selected_model_key in MODEL_NAMES else []
    )

    # Train classical models
    for key in classical_keys:
        model = get_model_instance(key, dataset_name)
        model_file = MODEL_DIR / f"{dataset_name}_{key}_model.pkl"
        report_file = METRICS_DIR / f"{dataset_name}_{key}_report.txt"

        try:
            acc, dr, far, delay, report, dur = train_classical_model(model, X_train, y_train, X_test, y_test)

            start_pred = time.perf_counter()
            _ = model.predict(X_test)
            classification_time = (time.perf_counter() - start_pred) * 1000
            joblib.dump(model, model_file)
            safe_write_text(report_file, report)
            with report_file.open("a", encoding="utf-8") as f:
                f.write("\n")
                f.write("Additional detection metrics:\n")
                f.write(f"DetectionRate (DR): {dr:.3f}\n")
                f.write(f"FalseAlarmRate (FAR): {far:.3f}\n")
                delay_str = f"{delay:.3f}" if delay is not None else "N/A"
                f.write(f"Delay (s): {delay_str}\n")
                f.write(f"Classification Time (ms): {classification_time:.3f}\n")
            print(f"[RESULT] {dataset_name} | {key} | acc={acc:.3f} | dr={dr:.3f} | far={far:.3f} | delay={delay_str} | dur={dur:.1f}s | class_time={classification_time:.1f}ms")

            results_rows.append({
                "Dataset": dataset_name,
                "Model": key,
                "Accuracy": acc,
                "DurationSec": round(dur, 3),
                "ClassificationTimeMs": round(classification_time, 3),
                "Status": "OK",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ModelFile": model_file.name,
                "ReportFile": report_file.name,
            })
        except Exception as e:
            print(f"[ERROR] Failed training {key}: {e}")
            results_rows.append({
                "Dataset": dataset_name,
                "Model": key,
                "Accuracy": None,
                "DurationSec": None,
                "ClassificationTimeMs": None,
                "Status": f"FAILED: {e}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        # Save CSV incrementally after each model
        append_or_create_csv(RESULTS_DIR / f"training_results_{dataset_name}.csv",
                             pd.DataFrame(results_rows))
        results_rows.clear()

    # Train LSTM (if requested)
    if should_run_lstm:
        lstm_file = MODEL_DIR / f"{dataset_name}_LSTM_model.h5"
        report_file = METRICS_DIR / f"{dataset_name}_LSTM_report.txt"

        try:
            acc, report, dur, model = train_lstm_model(X_train_scaled, y_train, X_test_scaled, y_test)
            model.save(lstm_file)
            safe_write_text(report_file, report)
            with report_file.open("a", encoding="utf-8") as f:
                f.write("\n")
                f.write("Additional detection metrics:\n")
                f.write(f"DetectionRate (DR): {dr:.3f}\n")
                f.write(f"FalseAlarmRate (FAR): {far:.3f}\n")
                delay_str = f"{delay:.3f}" if delay is not None else "N/A"
                f.write(f"Delay (s): {delay_str}\n")
                f.write(f"Classification Time (ms): {classification_time:.3f}\n")
            print(f"[RESULT] {dataset_name} | LSTM | acc={acc:.3f} | dur={dur:.1f}s")

            lstm_result = pd.DataFrame([{
                "Dataset": dataset_name,
                "Model": "LSTM",
                "Accuracy": acc,
                "DurationSec": round(dur, 3),
                "Status": "OK",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ModelFile": lstm_file.name,
                "ReportFile": report_file.name,
            }])
            append_or_create_csv(RESULTS_DIR / f"training_results_{dataset_name}.csv", lstm_result)

        except Exception as e:
            print(f"[ERROR] LSTM training failed: {e}")
            lstm_result = pd.DataFrame([{
                "Dataset": dataset_name,
                "Model": "LSTM",
                "Accuracy": None,
                "DurationSec": None,
                "Status": f"FAILED: {e}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }])
            append_or_create_csv(RESULTS_DIR / f"training_results_{dataset_name}.csv", lstm_result)

    elapsed = time.perf_counter() - start_total
    print(f"[DATASET END] {dataset_name} completed in {elapsed:.1f}s\n")


# ===========================
# CLI Entry Point
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Train models on df_windows_<dataset>.csv files.")
    parser.add_argument("--model", type=str, default="ALL",
                        help="Model name or ALL. Options: " + ", ".join(MODEL_NAMES + ["LSTM"]))
    parser.add_argument("--attack", type=str, default="ALL",
                        help="Dataset suffix (e.g. 'dos', 'ctf', 'portscan') or ALL for all datasets.")
    args = parser.parse_args()

    selected_model = args.model
    selected_attack = args.attack

    datasets = discover_datasets(TRAINING_DIR)
    if not datasets:
        print(f"[ERROR] No datasets found in {TRAINING_DIR}. Exiting.")
        return

    if selected_attack != "ALL":
        datasets = [(n, p) for n, p in datasets if n.lower() == selected_attack.lower()]
        if not datasets:
            print(f"[ERROR] No dataset matches '--attack {selected_attack}'. Exiting.")
            return
        print(f"[INFO] Selected dataset: {selected_attack}")

    print(f"[INFO] Found {len(datasets)} dataset(s) to process: {[n for n, _ in datasets]}")
    start = time.perf_counter()

    for dataset_name, csv_path in datasets:
        process_dataset_file(dataset_name, csv_path, selected_model_key=selected_model)

    total_time = time.perf_counter() - start
    print(f"\n[ALL DONE] Total time: {total_time:.1f}s")
    print(f"[INFO] Results saved under: {RESULTS_DIR.resolve()}")
    print(f"[INFO] Models stored under: {MODEL_DIR.resolve()}")
    print(f"[INFO] Metrics stored under: {METRICS_DIR.resolve()}")
    print(f"[INFO] Splits/SMOTE stored under: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
