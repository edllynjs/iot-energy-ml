import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# LOGGING
# ============================================================

def log_header(title: str):
    print("\n" + "=" * 80)
    print(f"{title.upper()}")
    print("=" * 80 + "\n")


def log_step(step: str):
    print(f"[STEP] {step}")


def log_info(msg: str):
    print(f"[INFO] {msg}")


def log_warn(msg: str):
    print(f"[WARN] {msg}")


def log_error(msg: str):
    print(f"[ERROR] {msg}")


# ============================================================
# PATH MANAGEMENT
# ============================================================

def ensure_dirs(path: str = None):
    """Cria as pastas necessárias para armazenar dados, modelos e resultados."""
    dirs = [
        "./training",
        "./models",
        "./results",
        "./results/training_size",
        "./results/evaluation",
        "./results/plots",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    log_info("Pastas criadas/verificadas com sucesso.")


# ============================================================
# METRICS HANDLING
# ============================================================

def append_results(results_path: str, new_df: pd.DataFrame):
    """Concatena novos resultados com resultados anteriores."""
    if os.path.exists(results_path):
        old = pd.read_csv(results_path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(results_path, index=False)
    log_info(f"Resultados salvos em: {results_path}")


def record_metrics(model_name: str, metrics: dict, attack: str, elapsed_time: float) -> pd.DataFrame:
    """Gera DataFrame padrão de métricas de avaliação."""
    df = pd.DataFrame([{
        "AttackType": attack,
        "Model": model_name,
        "Accuracy": metrics.get("accuracy", 0),
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "F1": metrics.get("f1", 0),
        "ExecTime(s)": round(elapsed_time, 2),
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }])
    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_comparison(df: pd.DataFrame, attack: str, save_path: str = "./results/plots"):
    """Cria gráfico comparando desempenho entre modelos."""
    if df.empty:
        log_warn("Nenhum dado para plotar.")
        return

    os.makedirs(save_path, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    melted = df.melt(
        id_vars=["Model"], value_vars=metrics,
        var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title(f"Model Comparison for {attack}", fontsize=14)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    output_file = os.path.join(save_path, f"metrics_comparison_{attack}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    log_info(f"Gráfico salvo em: {output_file}")
