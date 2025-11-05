# IoT Energy Consumption — Multi-Stage ML Pipeline

This repository contains all source code used for the work of "IoT Energy Consumption — Multi-Stage ML Pipeline".

For more information contact: edllyn@ime.usp.br


## Table of Contents

1. [Introduction](#1-introduction)  
   1.1 [Directory Organization](#11-directory-organization)  
   1.2 [Dragon_Pi Dataset](#12-dragon_pi-dataset)  
   1.3 [Python Environment](#13-python-environment)  
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)  
3. [Running the Pipeline](#3-running-the-pipeline)  
   3.1 [Parameters and Examples](#31-parameters-and-examples)  
   3.2 [Stage: Training Size Estimation](#32-stage-training-size-estimation)  
   3.3 [Outputs and Results](#33-outputs-and-results)  
4. [Model Structure and Abbreviations](#4-model-structure-and-abbreviations)  
5. [Best Practices and Quick Troubleshooting](#5-best-practices-and-quick-troubleshooting)  
6. [Citation](#6-citation)

## 1. Getting Started

Follow these steps to replicate our data organization and Python environment:

### Directory Organization

#### Multi-Stage Pipeline Structure

Execute the following script to create the data directories:
```shell
cd ./scripts
bash run_create_dirs.sh
```

The following directories will be created:
```
.
├── multistage_pipeline/
│   ├── pipeline.py
│   ├── data_ingestion.py
│   ├── feature_selection.py
│   ├── model_selection.py
│   ├── training_size.py
│   └── utils.py
├── scripts/
│   ├── execute_pipeline.py
│   ├── generate_feature_files.py
│   ├── run_create_dirs.sh
│   ├── run_save_feature_files_train.sh
│   └── run_save_feature_files_test.sh
├── training/
│   └── df_windows.csv
├── models/
├── results/
│   ├── evaluation/
│   ├── training_size/
│   └── plots/
└── dragon_pi/
    ├── dragon/
    │   ├── dragon_bruteforce_large/
    │   ├── dragon_portscan_large/
    │   ├── dragon_ctfs_large/
    │   └── dragon_dos_large/
    │       └── dragon_synflood.csv
    └── pi/
        ├── pi_bruteforce_large/
        ├── pi_portscan_large/
        ├── pi_ctfs_large/
        └── pi_dos_large/
            └── pi_synflood.csv
```

### Configuring the Dragon_Pi dataset

The `Dragon_Pi` dataset are publicly available at the following location: https://zenodo.org/records/10784947

### Replicating Python Environment (with Conda and GPU Support)

#### Create and Activate Mamba Environment

```shell
conda create -n pipeline python=3.10
source /opt/conda/etc/profile.d/conda.sh
conda activate pipeline
```

#### Install Pip dependencies

For the `multi-stage pipeline` environment, use the following command:
```shell

```

## 2. Exploratory Data Analysis (EDA)

We conducted an Exploratory Data Analysis (EDA) of the Dragon_pi dataset to further understand its structure. All insights are available at the following notebook `notebooks/exploratory_data_analysis_eda.ipynb`.

## 3. Running the Pipeline

To execute the full **multi-stage ML pipeline** for energy-consumption–based intrusion detection, simply run:

> In our experiments, we used a system with an **Intel Core i7-8665U** for local runs and a **GPU-enabled node** (Ubuntu 24.04 with Conda, CUDA, and TensorFlow) on the USP cluster for deep learning tests.  
> At least **16 GB RAM** is recommended for CPU-based runs, or **8 GB GPU memory** for LSTM training.

```bash
python -m multistage_pipeline.pipeline \
    --attack ALL \
    --model ALL \
    --feature_selection mutual_info \
    --estimate_size \
    --regenerate
```

### Instructions

#### Sufficient Training Size

The pipeline can estimate the minimum training proportion automatically:

```bash
python3 -m multistage_pipeline.pipeline --estimate_size
```

A chart and CSV will be generated under: `results/training_size/`. 
The proportion where accuracy stabilizes (Δ ≈ 0) indicates the sufficient training size.

#### Results

After running the pipeline:
- **First Stage - Dataset generation**
    - Output dataset: `training/df_windows.csv`
    - Class distribution printed in console.
- **Second Stage - Training size estimation**
    - Plot: `results/training_size/metrics_<attack>.png`
    - CSV: `results/training_size/training_size_results_<attack>.csv`
- **Third Stage - Model training and evaluation**
    - Metrics CSV: `results/evaluation/results.csv`
    - Comparative plot: `results/plots/metrics_comparison_<attack>.png`
    - Trained models: `models/{AttackType}_{Model}.pkl` or `.h5` for LSTM

## 4. Citation

To be published.
