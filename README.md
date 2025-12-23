# Energy-Based Intrusion Detection for IoT Devices Using a Statistical Threshold

This repository contains all the source code used for the work of "Energy-Based Intrusion Detection for IoT Devices Using a Statistical Threshold".

For more information, contact: edllyn@ime.usp.br


===
## 1. Getting Started

Follow these steps to replicate our data organization and Python environment:

### Directory Organization

### Configuring the Dragon_Pi dataset

The `Dragon_Pi` dataset is publicly available at the following location: https://zenodo.org/records/10784947

### Replicating Python Environment (with Conda)

#### Create and Activate Conda Environment
> In our experiments, we used a system with an **Intel Xeon Gold 6148** for local runs and a **GPU-enabled node** (Ubuntu 24.04 with Conda, CUDA, and TensorFlow) on the USP cluster for deep learning tests.

```shell
conda create -n notebook python=3.10
source /opt/conda/etc/profile.d/conda.sh
conda activate notebook
```

#### Install Pip dependencies

For the environment, use the following command:
```shell
pip install -r requirements.txt
```

## 2. Exploratory Data Analysis (EDA)

We conducted an Exploratory Data Analysis (EDA) of the Dragon_pi dataset to further understand its structure. All insights are available at the following notebook `pipeline/notebooks`.


## 3. Citation

To be published.
