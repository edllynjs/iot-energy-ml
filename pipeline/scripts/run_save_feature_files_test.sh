#!/usr/bin/env bash
SEED=0
python3 generate_feature_files.py --seed $SEED --split test --attack ALL
