#!/usr/bin/env bash
python3 - <<'PY'
from multistage_pipeline.utils import ensure_dirs

ensure_dirs()
print("Directories created successfully.")
PY
