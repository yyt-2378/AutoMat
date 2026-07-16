"""Default paths for the AutoMat harness lines."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


PIPELINE_DEFAULTS = {
    "weight_path": os.environ.get(
        "AUTOMAT_WEIGHT_PATH",
        "MOE_model_weights/moe_model.ckpt",
    ),
    "label_dir": os.environ.get(
        "AUTOMAT_LABEL_DIR",
        "data_generation/label",
    ),
    "metadata_csv": os.environ.get(
        "AUTOMAT_METADATA_CSV",
        "baseline/property.csv",
    ),
    "work_root": os.environ.get("AUTOMAT_WORK_ROOT", "harness_runs/current"),
    "src_root": str(SRC_ROOT),
}
