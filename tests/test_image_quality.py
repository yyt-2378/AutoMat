from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structure_recongnition.justify_img_quality import assess_image_path


def test_assess_image_path_returns_quality_metrics(tmp_path: Path):
    image = np.zeros((64, 64), dtype=np.uint8)
    image[::8, :] = 255
    image[:, ::8] = 255
    image_path = tmp_path / "periodic.png"
    assert cv2.imwrite(str(image_path), image)

    result = assess_image_path(
        str(image_path),
        min_periodicity=0.0,
        min_snr=0.0,
        min_cnr=0.0,
    )

    assert result["image"] == str(image_path)
    assert result["decision"] == "PASS"
    assert set(result) >= {
        "periodicity_score",
        "bragg_peakiness",
        "autocorr_peak_ratio",
        "snr",
        "cnr",
        "noise_sigma",
        "thresholds",
    }


def test_assess_image_path_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        assess_image_path(str(tmp_path / "missing.png"))
