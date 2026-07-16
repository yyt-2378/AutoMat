"""Compatibility aliases for the historical ``structure_recongnition`` name."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_PAIRED_ROOT = _SRC_ROOT / "structure_paired_reconstruction"
if str(_PAIRED_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAIRED_ROOT))
