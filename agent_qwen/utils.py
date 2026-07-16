"""Small utilities shared by the agent_qwen harness and RPC agent."""
from __future__ import annotations

import re
from typing import Any


def extract_artifacts_from_text(text: str) -> dict[str, Any]:
    """Extract common artifact paths from assistant text."""
    if not text:
        return {}
    exts = (".cif", ".vasp", ".poscar", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".xyz", ".json", ".csv")
    ext_group = "|".join(re.escape(ext.lstrip(".")) for ext in exts)
    path_pat = re.compile(rf"((?:/|\./|[A-Za-z0-9_.-])\S*?\.({ext_group}))", re.I)
    found = []
    for match in path_pat.finditer(text):
        candidate = match.group(1)
        candidate = candidate.rstrip("，,.;；:：)。)]}>\"'")
        found.append(candidate)
    seen = set()
    uniq = []
    for item in found:
        if item not in seen:
            uniq.append(item)
            seen.add(item)
    return {"files": uniq} if uniq else {}
