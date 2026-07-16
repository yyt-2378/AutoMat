"""Normalize natural-language requests into agent_qwen harness inputs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .harness import normalize_harness_line


_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class HarnessRequest:
    harness_line: str = "classic"
    image_path: str | None = None
    denoised_img: str | None = None
    elements: list[str] = field(default_factory=list)
    user_message: str = ""
    work_root: str | None = None
    weight_path: str | None = None
    label_dir: str | None = None
    metadata_csv: str | None = None
    device: str = "cuda"
    dry_run: bool = False
    skip_property: bool = False
    run_confidence: bool = False
    coord: dict[str, Any] | None = None
    missing_required: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    @property
    def workflow(self) -> str:
        return self.harness_line

    @property
    def ready(self) -> bool:
        return not self.missing_required

    def as_context(self) -> dict[str, Any]:
        return {
            "harness_line": self.harness_line,
            "workflow": self.harness_line,
            "image_path": self.image_path,
            "denoised_img": self.denoised_img,
            "elements": self.elements,
            "user_message": self.user_message,
            "work_root": self.work_root,
            "weight_path": self.weight_path,
            "label_dir": self.label_dir,
            "metadata_csv": self.metadata_csv,
            "device": self.device,
            "dry_run": self.dry_run,
            "skip_property": self.skip_property,
            "run_confidence": self.run_confidence,
            "coord": self.coord,
        }

    def need_input_payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "need_input": True,
            "harness_line": self.harness_line,
            "workflow": self.harness_line,
            "missing_required": self.missing_required,
            "guidance": self.guidance,
            "parsed": self.as_context(),
        }


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "是", "需要", "跑"}
    return bool(value)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _extract_paths(text: str) -> list[str]:
    pattern = r'(?:"([^"]+)")|(?:\'([^\']+)\')|([^\s,，;；。]+)'
    paths: list[str] = []
    for groups in re.findall(pattern, text):
        token = next((part for part in groups if part), "")
        suffix = Path(token.rstrip("。.;；,，")).suffix.lower()
        if suffix in _IMAGE_EXTS or suffix == ".cif":
            paths.append(token.rstrip("。.;；,，"))
    return _dedupe(paths)


def _normalize_element(token: str) -> str | None:
    token = token.strip().strip("[](){}:：,，;；。")
    if not token:
        return None
    candidate = token[0].upper() + token[1:].lower()
    return candidate if candidate in _ELEMENTS else None


def _extract_elements(text: str) -> list[str]:
    spans: list[str] = []
    keyword_pattern = r'(?:元素|element(?:s)?|composition|成分)\s*[:：=为是]?\s*([A-Za-z0-9,\s，、/+\-]+)'
    for match in re.finditer(keyword_pattern, text, flags=re.IGNORECASE):
        spans.append(match.group(1))

    bracket_pattern = r'\[([A-Z][a-z]?(?:\s*,\s*[A-Z][a-z]?)+)\]'
    spans.extend(re.findall(bracket_pattern, text))

    if not spans:
        compact = re.findall(r'\b([A-Z][a-z]?)(?:\s*[,\s+/，、]\s*([A-Z][a-z]?))+\b', text)
        for groups in compact:
            spans.append(" ".join(groups))

    elements: list[str] = []
    for span in spans:
        for token in re.split(r'[\s,，、/+\-]+', span):
            element = _normalize_element(token)
            if element:
                elements.append(element)
    return _dedupe(elements)


def _choose_workflow(text: str, context: dict[str, Any], default: str = "classic") -> str:
    explicit = context.get("harness_line") or context.get("workflow")
    if explicit:
        return normalize_harness_line(str(explicit))
    lowered = text.lower()
    if any(marker in lowered for marker in ("compare", "comparison", "parallel")):
        return "compare"
    if any(marker in text for marker in ("比较", "对比", "并行")):
        return "compare"
    if any(marker in lowered for marker in ("direct", "denoised", "direct_reconstruct")):
        return "direct"
    if any(marker in text for marker in ("直通", "去噪后", "最小单胞", "绕过模板")):
        return "direct"
    return default


def normalize_harness_request(
    query: str = "",
    context: dict[str, Any] | None = None,
    *,
    default_workflow: str = "classic",
    default_dry_run: bool = False,
    default_skip_property: bool = False,
    default_run_confidence: bool = False,
) -> HarnessRequest:
    context = dict(context or {})
    query = query or ""
    text = " ".join(str(part) for part in (query, context.get("user_message", "")) if part)
    paths = _extract_paths(text)
    workflow = _choose_workflow(text, context, default_workflow)

    elements = context.get("elements") or _extract_elements(text)
    if isinstance(elements, str):
        elements = [_normalize_element(token) for token in re.split(r'[\s,，、/+\-]+', elements)]
        elements = [item for item in elements if item]

    image_path = context.get("image_path") or context.get("image")
    denoised_img = context.get("denoised_img")
    if not image_path and workflow in {"classic", "compare"}:
        image_path = next((path for path in paths if Path(path).suffix.lower() in _IMAGE_EXTS), None)
    if not denoised_img and workflow == "direct":
        denoised_img = next((path for path in paths if Path(path).suffix.lower() in _IMAGE_EXTS), None)

    run_confidence = _as_bool(context.get("run_confidence"), default_run_confidence)
    if re.search(r'置信|不确定|uncertainty|confidence', text, flags=re.IGNORECASE):
        run_confidence = True

    skip_property = _as_bool(context.get("skip_property"), default_skip_property)
    if re.search(r'跳过物性|不跑物性|不要物性|skip[-_ ]?property', text, flags=re.IGNORECASE):
        skip_property = True

    req = HarnessRequest(
        harness_line=workflow,
        image_path=image_path,
        denoised_img=denoised_img,
        elements=_dedupe(list(elements or [])),
        user_message=context.get("user_message", query),
        work_root=context.get("work_root"),
        weight_path=context.get("weight_path"),
        label_dir=context.get("label_dir"),
        metadata_csv=context.get("metadata_csv"),
        device=context.get("device", "cuda"),
        dry_run=_as_bool(context.get("dry_run"), default_dry_run),
        skip_property=skip_property,
        run_confidence=run_confidence,
        coord=context.get("coord"),
    )
    _validate(req)
    return req


def _validate(req: HarnessRequest) -> None:
    missing: list[str] = []
    guidance: list[str] = []
    if req.harness_line == "direct":
        if not req.denoised_img:
            missing.append("denoised_img")
            guidance.append("请提供去噪后的 STEM 图像路径，例如 denoised_img=examples/recon.png。")
        if not req.elements:
            missing.append("elements")
            guidance.append("请提供元素列表，例如 元素: Mo S 或 elements=[\"Mo\", \"S\"]。")
    else:
        if not req.image_path:
            missing.append("image_path")
            guidance.append("请提供原始 STEM 图像路径，例如 image_path=examples/input.tif。")
        if not req.elements:
            missing.append("elements")
            guidance.append("请提供元素列表，例如 元素: Mo S。模板匹配可辅助判断，但重建 CIF 时需要明确元素种类。")

    req.missing_required = missing
    req.guidance = guidance
