"""Runtime tool registry backed by the existing ``pipline_framework`` tools."""
from __future__ import annotations

import json
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .defaults import PIPELINE_DEFAULTS


@dataclass
class SkillContext:
    work_root: str = PIPELINE_DEFAULTS["work_root"]
    weight_path: str = PIPELINE_DEFAULTS["weight_path"]
    label_dir: str = PIPELINE_DEFAULTS["label_dir"]
    metadata_csv: str = PIPELINE_DEFAULTS["metadata_csv"]
    device: str = "cuda"
    dry_run: bool = False


@dataclass
class SkillResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str = ""

    def as_tool_payload(self) -> dict[str, Any]:
        payload = {"success": self.success, **self.data}
        if self.artifacts:
            payload["artifacts"] = self.artifacts
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass
class SkillSpec:
    name: str
    description: str
    parameters: list[dict[str, Any]]
    run: Callable[[SkillContext, dict[str, Any]], SkillResult]


class AgentQwenSkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"Duplicate skill: {spec.name}")
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._skills))
            raise KeyError(f"Unknown agent_qwen runtime skill '{name}'. Known skills: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._skills)

    def qwen_function_list(self) -> list[str]:
        return self.names()

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in sorted(self._skills.values(), key=lambda item: item.name)
        ]

    def run(self, name: str, context: SkillContext, params: dict[str, Any]) -> SkillResult:
        return self.get(name).run(context, params)


def _pipeline_module():
    orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    src_root = PIPELINE_DEFAULTS.get("src_root")
    if src_root and src_root not in sys.path:
        sys.path.insert(0, src_root)
    preprocess_root = str(Path(src_root) / "preprocess_model") if src_root else ""
    if preprocess_root and preprocess_root not in sys.path:
        sys.path.insert(0, preprocess_root)
    sr_root = str(Path(preprocess_root) / "SR_model") if preprocess_root else ""
    if sr_root and sr_root not in sys.path:
        sys.path.insert(0, sr_root)
    try:
        import pipline_framework
    finally:
        sys.argv = orig_argv

    return pipline_framework


def _dry_payload(context: SkillContext, skill_name: str, params: dict[str, Any], extra: dict[str, Any]) -> SkillResult:
    return SkillResult(
        True,
        data={
            "dry_run": True,
            "would_call": skill_name,
            "params": params,
            **extra,
        },
        artifacts={k: v for k, v in extra.items() if isinstance(v, str) and Path(v).suffix},
        message=f"dry-run {skill_name}",
    )


def _param(name: str, typ: str, description: str, required: bool = True, default: Any = None) -> dict[str, Any]:
    row = {"name": name, "type": typ, "description": description, "required": required}
    if default is not None:
        row["default"] = default
    return row


def _context_defaults(context: SkillContext, params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(params)
    merged.setdefault("work_root", context.work_root)
    merged.setdefault("weight_path", context.weight_path)
    merged.setdefault("label_dir", context.label_dir)
    merged.setdefault("metadata_csv", context.metadata_csv)
    merged.setdefault("device", context.device)
    return merged


def _run_assess_image(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = dict(params)
    if context.dry_run or params.pop("dry_run", False):
        return _dry_payload(context, "assess_image_path_tool", params, {"assessment": "skipped"})
    result = _pipeline_module().assess_image_path(**params)
    return SkillResult(True, data=result, message="image assessed")


def _run_denoise(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = _context_defaults(context, params)
    if context.dry_run or params.pop("dry_run", False):
        image_stem = Path(params["image_path"]).stem
        recon_png = str(Path(params["work_root"]) / "01_recon" / f"{image_stem}_recon.png")
        return _dry_payload(context, "denoise_patch_inference_tool", params, {"recon_png": recon_png})
    result = _pipeline_module().denoise_patch_inference_tool(
        image_path=params["image_path"],
        weight_path=params["weight_path"],
        work_root=params["work_root"],
        device=params.get("device", context.device),
    )
    return SkillResult(bool(result.get("success")), data=result, artifacts={"recon_png": result.get("recon_png", "")})


def _run_template_match(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = _context_defaults(context, params)
    params.setdefault("user_message", "")
    if context.dry_run or params.pop("dry_run", False):
        label_path = str(Path(params["work_root"]) / "02_label" / "dry_run_best_label.png")
        elements = params.get("elements") or ["Mo", "S"]
        return _dry_payload(
            context,
            "template_match_tool",
            params,
            {"label_path": label_path, "elements": elements},
        )
    result = _pipeline_module().template_match_tool(
        **_filter_call_kwargs(
            _pipeline_module().template_match_tool,
            {
                "recon_png": params["recon_png"],
                "label_dir": params["label_dir"],
                "metadata_csv": params["metadata_csv"],
                "user_message": _message_with_elements(params["user_message"], params.get("elements")),
                "work_root": params["work_root"],
                "elements": params.get("elements"),
            },
        )
    )
    artifacts = {"label_path": result.get("label_path", "")}
    return SkillResult(bool(result.get("success")), data=result, artifacts=artifacts)


def _run_stem2cif(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = _context_defaults(context, params)
    if context.dry_run or params.pop("dry_run", False):
        cif_path = str(Path(params["work_root"]) / "03_recon_cif" / "output_final.cif")
        return _dry_payload(context, "stem2cif_tool", params, {"cif_path": cif_path})
    result = _pipeline_module().stem2cif_tool(
        label_path=params["label_path"],
        elements=params["elements"],
        work_root=params["work_root"],
        max_atoms=int(params.get("max_atoms", 50)),
        max_shrink_iter=int(params.get("max_shrink_iter", 4)),
    )
    return SkillResult(bool(result.get("success")), data=result, artifacts={"cif_path": result.get("cif_path", "")})


def _run_property(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = _context_defaults(context, params)
    if context.dry_run or params.pop("dry_run", False):
        relaxed_cif = str(Path(params["work_root"]) / "04_relax" / "relaxed.cif")
        return _dry_payload(
            context,
            "property_prediction_tool",
            params,
            {"relaxed_cif": relaxed_cif, "energy_eV": 0.0, "converged": True},
        )
    result = _pipeline_module().property_prediction_tool(
        cif_path=params["cif_path"],
        work_root=params["work_root"],
        noise_amp=float(params.get("noise_amp", 0.05)),
        relax_steps=int(params.get("relax_steps", 500)),
        device=params.get("device", context.device),
    )
    artifacts = {"relaxed_cif": result.get("relaxed_cif", "")}
    return SkillResult(bool(result.get("success")), data=result, artifacts=artifacts)


def _run_direct_reconstruct(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = dict(params)
    params.setdefault("out_dir", str(Path(context.work_root) / "direct_reconstruct"))
    if context.dry_run or params.pop("dry_run", False):
        final_cif = str(Path(params["out_dir"]) / "minimal_cell_from_top3_pipeline.cif")
        return _dry_payload(
            context,
            "direct_reconstruct_tool",
            params,
            {"final_cif": final_cif, "cell": {}, "basis_atoms": []},
        )
    final_cif, cell, basis_atoms = _pipeline_module().reconstruct_from_denoised_img(
        denoised_img=params["denoised_img"],
        user_elements=params["elements"],
        coord=params.get("coord"),
        pixel_size=float(params.get("pixel_size", 0.10)),
        top_n=int(params.get("top_n", 3)),
        out_dir=params["out_dir"],
    )
    data = {"success": True, "final_cif": final_cif, "cell": cell, "basis_atoms": basis_atoms}
    return SkillResult(True, data=data, artifacts={"final_cif": final_cif})


def _run_confidence(context: SkillContext, params: dict[str, Any]) -> SkillResult:
    params = dict(params)
    params.setdefault("pixel_size", 0.10)
    params.setdefault("output", str(Path(context.work_root) / "confidence" / "confidence_report.json"))
    if context.dry_run or params.pop("dry_run", False):
        return _dry_payload(
            context,
            "confidence_estimation_tool",
            params,
            {
                "confidence_report": params["output"],
                "global_confidence": 1.0,
                "summary": {"total_atoms_detected": 0},
            },
        )
    from structure_recongnition.confidence_estimation import generate_confidence_report

    report = generate_confidence_report(
        image_path=params["image_path"],
        elements_type=params["elements"],
        pixel_size=float(params.get("pixel_size", 0.10)),
        output_json=params.get("output"),
    )
    artifacts = {"confidence_report": params["output"]} if params.get("output") else {}
    data = {
        "confidence_report": params.get("output"),
        "global_confidence": report.get("global_confidence"),
        "summary": report.get("summary", {}),
        "lattice_confidence": report.get("lattice_confidence", {}),
    }
    return SkillResult(True, data=data, artifacts=artifacts)


def _filter_call_kwargs(func: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params
    return {key: value for key, value in params.items() if key in signature.parameters}


def _message_with_elements(user_message: str, elements: list[str] | None) -> str:
    if not elements:
        return user_message
    if "元素" in user_message or "element" in user_message.lower():
        return user_message
    return f"{user_message} 元素: {' '.join(elements)}".strip()


def default_registry() -> AgentQwenSkillRegistry:
    registry = AgentQwenSkillRegistry()
    registry.register(
        SkillSpec(
            name="assess_image_path_tool",
            description="评估图像路径有效性和图像质量。",
            parameters=[_param("path", "string", "图像路径")],
            run=_run_assess_image,
        )
    )
    registry.register(
        SkillSpec(
            name="denoise_patch_inference_tool",
            description="对 STEM 大图进行 patch 去噪和重建，返回 recon_png。",
            parameters=[
                _param("image_path", "string", "STEM 大图路径"),
                _param("weight_path", "string", "去噪模型权重路径", False),
                _param("work_root", "string", "工作目录", False),
                _param("device", "string", "推理设备", False, "cuda"),
            ],
            run=_run_denoise,
        )
    )
    registry.register(
        SkillSpec(
            name="template_match_tool",
            description="对去噪后图片做模板匹配，返回 label_path 和元素信息。",
            parameters=[
                _param("recon_png", "string", "去噪后图片路径"),
                _param("label_dir", "string", "模板 label 目录", False),
                _param("metadata_csv", "string", "材料元素元数据 CSV", False),
                _param("user_message", "string", "用户补充说明", False),
                _param("work_root", "string", "工作目录", False),
                _param("elements", "array", "已知元素列表", False),
            ],
            run=_run_template_match,
        )
    )
    registry.register(
        SkillSpec(
            name="stem2cif_tool",
            description="将 label 图片和元素类型转换为 CIF 结构。",
            parameters=[
                _param("label_path", "string", "label 图片路径"),
                _param("elements", "array", "元素类型列表"),
                _param("work_root", "string", "工作目录", False),
                _param("max_atoms", "integer", "最大原子数", False, 50),
                _param("max_shrink_iter", "integer", "最大缩减迭代次数", False, 4),
            ],
            run=_run_stem2cif,
        )
    )
    registry.register(
        SkillSpec(
            name="property_prediction_tool",
            description="对 CIF 结构进行 MatterSim 物性预测和 relaxation。",
            parameters=[
                _param("cif_path", "string", "CIF 结构路径"),
                _param("work_root", "string", "工作目录", False),
                _param("noise_amp", "number", "扰动幅度", False, 0.05),
                _param("relax_steps", "integer", "松弛步数", False, 500),
                _param("device", "string", "推理设备", False, "cuda"),
            ],
            run=_run_property,
        )
    )
    registry.register(
        SkillSpec(
            name="direct_reconstruct_tool",
            description="绕过模板匹配，使用 FFT 候选窗口到最小单胞的直通重建。",
            parameters=[
                _param("denoised_img", "string", "去噪后的 STEM 图像路径"),
                _param("elements", "array", "元素列表"),
                _param("coord", "object", "可选配位关系", False),
                _param("pixel_size", "number", "像素尺寸", False, 0.10),
                _param("top_n", "integer", "候选窗口数量", False, 3),
                _param("out_dir", "string", "输出目录", False),
            ],
            run=_run_direct_reconstruct,
        )
    )
    registry.register(
        SkillSpec(
            name="confidence_estimation_tool",
            description="对去噪图像进行原子/晶格置信度估计，输出 uncertainty/confidence JSON。",
            parameters=[
                _param("image_path", "string", "去噪后图像路径"),
                _param("elements", "array", "元素列表"),
                _param("pixel_size", "number", "像素尺寸", False, 0.10),
                _param("output", "string", "输出 JSON 路径", False),
            ],
            run=_run_confidence,
        )
    )
    return registry


V1SkillRegistry = AgentQwenSkillRegistry


def dumps_tool_result(result: SkillResult) -> str:
    return json.dumps(result.as_tool_payload(), ensure_ascii=False)
