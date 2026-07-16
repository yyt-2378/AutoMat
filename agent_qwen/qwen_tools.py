"""Qwen-Agent tool adapters backed by ``agent_qwen`` skills."""
from __future__ import annotations

import json5

from .runtime_skills import SkillContext, default_registry, dumps_tool_result

from qwen_agent.tools.base import BaseTool, register_tool


_REGISTRY = default_registry()
_CONTEXT = SkillContext()


def configure_qwen_tool_context(**kwargs) -> None:
    """Update defaults used by Qwen tool calls."""
    for key, value in kwargs.items():
        if value is not None and hasattr(_CONTEXT, key):
            setattr(_CONTEXT, key, value)


def qwen_function_list() -> list[str]:
    return _REGISTRY.qwen_function_list()


def _call_skill(name: str, params: str) -> str:
    args = json5.loads(params or "{}")
    return dumps_tool_result(_REGISTRY.run(name, _CONTEXT, args))


@register_tool("denoise_patch_inference_tool")
class DenoisePatchTool(BaseTool):
    description = _REGISTRY.get("denoise_patch_inference_tool").description
    parameters = _REGISTRY.get("denoise_patch_inference_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("denoise_patch_inference_tool", params)


@register_tool("template_match_tool")
class TemplateMatchTool(BaseTool):
    description = _REGISTRY.get("template_match_tool").description
    parameters = _REGISTRY.get("template_match_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("template_match_tool", params)


@register_tool("stem2cif_tool")
class Stem2CifTool(BaseTool):
    description = _REGISTRY.get("stem2cif_tool").description
    parameters = _REGISTRY.get("stem2cif_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("stem2cif_tool", params)


@register_tool("property_prediction_tool")
class PropertyPredictionTool(BaseTool):
    description = _REGISTRY.get("property_prediction_tool").description
    parameters = _REGISTRY.get("property_prediction_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("property_prediction_tool", params)


@register_tool("direct_reconstruct_tool")
class DirectReconstructTool(BaseTool):
    description = _REGISTRY.get("direct_reconstruct_tool").description
    parameters = _REGISTRY.get("direct_reconstruct_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("direct_reconstruct_tool", params)


@register_tool("assess_image_path_tool")
class AssessImagePathTool(BaseTool):
    description = _REGISTRY.get("assess_image_path_tool").description
    parameters = _REGISTRY.get("assess_image_path_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("assess_image_path_tool", params)


@register_tool("confidence_estimation_tool")
class ConfidenceEstimationTool(BaseTool):
    description = _REGISTRY.get("confidence_estimation_tool").description
    parameters = _REGISTRY.get("confidence_estimation_tool").parameters

    def call(self, params: str, **kwargs) -> str:
        return _call_skill("confidence_estimation_tool", params)
