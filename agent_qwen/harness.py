"""Harness lines for the agent_qwen staged reconstruction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import PIPELINE_DEFAULTS
from .runtime_skills import AgentQwenSkillRegistry, SkillContext, SkillResult, default_registry


@dataclass(frozen=True)
class HarnessLineSpec:
    name: str
    description: str
    required_inputs: list[str]
    execution_mode: str = "sequential"


HARNESS_LINE_SPECS = {
    "classic": HarnessLineSpec(
        name="classic",
        description="Raw STEM image to CIF through denoise, template matching, and STEM-to-CIF.",
        required_inputs=["image_path", "elements"],
        execution_mode="sequential",
    ),
    "direct": HarnessLineSpec(
        name="direct",
        description="Denoised STEM image to minimal-cell CIF through the direct reconstruction tool.",
        required_inputs=["denoised_img", "elements"],
        execution_mode="sequential",
    ),
    "compare": HarnessLineSpec(
        name="compare",
        description="Shared denoise followed by classic and direct reconstruction branches in one result.",
        required_inputs=["image_path", "elements"],
        execution_mode="sequential-branches",
    ),
}
HARNESS_LINES = tuple(HARNESS_LINE_SPECS)


@dataclass
class HarnessResult:
    ok: bool
    harness_line: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    workflow: str = ""

    def __post_init__(self) -> None:
        if not self.workflow:
            self.workflow = self.harness_line


WorkflowResult = HarnessResult


class AgentQwenHarness:
    """Run agent_qwen harness lines as explicit, inspectable steps."""

    def __init__(
        self,
        context: SkillContext | None = None,
        registry: AgentQwenSkillRegistry | None = None,
    ) -> None:
        self.context = context or SkillContext()
        self.registry = registry or default_registry()

    @classmethod
    def from_defaults(
        cls,
        *,
        work_root: str | None = None,
        weight_path: str | None = None,
        label_dir: str | None = None,
        metadata_csv: str | None = None,
        device: str = "cuda",
        dry_run: bool = False,
    ) -> "AgentQwenHarness":
        context = SkillContext(
            work_root=work_root or PIPELINE_DEFAULTS["work_root"],
            weight_path=weight_path or PIPELINE_DEFAULTS["weight_path"],
            label_dir=label_dir or PIPELINE_DEFAULTS["label_dir"],
            metadata_csv=metadata_csv or PIPELINE_DEFAULTS["metadata_csv"],
            device=device,
            dry_run=dry_run,
        )
        return cls(context=context)

    def list_skills(self) -> list[dict[str, Any]]:
        return self.registry.describe()

    def list_lines(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "required_inputs": spec.required_inputs,
                "execution_mode": spec.execution_mode,
            }
            for spec in HARNESS_LINE_SPECS.values()
        ]

    def run_skill(self, name: str, params: dict[str, Any]) -> SkillResult:
        return self.registry.run(name, self.context, params)

    def run_line(self, harness_line: str = "classic", **kwargs: Any) -> HarnessResult:
        line = normalize_harness_line(harness_line)
        if line == "classic":
            return self.run_classic_line(**kwargs)
        if line == "direct":
            return self.run_direct_line(**kwargs)
        if line == "compare":
            return self.run_compare_line(**kwargs)
        raise ValueError(f"Unknown harness line: {harness_line}")

    def _record(
        self,
        rows: list[dict[str, Any]],
        artifacts: dict[str, str],
        state: dict[str, Any],
        name: str,
        result: SkillResult,
    ) -> bool:
        payload = result.as_tool_payload()
        rows.append(
            {
                "skill": name,
                "success": result.success,
                "message": result.message,
                "result": payload,
            }
        )
        artifacts.update({k: v for k, v in result.artifacts.items() if v})
        for key in (
            "recon_png",
            "label_path",
            "elements",
            "cif_path",
            "relaxed_cif",
            "final_cif",
            "confidence_report",
            "global_confidence",
        ):
            if key in payload and payload[key]:
                state[key] = payload[key]
        return result.success

    def run_classic_line(
        self,
        *,
        image_path: str,
        user_message: str = "",
        elements: list[str] | None = None,
        run_property: bool = True,
        run_confidence: bool = False,
        max_atoms: int = 50,
        max_shrink_iter: int = 4,
        relax_steps: int = 500,
        noise_amp: float = 0.05,
    ) -> HarnessResult:
        steps: list[dict[str, Any]] = []
        artifacts: dict[str, str] = {}
        state: dict[str, Any] = {"image_path": image_path, "user_message": user_message}

        result = self.run_skill("denoise_patch_inference_tool", {"image_path": image_path})
        ok = self._record(steps, artifacts, state, "denoise_patch_inference_tool", result)
        if not ok:
            return HarnessResult(False, "classic", steps, artifacts, state)

        result = self.run_skill(
            "template_match_tool",
            {
                "recon_png": state["recon_png"],
                "user_message": user_message,
                "elements": elements,
            },
        )
        ok = self._record(steps, artifacts, state, "template_match_tool", result)
        if not ok:
            return HarnessResult(False, "classic", steps, artifacts, state)

        result = self.run_skill(
            "stem2cif_tool",
            {
                "label_path": state["label_path"],
                "elements": state.get("elements") or elements or [],
                "max_atoms": max_atoms,
                "max_shrink_iter": max_shrink_iter,
            },
        )
        ok = self._record(steps, artifacts, state, "stem2cif_tool", result)
        if not ok:
            return HarnessResult(False, "classic", steps, artifacts, state)

        if run_confidence:
            result = self.run_skill(
                "confidence_estimation_tool",
                {
                    "image_path": state["recon_png"],
                    "elements": state.get("elements") or elements or [],
                    "output": str(Path(self.context.work_root) / "confidence" / "confidence_report.json"),
                },
            )
            ok = self._record(steps, artifacts, state, "confidence_estimation_tool", result)
            if not ok:
                return HarnessResult(False, "classic", steps, artifacts, state)

        if run_property:
            result = self.run_skill(
                "property_prediction_tool",
                {
                    "cif_path": state["cif_path"],
                    "relax_steps": relax_steps,
                    "noise_amp": noise_amp,
                },
            )
            ok = self._record(steps, artifacts, state, "property_prediction_tool", result)
            if not ok:
                return HarnessResult(False, "classic", steps, artifacts, state)

        return HarnessResult(True, "classic", steps, artifacts, state)

    def run_classic_workflow(self, **kwargs: Any) -> HarnessResult:
        return self.run_classic_line(**kwargs)

    def run_direct_line(
        self,
        *,
        denoised_img: str,
        elements: list[str],
        coord: dict[str, Any] | None = None,
        run_property: bool = False,
        run_confidence: bool = False,
    ) -> HarnessResult:
        steps: list[dict[str, Any]] = []
        artifacts: dict[str, str] = {}
        state: dict[str, Any] = {"denoised_img": denoised_img, "elements": elements}

        result = self.run_skill(
            "direct_reconstruct_tool",
            {
                "denoised_img": denoised_img,
                "elements": elements,
                "coord": coord,
                "out_dir": str(Path(self.context.work_root) / "direct_reconstruct"),
            },
        )
        ok = self._record(steps, artifacts, state, "direct_reconstruct_tool", result)
        if not ok:
            return HarnessResult(False, "direct", steps, artifacts, state)

        cif_path = state.get("final_cif")
        if run_confidence:
            result = self.run_skill(
                "confidence_estimation_tool",
                {
                    "image_path": denoised_img,
                    "elements": elements,
                    "output": str(Path(self.context.work_root) / "confidence" / "confidence_report.json"),
                },
            )
            ok = self._record(steps, artifacts, state, "confidence_estimation_tool", result)
            if not ok:
                return HarnessResult(False, "direct", steps, artifacts, state)

        if run_property and cif_path:
            result = self.run_skill("property_prediction_tool", {"cif_path": cif_path})
            ok = self._record(steps, artifacts, state, "property_prediction_tool", result)
            if not ok:
                return HarnessResult(False, "direct", steps, artifacts, state)

        return HarnessResult(True, "direct", steps, artifacts, state)

    def run_direct_workflow(self, **kwargs: Any) -> HarnessResult:
        return self.run_direct_line(**kwargs)

    def run_compare_line(
        self,
        *,
        image_path: str,
        user_message: str = "",
        elements: list[str] | None = None,
        run_property: bool = False,
        run_confidence: bool = True,
        max_atoms: int = 50,
        max_shrink_iter: int = 4,
    ) -> HarnessResult:
        """Run classic and direct reconstruction branches from a shared denoise step.

        This is the first parallel harness line: both branches live under one
        harness result, but execution is kept sequential for GPU/resource safety.
        """
        steps: list[dict[str, Any]] = []
        artifacts: dict[str, str] = {}
        state: dict[str, Any] = {"image_path": image_path, "user_message": user_message, "elements": elements or []}

        result = self.run_skill("denoise_patch_inference_tool", {"image_path": image_path})
        ok = self._record(steps, artifacts, state, "denoise_patch_inference_tool", result)
        if not ok:
            return HarnessResult(False, "compare", steps, artifacts, state)

        result = self.run_skill(
            "template_match_tool",
            {
                "recon_png": state["recon_png"],
                "user_message": user_message,
                "elements": elements,
            },
        )
        ok = self._record(steps, artifacts, state, "classic.template_match_tool", result)
        if not ok:
            return HarnessResult(False, "compare", steps, artifacts, state)

        branch_elements = state.get("elements") or elements or []
        result = self.run_skill(
            "stem2cif_tool",
            {
                "label_path": state["label_path"],
                "elements": branch_elements,
                "max_atoms": max_atoms,
                "max_shrink_iter": max_shrink_iter,
            },
        )
        ok = self._record(steps, artifacts, state, "classic.stem2cif_tool", result)
        if not ok:
            return HarnessResult(False, "compare", steps, artifacts, state)
        state["classic_cif_path"] = state.get("cif_path")
        if state.get("cif_path"):
            artifacts["classic_cif_path"] = state["cif_path"]

        result = self.run_skill(
            "direct_reconstruct_tool",
            {
                "denoised_img": state["recon_png"],
                "elements": branch_elements,
                "coord": None,
                "out_dir": str(Path(self.context.work_root) / "compare" / "direct_reconstruct"),
            },
        )
        ok = self._record(steps, artifacts, state, "direct.direct_reconstruct_tool", result)
        if not ok:
            return HarnessResult(False, "compare", steps, artifacts, state)
        state["direct_cif_path"] = state.get("final_cif")
        if state.get("final_cif"):
            artifacts["direct_cif_path"] = state["final_cif"]

        if run_confidence:
            result = self.run_skill(
                "confidence_estimation_tool",
                {
                    "image_path": state["recon_png"],
                    "elements": branch_elements,
                    "output": str(Path(self.context.work_root) / "compare" / "confidence" / "confidence_report.json"),
                },
            )
            ok = self._record(steps, artifacts, state, "compare.confidence_estimation_tool", result)
            if not ok:
                return HarnessResult(False, "compare", steps, artifacts, state)

        if run_property and state.get("classic_cif_path"):
            result = self.run_skill("property_prediction_tool", {"cif_path": state["classic_cif_path"]})
            ok = self._record(steps, artifacts, state, "classic.property_prediction_tool", result)
            if not ok:
                return HarnessResult(False, "compare", steps, artifacts, state)

        return HarnessResult(True, "compare", steps, artifacts, state)


V1AgentHarness = AgentQwenHarness


def normalize_harness_line(value: str | None) -> str:
    line = (value or "classic").strip().lower().replace("_", "-")
    aliases = {
        "workflow": "classic",
        "classic-workflow": "classic",
        "classic-line": "classic",
        "direct-workflow": "direct",
        "direct-line": "direct",
        "compare-workflow": "compare",
        "comparison": "compare",
        "comparison-line": "compare",
        "parallel": "compare",
    }
    line = aliases.get(line, line)
    if line not in HARNESS_LINES:
        raise ValueError(f"Unknown harness line '{value}'. Expected one of: {', '.join(HARNESS_LINES)}")
    return line
