from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent_qwen import (
    AgentQwenHarness,
    default_registry,
    extract_artifacts_from_text,
    normalize_harness_request,
)


def test_agent_qwen_registry_exposes_old_tools_and_new_confidence_skill():
    names = set(default_registry().names())
    assert "denoise_patch_inference_tool" in names
    assert "template_match_tool" in names
    assert "stem2cif_tool" in names
    assert "property_prediction_tool" in names
    assert "direct_reconstruct_tool" in names
    assert "confidence_estimation_tool" in names


def test_harness_lists_standard_lines():
    harness = AgentQwenHarness.from_defaults(dry_run=True)
    lines = {row["name"]: row for row in harness.list_lines()}

    assert set(lines) == {"classic", "direct", "compare"}
    assert lines["classic"]["required_inputs"] == ["image_path", "elements"]
    assert lines["direct"]["required_inputs"] == ["denoised_img", "elements"]
    assert lines["compare"]["execution_mode"] == "sequential-branches"


def test_classic_workflow_dry_run_propagates_artifacts(tmp_path: Path):
    harness = AgentQwenHarness.from_defaults(work_root=str(tmp_path), dry_run=True)
    result = harness.run_classic_workflow(
        image_path="examples/sample_sampling0.1_iDPC_V3.png",
        user_message="元素: Mo S",
        elements=["Mo", "S"],
        run_property=True,
        run_confidence=True,
    )

    assert result.ok
    assert result.harness_line == "classic"
    assert result.workflow == "classic"
    assert [step["skill"] for step in result.steps] == [
        "denoise_patch_inference_tool",
        "template_match_tool",
        "stem2cif_tool",
        "confidence_estimation_tool",
        "property_prediction_tool",
    ]
    assert result.state["recon_png"].endswith("_recon.png")
    assert result.state["label_path"].endswith("dry_run_best_label.png")
    assert result.state["cif_path"].endswith("output_final.cif")
    assert result.state["relaxed_cif"].endswith("relaxed.cif")
    assert "confidence_report" in result.artifacts


def test_direct_workflow_dry_run(tmp_path: Path):
    harness = AgentQwenHarness.from_defaults(work_root=str(tmp_path), dry_run=True)
    result = harness.run_direct_workflow(
        denoised_img="examples/denoised.png",
        elements=["Zr", "O"],
        run_property=False,
        run_confidence=True,
    )

    assert result.ok
    assert result.harness_line == "direct"
    assert result.workflow == "direct"
    assert result.state["final_cif"].endswith("minimal_cell_from_top3_pipeline.cif")
    assert result.state["confidence_report"].endswith("confidence_report.json")


def test_extract_artifacts_from_text_keeps_order():
    text = "输出 results/a.cif 和 ./b.png，然后还有 results/a.cif"
    assert extract_artifacts_from_text(text) == {"files": ["results/a.cif", "./b.png"]}


def test_natural_language_intake_extracts_classic_inputs():
    req = normalize_harness_request(
        "请用 examples/sample_sampling0.1_iDPC_V3.png 重建，元素: Mo S，并给出不确定性，跳过物性。",
        {"dry_run": True},
    )

    assert req.ready
    assert req.harness_line == "classic"
    assert req.workflow == "classic"
    assert req.image_path == "examples/sample_sampling0.1_iDPC_V3.png"
    assert req.elements == ["Mo", "S"]
    assert req.run_confidence is True
    assert req.skip_property is True


def test_natural_language_intake_extracts_compare_line():
    req = normalize_harness_request(
        "请比较 classic 和 direct 两条线，用 examples/sample.png，元素: Mo S，给出置信度。",
        {"dry_run": True},
    )

    assert req.ready
    assert req.harness_line == "compare"
    assert req.workflow == "compare"
    assert req.image_path == "examples/sample.png"
    assert req.elements == ["Mo", "S"]


def test_natural_language_intake_guides_missing_required_fields():
    req = normalize_harness_request("请帮我重建并给出置信度")

    assert not req.ready
    assert req.missing_required == ["image_path", "elements"]
    assert any("STEM 图像路径" in item for item in req.guidance)
    assert any("元素列表" in item for item in req.guidance)


def test_agent_qwen_harness_cli_dry_run(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_qwen.cli",
            "--workflow",
            "classic",
            "--image",
            "examples/sample.png",
            "--elements",
            "Mo",
            "S",
            "--work-root",
            str(tmp_path),
            "--dry-run",
            "--run-confidence",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["harness_line"] == "classic"
    assert payload["workflow"] == "classic"
    assert payload["state"]["cif_path"].endswith("output_final.cif")


def test_agent_qwen_harness_cli_lists_lines():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_qwen.cli",
            "--list-lines",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert [row["name"] for row in payload] == ["classic", "direct", "compare"]


def test_compare_harness_line_dry_run(tmp_path: Path):
    harness = AgentQwenHarness.from_defaults(work_root=str(tmp_path), dry_run=True)
    result = harness.run_line(
        "compare",
        image_path="examples/sample.png",
        user_message="元素: Mo S",
        elements=["Mo", "S"],
        run_confidence=True,
        run_property=False,
    )

    assert result.ok
    assert result.harness_line == "compare"
    assert result.workflow == "compare"
    assert [step["skill"] for step in result.steps] == [
        "denoise_patch_inference_tool",
        "classic.template_match_tool",
        "classic.stem2cif_tool",
        "direct.direct_reconstruct_tool",
        "compare.confidence_estimation_tool",
    ]
    assert result.state["classic_cif_path"].endswith("output_final.cif")
    assert result.state["direct_cif_path"].endswith("minimal_cell_from_top3_pipeline.cif")


def test_rpc_harness_returns_guidance_for_missing_inputs():
    proc = subprocess.run(
        [
            sys.executable,
            "agent_qwen_vl_rpc.py",
            "--rpc",
            "--harness",
            "--dry_run",
        ],
        input='{"query":"请帮我重建并给出置信度","context":{}}\n',
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["need_input"] is True
    assert payload["missing_required"] == ["image_path", "elements"]
