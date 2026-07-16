---
name: stem-reconstruction
description: Run and extend the agent_qwen staged STEM image to CIF harness lines. Use when working on agent_qwen_vl_rpc.py, denoise/template-match/STEM-to-CIF orchestration, runtime tool calls, harness state passing, or real/dry-run testing of the legacy AutoMat v1-style agent pipeline.
---

# STEM Reconstruction

Use this skill for the `agent_qwen` staged reconstruction harness.

## Runtime Surface

- Harness: `agent_qwen/harness.py`
- Runtime tools: `agent_qwen/runtime_skills.py`
- Request intake: `agent_qwen/request_intake.py`
- Qwen tool adapters: `agent_qwen/qwen_tools.py`
- RPC entrypoint: `agent_qwen_vl_rpc.py`
- CLI smoke path: `python -m agent_qwen.cli`

Do not add new root-level `agent_*.py` scripts for harness steps. Add a runtime
tool wrapper in `agent_qwen/runtime_skills.py`, register it in
`default_registry()`, then call it from `AgentQwenHarness`.

Before running the harness from RPC or browser demo, normalize the user's natural
language with `normalize_harness_request()`. If `ready` is false, return
`need_input_payload()` and ask for the missing fields instead of running partial
state through the harness.

Required user inputs:

- classic line: raw STEM image path and element list
- direct line: denoised STEM image path and element list
- compare line: raw STEM image path and element list

The public MOE-DIVAESR checkpoint was trained at **0.1 Å/pixel**. Real
multiscale checkpoints for varying pixel sizes are not included in the public
release; real runs with the public checkpoint must use calibrated or resampled
0.1 Å/pixel inputs.

## Harness Lines

Classic line:

1. `denoise_patch_inference_tool`
2. `template_match_tool`
3. `stem2cif_tool`
4. Optional `confidence_estimation_tool`
5. Optional `property_prediction_tool`

Direct line:

1. `direct_reconstruct_tool`
2. Optional `confidence_estimation_tool`
3. Optional `property_prediction_tool`

Compare line:

1. `denoise_patch_inference_tool`
2. `classic.template_match_tool`
3. `classic.stem2cif_tool`
4. `direct.direct_reconstruct_tool`
5. Optional `compare.confidence_estimation_tool`

State handoff:

- denoise returns `recon_png`
- template matching consumes `recon_png`, returns `label_path` and `elements`
- stem2cif consumes `label_path` and `elements`, returns `cif_path`
- confidence consumes `recon_png` and `elements`, returns `confidence_report`
- property consumes `cif_path`, returns `relaxed_cif` and metrics

## Validation

Use dry-run first:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image examples/sample.png \
  --elements Mo S \
  --dry-run \
  --run-confidence
```

Use `img2struc_agent` for real runs:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image examples/stem.png \
  --elements Mo S \
  --run-confidence \
  --skip-property
```
