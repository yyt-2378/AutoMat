---
name: confidence-estimation
description: Add, run, or debug uncertainty and confidence estimation in the agent_qwen workflow. Use when integrating structure_recongnition/confidence_estimation.py, generating per-atom/lattice confidence reports, or adding confidence artifacts to agent_qwen harness outputs.
---

# Confidence Estimation

Use `confidence_estimation_tool` for uncertainty quantification in the
`agent_qwen` workflow.

## Runtime Surface

- Wrapper: `agent_qwen/runtime_skills.py::_run_confidence`
- Source implementation: `structure_recongnition/confidence_estimation.py`
- Harness insertion point: after `stem2cif_tool` in classic workflow, after
  `direct_reconstruct_tool` in direct workflow

## Inputs

- `image_path`: denoised STEM image, usually `recon_png`
- `elements`: element symbols from user or template matching
- `pixel_size`: default `0.10`
- `output`: JSON report path

## Outputs

Return and preserve:

- `confidence_report`
- `global_confidence`
- `summary`

Treat this as an analysis artifact, not as a model checkpoint.
