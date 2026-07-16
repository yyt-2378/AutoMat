# agent_qwen Harness + Skills

This package turns the original `agent_qwen_*` staged agent into explicit
harness lines while preserving the old tool names used by
`agent_qwen_vl_rpc.py`.

## Normative Skills

Markdown skill specs live under `agent_qwen/skills/*/SKILL.md`:

- `skills/stem-reconstruction/SKILL.md`
- `skills/confidence-estimation/SKILL.md`

These files define the intended harness behavior and maintenance rules. Python runtime
wrappers live in `runtime_skills.py`; they are execution code, not the normative
skill spec.

## Runtime Tools

- `denoise_patch_inference_tool`
- `template_match_tool`
- `stem2cif_tool`
- `property_prediction_tool`
- `direct_reconstruct_tool`
- `assess_image_path_tool`
- `confidence_estimation_tool`

`confidence_estimation_tool` wraps the existing
`structure_recongnition/confidence_estimation.py` uncertainty/confidence report.

## Public Checkpoint Scope

The released MOE-DIVAESR checkpoint was trained at a fixed real-space sampling
of **0.1 Å/pixel**. Calibrate or resample input images to 0.1 Å/pixel before
inference. The real multiscale checkpoints for varying pixel sizes are not part
of this public release.

## Harness

See `HARNESS.md`. The harness implementation is `AgentQwenHarness` in
`harness.py`.

List the standard harness lines:

```bash
python -m agent_qwen.cli --list-lines
```

## Natural-Language Intake

`request_intake.py` normalizes a user query plus optional JSON context before the
harness runs. It extracts:

- harness line: `classic`, `direct`, or `compare`
- image inputs: `image_path` or `denoised_img`
- elements, for example `元素: Mo S`
- flags such as confidence/uncertainty and skip-property requests

If required inputs are missing, RPC/demo callers receive `need_input: true`,
`missing_required`, and user-facing `guidance` instead of a failed harness run.

## Harness Lines

Classic line:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image path/to/input.png \
  --elements Mo S \
  --run-confidence
```

Dry-run, useful for testing wiring without loading cv2/model/MatterSim:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image examples/sample.png \
  --elements Mo S \
  --dry-run \
  --run-confidence
```

Compare line, useful for keeping classic and direct branches side by side:

```bash
python -m agent_qwen.cli \
  --harness-line compare \
  --image examples/sample.png \
  --elements Mo S \
  --dry-run \
  --run-confidence \
  --skip-property
```

The old RPC agent can also run the harness directly:

```bash
printf '%s\n' '{"query":"元素: Mo S","context":{"harness_line":"classic","image_path":"examples/sample.png","elements":["Mo","S"],"dry_run":true,"run_confidence":true}}' \
  | python agent_qwen_vl_rpc.py --rpc --harness --dry_run
```

Natural-language only requests are also parsed:

```bash
printf '%s\n' '{"query":"请用 examples/sample.png 重建，元素: Mo S，并给出不确定性","context":{"dry_run":true,"skip_property":true}}' \
  | python agent_qwen_vl_rpc.py --rpc --harness --dry_run
```

Without `--harness`, `agent_qwen_vl_rpc.py` keeps the original Qwen-Agent LLM
tool-calling behavior. Qwen-Agent and pipeline dependencies are now optional for
harness dry-runs and skill listing.

`workflow` is still accepted as a deprecated alias for `harness_line` so older
scripts keep running.
