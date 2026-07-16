# agent_qwen Harness

`AgentQwenHarness` in `agent_qwen/harness.py` is the harness-line runner for
the existing `agent_qwen_*` family. It is not a separate agent; it is the
structured execution layer behind `agent_qwen_vl_rpc.py --harness`.

## Responsibilities

- Normalize user query/context into runnable harness inputs.
- Execute ordered harness-line steps.
- Pass state between runtime tools.
- Collect artifacts and metrics.
- Support dry-run without importing heavy pipeline dependencies.
- Keep the old Qwen tool names compatible.

## Runtime Skills Versus SKILL.md

- `agent_qwen/skills/*/SKILL.md`: normative instructions for agents and
  maintainers.
- `agent_qwen/runtime_skills.py`: Python wrappers that actually call
  `pipline_framework.py` and related modules.
- `agent_qwen/request_intake.py`: natural-language/context parser that extracts
  harness line, paths, elements, and run flags; returns guidance when required
  inputs are missing.
- `agent_qwen/qwen_tools.py`: Qwen-Agent adapter classes that expose runtime
  tools to the LLM.

## Harness Lines

`harness_line` is the preferred public name. The older `workflow` field is kept
as a compatibility alias in CLI/RPC outputs.

- `classic`: raw STEM image -> denoise -> template match -> STEM-to-CIF
- `direct`: denoised image -> direct minimal-cell reconstruction
- `compare`: raw STEM image -> shared denoise -> classic branch and direct
  branch under one result

The `compare` line is intentionally sequential for now. It gives a standard
parallel-line result shape without risking GPU/model/MatterSim resource races.

## Public Checkpoint Scope

The public MOE-DIVAESR checkpoint supports the fixed real-space sampling used
during training: **0.1 Å/pixel**. The real multiscale checkpoints are not
included in the public release.

Discover lines programmatically with `AgentQwenHarness.list_lines()` or:

```bash
python -m agent_qwen.cli --list-lines
```

## RPC Entry

Use:

```bash
python agent_qwen_vl_rpc.py --rpc --harness
```

The original LLM tool-calling path remains available without `--harness`.
