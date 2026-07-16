# AutoMat

<p align="center">
  <img src="https://github.com/yyt-2378/AutoMat/blob/main/AutoMat_Logo.png" alt="AutoMat Logo" width="250"/>
</p>

AutoMat is an end-to-end agent framework for going from scanning transmission
electron microscopy (STEM) images to crystal-structure reconstruction and
materials-property analysis.

<h2 align="center">AutoMat has been accepted to ICML 2026.</h2>

The current release also includes an explicit `agent_qwen` harness layer with
standardized harness lines, natural-language request intake, Markdown skills,
and a small browser demo for interactive testing.

## Resources

- Paper: [AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use](https://arxiv.org/abs/2505.12650)
- Dataset: [STEM2Mat on Hugging Face](https://huggingface.co/datasets/yaotianvector/STEM2Mat)
- Code: [yyt-2378/AutoMat](https://github.com/yyt-2378/AutoMat)

## Core Pipeline

AutoMat is organized around four scientific modules:

| Module | Role |
| --- | --- |
| MOE-DIVAESR | Pattern-adaptive STEM denoising and super-resolution |
| Template Retriever | Physics-constrained structure-template search |
| STEM2CIF | Atomic-coordinate extraction and CIF generation |
| MatterSim Wrapper | Optional relaxation and property evaluation |

The agent harness makes these steps explicit and reproducible instead of
leaving step selection entirely to an LLM.

## What Is New

- `agent_qwen` harness lines for reproducible execution.
- Natural-language intake that parses user requests into structured harness
  inputs.
- Markdown skills under `agent_qwen/skills/*/SKILL.md`.
- Uncertainty and confidence estimation for reconstructed structures.
- CLI, RPC, and browser-demo entry points.
- Compatibility wrappers for the historical `structure_recongnition` import
  name.

## Repository Layout

```text
AutoMat-main/
├── agent_qwen/                         # Harness, runtime tools, skills, demo
│   ├── harness.py                      # Standard harness-line runner
│   ├── request_intake.py               # Natural-language request parser
│   ├── runtime_skills.py               # Python wrappers around pipeline tools
│   ├── confidence_estimation.py        # Uncertainty/confidence report logic
│   └── skills/*/SKILL.md               # Normative Markdown skills
├── agent_qwen_vl_rpc.py                # RPC entry point for harness/LLM modes
├── src/
│   ├── pipline_framework.py            # Original STEM -> CIF -> property tools
│   ├── preprocess_model/               # MOE-DIVAESR model code
│   └── structure_paired_reconstruction # Template and atom reconstruction code
├── src/structure_recongnition/         # Compatibility aliases
├── utils/fft_convert.py                # FFT helper for confidence estimation
├── tests/test_agent_qwen_harness.py    # Harness/intake tests
└── harness_runs/                       # Local outputs, ignored by git
```

## Installation

A conda environment is recommended.

```bash
conda env create -f src/img2struc.yaml
conda activate img2struc
```

If you already have a compatible environment, install the Python requirements:

```bash
pip install -r src/requirement.txt
```

For the harness-only dry-run tests, heavy model dependencies are not required.
For real reconstruction, the environment must include the MOE-DIVAESR, OpenCV,
ASE/pymatgen, scipy/sklearn, and optional MatterSim dependencies.

## Weights And Data

Real reconstruction needs:

- MOE-DIVAESR checkpoint, for example `moe_model.ckpt`
- template label directory
- metadata CSV with `material_id` and `elements`
- input STEM image

The pretrained MOE-DIVAESR checkpoint can be downloaded from:
[Google Drive](https://drive.google.com/file/d/1p3rctPDUn0KHJ81pZV8MgBSYFvyMliZ0/view?usp=drive_link)

> **Public checkpoint scope:** The released checkpoint was trained at a fixed
> real-space sampling of **0.1 Å/pixel**. Inputs should be calibrated or
> resampled to 0.1 Å/pixel before inference. The real multiscale checkpoints
> for varying pixel sizes are **not included in this public release version**.


The benchmark data are released as
[STEM2Mat on Hugging Face](https://huggingface.co/datasets/yaotianvector/STEM2Mat).

You can pass these explicitly:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image examples/stem.png \
  --elements Si Sn \
  --weight-path artifacts/moe_model.ckpt \
  --label-dir artifacts/labels \
  --metadata-csv artifacts/property.csv \
  --work-root harness_runs/example \
  --run-confidence \
  --skip-property
```

Or configure defaults with environment variables:

```bash
export AUTOMAT_WEIGHT_PATH=artifacts/moe_model.ckpt
export AUTOMAT_LABEL_DIR=artifacts/labels
export AUTOMAT_METADATA_CSV=artifacts/property.csv
export AUTOMAT_WORK_ROOT=harness_runs/current
```

Large checkpoints and local harness outputs are intentionally ignored by git.

## Harness Lines

`harness_line` is the preferred public name. The older `workflow` field is
still accepted as a compatibility alias.

| Harness line | Required inputs | Execution |
| --- | --- | --- |
| `classic` | `image_path`, `elements` | raw STEM -> denoise -> template match -> STEM2CIF |
| `direct` | `denoised_img`, `elements` | denoised image -> direct reconstruction |
| `compare` | `image_path`, `elements` | shared denoise -> classic branch and direct branch |

List the available lines:

```bash
python -m agent_qwen.cli --list-lines
```

## Quick Start

Dry-run the harness without loading model weights:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image examples/sample.png \
  --elements Mo S \
  --dry-run \
  --run-confidence \
  --skip-property
```

Run the compare line:

```bash
python -m agent_qwen.cli \
  --harness-line compare \
  --image examples/sample.png \
  --elements Mo S \
  --dry-run \
  --run-confidence \
  --skip-property
```

Run a real reconstruction with weights:

```bash
python -m agent_qwen.cli \
  --harness-line classic \
  --image examples/stem.png \
  --elements Si Sn \
  --weight-path artifacts/moe_model.ckpt \
  --label-dir artifacts/labels \
  --metadata-csv artifacts/property.csv \
  --work-root harness_runs/real_example \
  --run-confidence \
  --skip-property
```

Remove `--skip-property` to include MatterSim relaxation and property
prediction, assuming the MatterSim environment and weights are available.

## Natural-Language RPC

`agent_qwen_vl_rpc.py` accepts natural-language requests and uses
`agent_qwen/request_intake.py` to normalize them before running the harness.

Example:

```bash
printf '%s\n' \
  '{"query":"Please reconstruct examples/sample.png. Elements: Mo S. Also estimate uncertainty and skip property prediction.","context":{"dry_run":true}}' \
  | python agent_qwen_vl_rpc.py --rpc --harness --dry_run
```

If required information is missing, the agent returns `need_input: true` with
specific guidance instead of running with incomplete state:

```json
{
  "ok": false,
  "need_input": true,
  "missing_required": ["image_path", "elements"],
  "guidance": [
    "Please provide the raw STEM image path.",
    "Please provide the element list."
  ]
}
```

The parser recognizes common phrases such as:

- "elements: Mo S" or "元素: Mo S"
- "estimate uncertainty" / "confidence" / "不确定性" / "置信度"
- "skip property" / "do not run relaxation" / "跳过物性"
- "compare classic and direct" / "比较两条线"

## Browser Demo

Start the local demo server:

```bash
python -m agent_qwen.demo_server --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The browser demo calls the same request intake and harness APIs used by the RPC
entry point. It supports `classic`, `direct`, and `compare` lines, dry-run mode,
confidence estimation, and missing-input guidance.

## Skills

AutoMat uses Markdown skills as normative instructions for agents and
maintainers:

```text
agent_qwen/skills/stem-reconstruction/SKILL.md
agent_qwen/skills/confidence-estimation/SKILL.md
```

Important distinction:

- `SKILL.md` files define the intended behavior and maintenance rules.
- `runtime_skills.py` contains executable Python wrappers around pipeline tools.
- `harness.py` selects and executes harness lines.
- `request_intake.py` parses user language into structured harness inputs.

## Extending AutoMat

To add a new runtime capability:

1. Add a tool wrapper in `agent_qwen/runtime_skills.py`.
2. Register it in `default_registry()`.
3. Add or update a harness line in `agent_qwen/harness.py`.
4. Document the behavior in the relevant `SKILL.md`.
5. Add tests in `tests/test_agent_qwen_harness.py`.

To add a new harness line:

1. Add a `HarnessLineSpec` entry in `agent_qwen/harness.py`.
2. Implement `run_<name>_line(...)`.
3. Route it through `AgentQwenHarness.run_line(...)`.
4. Update `request_intake.py` if natural-language selection is needed.
5. Add CLI/RPC/demo tests.

Keep generated files under `harness_runs/` or another ignored output directory.

## Training MOE-DIVAESR

You can reuse [STEM2Mat](https://huggingface.co/datasets/yaotianvector/STEM2Mat)
or prepare your own paired low-resolution/high-resolution STEM tiles:

```text
dataset/
└── SRDATA/
    ├── training/
    │   ├── LR_original/
    │   └── HR/
    ├── validation/
    │   ├── LR_original/
    │   └── HR/
    └── test/
        ├── LR_original/
        └── HR/
```

Launch training:

```bash
python src/ensemble_model_train.py \
  --config src/preprocess_model/configs/vae.yaml \
  --dir_data dataset
```

`src/preprocess_model/configs/vae.yaml` is the bundled example training
configuration. Adapt its `data_params`, `trainer_params`, and logging paths for
your dataset. The resulting checkpoint can be passed to the harness with
`--weight-path`.

## Legacy Agent Entry Points

The repository still includes earlier agent scripts under `src/`:

```bash
export AUTOMAT_LLM_API_KEY='<your-key>'
python src/agent_based.py \
  --image_path examples/stem.png \
  --work_root results \
  --user_message "Elements: Al, Sb; dose = 30k"
```

Qwen-VL function-calling mode:

```bash
export DASHSCOPE_API_KEY='<your-key>'
python src/agent_qwen_vl.py --model qwen-plus-2025-04-28
```

For reproducible experiments, prefer the explicit `agent_qwen` harness lines.

## Outputs

Harness outputs are written under `--work-root`:

| Folder | Contents |
| --- | --- |
| `01_recon/` | Denoised or super-resolved STEM image |
| `02_label/` | Best-matched structure template |
| `03_recon_cif/` | Reconstructed CIF files |
| `04_relax/` | MatterSim-relaxed structure and property results |
| `confidence/` | JSON uncertainty/confidence report |

## Testing

Run the harness and intake tests:

```bash
python -m pytest tests/test_agent_qwen_harness.py -q
```

The current harness test suite covers:

- standard harness-line discovery
- dry-run execution for `classic`, `direct`, and `compare`
- natural-language intake and missing-input guidance
- CLI and RPC compatibility

## Citation

If you use AutoMat or STEM2Mat, please cite the paper:

```bibtex
@article{yang2025automat,
  title = {AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use},
  author = {Yang, Yaotian and Tang, Yiwen and Chen, Yizhe and Chen, Xiao and Qiu, Jiangjie and Xiong, Hao and Yin, Haoyu and Luo, Zhiyao and Zhang, Yifei and Tao, Sijia and Li, Wentao and Zhang, Qinghua and Li, Yuqiang and Ouyang, Wanli and Zhao, Bin and Wang, Xiaonan and Wei, Fei},
  journal = {arXiv preprint arXiv:2505.12650},
  year = {2025},
  url = {https://arxiv.org/abs/2505.12650}
}
```

The ICML 2026 proceedings citation will be updated when the final metadata is
available.

## License

See [LICENSE](LICENSE).
