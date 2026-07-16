"""CLI for running the agent_qwen harness without an LLM."""
from __future__ import annotations

import argparse
import json

from .harness import AgentQwenHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_qwen STEM harness line runner")
    parser.add_argument("--list-skills", action="store_true")
    parser.add_argument("--list-lines", action="store_true")
    parser.add_argument("--harness-line", "--workflow", dest="harness_line", choices=("classic", "direct", "compare"), default="classic")
    parser.add_argument("--image", default=None, help="Raw STEM image for classic/compare lines")
    parser.add_argument("--denoised-img", default=None, help="Denoised image for direct line")
    parser.add_argument("--elements", nargs="*", default=None)
    parser.add_argument("--user-message", default="")
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--weight-path", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-property", action="store_true")
    parser.add_argument("--run-confidence", action="store_true")
    args = parser.parse_args()

    harness = AgentQwenHarness.from_defaults(
        work_root=args.work_root,
        weight_path=args.weight_path,
        label_dir=args.label_dir,
        metadata_csv=args.metadata_csv,
        device=args.device,
        dry_run=args.dry_run,
    )
    if args.list_skills:
        print(json.dumps(harness.list_skills(), ensure_ascii=False, indent=2))
        return
    if args.list_lines:
        print(json.dumps(harness.list_lines(), ensure_ascii=False, indent=2))
        return

    if args.harness_line == "classic":
        if not args.image:
            parser.error("--image is required for classic harness line")
        result = harness.run_line(
            "classic",
            image_path=args.image,
            user_message=args.user_message,
            elements=args.elements,
            run_property=not args.skip_property,
            run_confidence=args.run_confidence,
        )
    elif args.harness_line == "direct":
        if not args.denoised_img:
            parser.error("--denoised-img is required for direct harness line")
        if not args.elements:
            parser.error("--elements are required for direct harness line")
        result = harness.run_line(
            "direct",
            denoised_img=args.denoised_img,
            elements=args.elements,
            run_property=not args.skip_property,
            run_confidence=args.run_confidence,
        )
    else:
        if not args.image:
            parser.error("--image is required for compare harness line")
        if not args.elements:
            parser.error("--elements are required for compare harness line")
        result = harness.run_line(
            "compare",
            image_path=args.image,
            user_message=args.user_message,
            elements=args.elements,
            run_property=not args.skip_property,
            run_confidence=args.run_confidence,
        )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
