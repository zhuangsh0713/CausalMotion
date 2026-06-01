#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROMPT_KEYS = [
    "description",
    "prompt",
    "caption",
    "text",
    "input_prompt",
    "query",
]
EXP_NAME_KEYS = [
    "exp_name",
    "generated_video_name",
    "video_name",
    "name",
    "id",
]
FIRST_FRAME_KEYS = [
    "first_frame_path",
    "first_frame",
    "image_path",
    "image",
]


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_prompt(item: dict[str, str], prompt_key: str | None) -> str:
    if prompt_key:
        text = _normalize_text(item.get(prompt_key))
        if text:
            return text
    for key in PROMPT_KEYS:
        text = _normalize_text(item.get(key))
        if text:
            return text
    return ""


def _resolve_exp_name(item: dict[str, str], exp_name_key: str | None, row_index: int) -> str:
    keys = [exp_name_key] if exp_name_key else EXP_NAME_KEYS
    for key in keys:
        if not key:
            continue
        value = _normalize_text(item.get(key))
        if not value:
            continue
        name = Path(value).name
        stem = Path(name).stem if Path(name).suffix else name
        safe = stem.replace("/", "_").replace("\\", "_").strip()
        if safe:
            return safe
    return f"exp_{row_index:05d}"


def _resolve_first_frame_path(item: dict[str, str], first_frame_key: str | None) -> str | None:
    keys = [first_frame_key] if first_frame_key else FIRST_FRAME_KEYS
    for key in keys:
        if not key:
            continue
        value = _normalize_text(item.get(key))
        if value:
            return value
    return None


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _append_arg(command: list[str], flag: str, value: object, default: object | None = None) -> None:
    if value is None:
        return
    if default is not None and value == default:
        return
    command.extend([flag, str(value)])


def _build_command(args: argparse.Namespace, prompt: str, exp_name: str, first_frame_path: str | None) -> list[str]:
    command = [
        args.python,
        "-m",
        "causalmotion.main",
        "--exp_name",
        exp_name,
        "--prompt",
        prompt,
        "--data_root",
        str(args.data_root),
        "--grounding_model_path",
        args.grounding_model_path,
        "--sam2_checkpoint",
        args.sam2_checkpoint,
        "--sam2_model_config",
        args.sam2_model_config,
        "--ltx_python",
        args.ltx_python,
        "--ltx_pipeline_config",
        args.ltx_pipeline_config,
    ]

    if first_frame_path:
        command.extend(["--first_frame_path", first_frame_path])

    _append_arg(command, "--ltx_num_frames", args.ltx_num_frames, 121)
    _append_arg(command, "--ltx_height", args.ltx_height, 480)
    _append_arg(command, "--ltx_width", args.ltx_width, 720)
    _append_arg(command, "--ltx_seed", args.ltx_seed)
    _append_arg(command, "--conditioning_strength", args.conditioning_strength, 0.9)

    if args.adaptive_match_conditioning:
        command.append("--adaptive_match_conditioning")
    if args.disable_trajectory_guidance:
        command.append("--disable_trajectory_guidance")

    _append_arg(command, "--default_gap_for_unmatched", args.default_gap_for_unmatched, 10)
    _append_arg(command, "--first_frame_strength", args.first_frame_strength, 1.0)
    _append_arg(command, "--matched_strength", args.matched_strength, 0.85)
    _append_arg(command, "--unmatched_strength", args.unmatched_strength, 0.5)
    _append_arg(command, "--trajectory_warp_every", args.trajectory_warp_every, 2)
    _append_arg(command, "--trajectory_alpha", args.trajectory_alpha, 0.3)
    _append_arg(command, "--trajectory_start_ratio", args.trajectory_start_ratio, 0.2)
    _append_arg(command, "--trajectory_end_ratio", args.trajectory_end_ratio, 0.75)
    _append_arg(command, "--trajectory_source_shrink", args.trajectory_source_shrink, 0.8)
    _append_arg(command, "--trajectory_target_expand", args.trajectory_target_expand, 1.1)
    _append_arg(command, "--trajectory_time_travel_repeat", args.trajectory_time_travel_repeat, 1)
    _append_arg(command, "--trajectory_time_travel_start_ratio", args.trajectory_time_travel_start_ratio)
    _append_arg(command, "--trajectory_time_travel_end_ratio", args.trajectory_time_travel_end_ratio)
    _append_arg(command, "--trajectory_time_travel_noise_scale", args.trajectory_time_travel_noise_scale, 0.35)
    return command


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Run CausalMotion on a CSV of prompts.",
    )
    ap.add_argument("--prompts-csv", required=True, help="CSV file containing prompts.")
    ap.add_argument("--data-root", required=True, help="Workspace root for generated outputs.")
    ap.add_argument("--prompt-key", default=None, help="Optional explicit prompt column name.")
    ap.add_argument("--exp-name-key", default=None, help="Optional explicit experiment-name column name.")
    ap.add_argument("--first-frame-key", default=None, help="Optional explicit first-frame column name.")
    ap.add_argument("--start", type=int, default=0, help="Start row index.")
    ap.add_argument("--limit", type=int, default=None, help="Maximum number of rows to run.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip runs with an existing results video.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    ap.add_argument("--python", default=sys.executable, help="Python executable used to launch causalmotion.main.")
    ap.add_argument("--ltx-python", default="python", help="Python executable forwarded to LTX inference.")
    ap.add_argument("--grounding-model-path", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam2-checkpoint", default="./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--ltx-pipeline-config", default="LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml")
    ap.add_argument("--ltx-num-frames", type=int, default=121)
    ap.add_argument("--ltx-height", type=int, default=480)
    ap.add_argument("--ltx-width", type=int, default=720)
    ap.add_argument("--ltx-seed", type=int, default=None)
    ap.add_argument("--conditioning-strength", type=float, default=0.9)
    ap.add_argument("--adaptive-match-conditioning", action="store_true")
    ap.add_argument("--disable-trajectory-guidance", action="store_true")
    ap.add_argument("--default-gap-for-unmatched", type=int, default=10)
    ap.add_argument("--first-frame-strength", type=float, default=1.0)
    ap.add_argument("--matched-strength", type=float, default=0.85)
    ap.add_argument("--unmatched-strength", type=float, default=0.5)
    ap.add_argument("--trajectory-warp-every", type=int, default=2)
    ap.add_argument("--trajectory-alpha", type=float, default=0.3)
    ap.add_argument("--trajectory-start-ratio", type=float, default=0.2)
    ap.add_argument("--trajectory-end-ratio", type=float, default=0.75)
    ap.add_argument("--trajectory-source-shrink", type=float, default=0.8)
    ap.add_argument("--trajectory-target-expand", type=float, default=1.1)
    ap.add_argument("--trajectory-time-travel-repeat", type=int, default=1)
    ap.add_argument("--trajectory-time-travel-start-ratio", type=float, default=None)
    ap.add_argument("--trajectory-time-travel-end-ratio", type=float, default=None)
    ap.add_argument("--trajectory-time-travel-noise-scale", type=float, default=0.35)
    ap.add_argument(
        "--manifest-path",
        default=str(root / "runs" / "run_causalmotion_from_csv_manifest.json"),
        help="Where to write the run manifest.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.prompts_csv).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    rows = _load_rows(csv_path)
    selected = rows[max(args.start, 0):]
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]

    manifest: dict[str, Any] = {
        "description": "Batch CausalMotion run from CSV.",
        "csv_path": str(csv_path),
        "data_root": str(data_root),
        "repo_root": str(root),
        "runs": [],
    }

    for offset, item in enumerate(selected, start=max(args.start, 0)):
        prompt = _resolve_prompt(item, args.prompt_key)
        if not prompt:
            print(f"[Warn] Row {offset} has no prompt, skipping.")
            continue

        exp_name = _resolve_exp_name(item, args.exp_name_key, offset)
        first_frame_path = _resolve_first_frame_path(item, args.first_frame_key)
        result_video = data_root / "results" / f"{exp_name}.mp4"
        result_json = data_root / "json" / f"{exp_name}_results.json"

        if args.skip_existing and (result_video.exists() or result_json.exists()):
            print(f"[Skip] {exp_name} already has outputs.")
            continue

        command = _build_command(
            args=args,
            prompt=prompt,
            exp_name=exp_name,
            first_frame_path=first_frame_path,
        )

        record = {
            "row_index": offset,
            "exp_name": exp_name,
            "prompt": prompt,
            "first_frame_path": first_frame_path,
            "result_video": str(result_video),
            "result_json": str(result_json),
            "command": command,
        }
        manifest["runs"].append(record)

        print(f"\n=== [CausalMotion] row={offset} exp={exp_name} ===")
        print(" ".join(command))

        if args.dry_run:
            continue

        subprocess.run(command, check=True, cwd=str(root))

    _write_json(manifest_path, manifest)
    print(f"\nWrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
