import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _join_captions(captions: object) -> str:
    if isinstance(captions, str):
        return _normalize_text(captions)
    if isinstance(captions, list):
        parts = [_normalize_text(item) for item in captions if isinstance(item, str)]
        return " ".join([part for part in parts if part])
    return ""


def compose_prompt(base_prompt: str, json_prompt: str) -> str:
    base_text = _normalize_text(base_prompt)
    json_text = _normalize_text(json_prompt)
    if not base_text:
        return json_text
    if not json_text:
        return base_text
    if base_text in json_text:
        return json_text
    if json_text in base_text:
        return base_text
    return f"{base_text} {json_text}"


def make_run_output_dir(output_root: str | Path, exp_name: str) -> Path:
    output_root = Path(output_root)
    safe_name = exp_name.replace("/", "_").replace("\\", "_").strip() or "exp"
    return output_root / f"{safe_name}_{uuid.uuid4().hex[:10]}"


def move_only_videos(
    src_dir: str | Path,
    dst_root: str | Path,
    exp_name: str | None = None,
):
    src_dir = Path(src_dir)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {src_dir}")

    mp4_files = sorted(src_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"[Warning] No mp4 files found in {src_dir}")
        return

    for idx, mp4 in enumerate(mp4_files):
        if exp_name is not None:
            suffix = f"_{idx}" if len(mp4_files) > 1 else ""
            new_name = f"{exp_name}{suffix}.mp4"
        else:
            new_name = mp4.name

        target = dst_root / new_name
        shutil.move(str(mp4), str(target))
        print(f"Moved & renamed video: {mp4} -> {target}")


def sanitize_frame_indices(frame_indices: list[int], num_frames: int) -> list[int]:
    if not frame_indices:
        return []

    upper = max(0, num_frames - 1)
    sanitized: list[int] = []
    prev = -1
    for i, t in enumerate(frame_indices):
        tt = int(max(0, min(upper, int(t))))
        if i > 0 and tt <= prev:
            tt = min(upper, prev + 1)
        sanitized.append(tt)
        prev = tt
    return sanitized


def infer_match_flags_from_default_gap(
    frame_indices: list[int],
    default_gap: int,
) -> list[bool]:
    if not frame_indices:
        return []

    flags: list[bool] = [True]
    for i in range(1, len(frame_indices)):
        gap = frame_indices[i] - frame_indices[i - 1]
        flags.append(gap > 0 and gap != default_gap)
    return flags


def _redistribute_between(
    indices: list[int],
    left_i: int,
    right_i: int,
    right_t: int,
) -> None:
    span = right_i - left_i - 1
    if span <= 0:
        return

    left_t = indices[left_i]
    total_room = right_t - left_t
    min_gap = 1 if total_room >= (span + 1) else 0
    prev = left_t

    for k in range(1, span + 1):
        remain = span - k
        ideal = round(left_t + (k / (span + 1)) * (right_t - left_t))
        min_t = prev + min_gap
        max_t = right_t - (remain + 1) * min_gap
        if max_t < min_t:
            max_t = min_t
        tt = int(min(max(ideal, min_t), max_t))
        indices[left_i + k] = tt
        prev = tt


def relocate_unmatched_frame_indices(
    frame_indices: list[int],
    matched_flags: list[bool],
    num_frames: int,
) -> list[int]:
    if not frame_indices:
        return []
    if len(frame_indices) != len(matched_flags):
        raise ValueError("frame_indices and matched_flags must have the same length.")

    indices = sanitize_frame_indices(frame_indices, num_frames)
    n = len(indices)
    flags = matched_flags[:]
    flags[0] = True
    anchors = [i for i, ok in enumerate(flags) if ok] or [0]

    for i in range(len(anchors) - 1):
        left_i = anchors[i]
        right_i = anchors[i + 1]
        _redistribute_between(indices, left_i, right_i, indices[right_i])

    last_anchor = anchors[-1]
    if last_anchor < n - 1:
        _redistribute_between(indices, last_anchor, n, max(0, num_frames - 1))

    return sanitize_frame_indices(indices, num_frames)


def build_adaptive_strengths(
    matched_flags: list[bool],
    first_strength: float,
    matched_strength: float,
    unmatched_strength: float,
) -> list[float]:
    if not matched_flags:
        return []

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    strengths: list[float] = []
    for i, ok in enumerate(matched_flags):
        if i == 0:
            strengths.append(clamp01(first_strength))
        elif ok:
            strengths.append(clamp01(matched_strength))
        else:
            strengths.append(clamp01(unmatched_strength))
    return strengths


def resolve_uniform_conditioning_strength(count: int, strength: float) -> list[float]:
    if count <= 0:
        return []
    value = max(0.0, min(1.0, float(strength)))
    return [value] * count


def _sort_keyframes(paths: list[Path]) -> list[Path]:
    def key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.name)

    return sorted(paths, key=key)


def _write_mapping_file(
    data_root: Path,
    exp_name: str,
    mapping: list[int],
    keyframe_result: dict[str, Any],
    num_frames: int,
    data_dict: dict[str, Any],
) -> Path:
    mapping_dir = data_root / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = mapping_dir / "all_mappings.json"

    payload = {}
    if mapping_path.exists():
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    object_ids = []
    planning_frames = (data_dict.get("vlm_planning") or {}).get("Frames") or {}
    for frame_items in planning_frames.values():
        if isinstance(frame_items, list):
            for item in frame_items:
                obj_id = item.get("id")
                if obj_id is not None and obj_id not in object_ids:
                    object_ids.append(obj_id)

    payload[exp_name] = {
        "mapping": [int(v) for v in mapping],
        "keyframe_result": keyframe_result or {},
        "trajectory_info": {
            "num_frames": int(num_frames),
            "object_ids": object_ids,
        },
    }
    mapping_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mapping_path


def _build_prompt(data_dict: dict[str, Any]) -> str:
    base_prompt = str(
        data_dict.get("Input Prompt")
        or data_dict.get("prompt")
        or data_dict.get("Prompt")
        or ""
    ).strip()
    captions_text = _join_captions(data_dict.get("captions"))
    concise_prompt = _normalize_text(str(data_dict.get("Concise Prompt") or ""))
    fallback_prompt = " ".join([part for part in [concise_prompt, str(data_dict.get("Consequences") or "").strip()] if part])
    return compose_prompt(base_prompt, captions_text) or compose_prompt(base_prompt, fallback_prompt) or base_prompt or captions_text or fallback_prompt


def _append_arg(
    command: list[str],
    flag: str,
    value: object,
    default: object | None = None,
) -> None:
    if value is None:
        return
    if default is not None and value == default:
        return
    command.extend([flag, str(value)])


def generate_video(
    data_dict: dict[str, Any],
    index_mapping: list[int],
    data_root: str | Path,
    generation_options: Optional[dict[str, Any]] = None,
):
    generation_options = generation_options or {}
    data_root = Path(data_root)
    workspace_root = Path(__file__).resolve().parents[1]

    exp_name = str(data_dict["exp_name"])
    prompt = _build_prompt(data_dict)
    if not prompt:
        raise ValueError(f"{exp_name}: resolved prompt is empty.")

    conditioning_folder = data_root / "keyframes" / exp_name
    conditioning_paths = _sort_keyframes(list(conditioning_folder.glob("*.png")))
    if len(conditioning_paths) != len(index_mapping):
        raise ValueError(
            f"Number of keyframes ({len(conditioning_paths)}) "
            f"!= number of start frames ({len(index_mapping)})"
        )

    python_exe = str(generation_options.get("python", "python"))
    height = int(generation_options.get("height", 480))
    width = int(generation_options.get("width", 720))
    num_frames = int(generation_options.get("num_frames", 121))
    seed = generation_options.get("seed")
    pipeline_config = generation_options.get(
        "pipeline_config",
        "LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml",
    )
    output_root = Path(generation_options.get("output_root", data_root / "ltx_runs"))
    result_root = Path(generation_options.get("result_root", data_root / "results"))
    disable_trajectory_guidance = bool(
        generation_options.get("disable_trajectory_guidance", False)
    )
    adaptive_match_conditioning = bool(
        generation_options.get("adaptive_match_conditioning", True)
    )
    default_gap = int(generation_options.get("default_gap_for_unmatched", 10))

    conditioning_start_frames = sanitize_frame_indices(
        [int(v) for v in index_mapping],
        num_frames,
    )

    
    match_flags = infer_match_flags_from_default_gap(
        conditioning_start_frames,
        default_gap=default_gap,
    )
    conditioning_start_frames = relocate_unmatched_frame_indices(
        conditioning_start_frames,
        matched_flags=match_flags,
        num_frames=num_frames,
    )
    
    if adaptive_match_conditioning:
        conditioning_strengths = build_adaptive_strengths(
            matched_flags=match_flags,
            first_strength=float(generation_options.get("first_frame_strength", 1.0)),
            matched_strength=float(generation_options.get("matched_strength", 0.85)),
            unmatched_strength=float(generation_options.get("unmatched_strength", 0.5)),
        )
    else:
        conditioning_strengths = resolve_uniform_conditioning_strength(
            len(conditioning_paths),
            float(generation_options.get("conditioning_strength", 0.9)),
        )

    results_json = data_root / "json" / f"{exp_name}_results.json"
    mapping_path = _write_mapping_file(
        data_root=data_root,
        exp_name=exp_name,
        mapping=index_mapping,
        keyframe_result=data_dict.get("keyframe_result") or {},
        num_frames=num_frames,
        data_dict=data_dict,
    )

    command = [
        python_exe,
        str((workspace_root / "LTX-Video" / "inference.py").resolve()),
        "--prompt",
        prompt,
        "--num_frames",
        str(num_frames),
        "--height",
        str(height),
        "--width",
        str(width),
        "--pipeline_config",
        str(pipeline_config),
    ]

    run_output_dir = make_run_output_dir(output_root, exp_name)
    command += ["--output_path", str(run_output_dir)]

    if not disable_trajectory_guidance:
        command += [
            "--trajectory_path",
            str(results_json.resolve()),
            "--trajectory_mapping_path",
            str(mapping_path.resolve()),
        ]
        _append_arg(command, "--trajectory_warp_every", generation_options.get("trajectory_warp_every"), 2)
        _append_arg(command, "--trajectory_time_travel_repeat", generation_options.get("trajectory_time_travel_repeat"), 1)
        _append_arg(command, "--trajectory_time_travel_start_ratio", generation_options.get("trajectory_time_travel_start_ratio"))
        _append_arg(command, "--trajectory_time_travel_end_ratio", generation_options.get("trajectory_time_travel_end_ratio"))
        _append_arg(command, "--trajectory_time_travel_noise_scale", generation_options.get("trajectory_time_travel_noise_scale"), 0.35)
        _append_arg(command, "--trajectory_alpha", generation_options.get("trajectory_alpha"), 0.3)
        _append_arg(command, "--trajectory_start_ratio", generation_options.get("trajectory_start_ratio"), 0.2)
        _append_arg(command, "--trajectory_end_ratio", generation_options.get("trajectory_end_ratio"), 0.75)
        _append_arg(command, "--trajectory_source_shrink", generation_options.get("trajectory_source_shrink"), 0.8)
        _append_arg(command, "--trajectory_target_expand", generation_options.get("trajectory_target_expand"), 1.1)

    command += ["--conditioning_media_paths", *map(str, conditioning_paths)]
    command += ["--conditioning_start_frames", *map(str, conditioning_start_frames)]
    command += ["--conditioning_strengths", *map(str, conditioning_strengths)]
    _append_arg(command, "--seed", seed)

    print("Executing command:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True, cwd=str(workspace_root))
        print("Video generation completed successfully.")
        move_only_videos(
            src_dir=run_output_dir,
            dst_root=result_root,
            exp_name=exp_name,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during video generation: {e}")

if __name__ == "__main__":
    # args.parse_args()
    data_dict = {
        "exp_name": "output_video_76",
        "prompt": "A large number of soap bubbles are floating in the air under the sunlight.",
        "captions": ['Soap bubbles float gracefully in the sunlit air.A serene outdoor scene bathed in the warm glow of sunlight. Countless soap bubbles, each shimmering with iridescent colors, are suspended in the air. The camera is positioned at a low angle, looking upwards towards the sky, capturing the delicate spheres against the backdrop of a clear blue sky and scattered fluffy clouds. The gentle breeze causes the bubbles to drift slowly, creating a mesmerizing dance of light and color. In the background, faint silhouettes of trees and a distant building can be seen, but the focus remains on the floating bubbles.', 'Two soap bubbles are touching each other in the air. The point of contact between them is visible, and their shapes begin to merge slightly.', 'A single, larger soap bubble is floating in the air. The merged shape from the previous two bubbles is now a unified, slightly elongated form.', 'A soap bubble is in the process of bursting. The soapy film has thinned to a point where it can no longer hold its shape, and a small puff of air is being released as the bubble disappears.'],
        "Consequences": "The soap bubbles will continue to drift through the air, gradually changing shape as they encounter slight air currents. Some bubbles may collide and merge, forming larger bubbles, while others may burst due to the thinning of their soapy film, releasing a puff of air and disappearing into the atmosphere.",
        "Concise Prompt": "Soap bubbles float gracefully in the sunlit air.",        
        "keyframe_result": None,
    }
    
    index_mapping = [0, 30, 60, 90]
    data_root = ""
    generate_video(
        data_dict=data_dict,
        index_mapping=index_mapping,
        data_root=data_root,
        generation_options={
            "python": "python3",
            "height": 480,
            "width": 720,
            "num_frames": 121,
            "seed": 42,
            "pipeline_config": "LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml",
            "output_root": f"{data_root}/ltx_runs",
            "result_root": f"{data_root}/results",
            "disable_trajectory_guidance": True,
            "adaptive_match_conditioning": False,
            "default_gap_for_unmatched": 10,
            "first_frame_strength": 1.0,
            "matched_strength": 0.85,
            "unmatched_strength": 0.5,
        },
    )
    