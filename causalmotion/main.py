from causalmotion.detect_objs import detect_and_segment, extract_bboxes_from_keyframe_folder
from causalmotion.caption_and_keyframe import generate_keyframe_sequence
from causalmotion.traj_plan import vlm_planning
from causalmotion.utils.helpers import io_from_json
from causalmotion.align import (
    TrajectoryBuilder,
    align_keyframes_to_trajectory,
    render_trajectory_video
)
from causalmotion.generate_video import generate_video

import argparse
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time

import os
if "SSL_CERT_FILE" in os.environ:
    if not os.path.exists(os.environ["SSL_CERT_FILE"]):
        del os.environ["SSL_CERT_FILE"]



def parse_args():
    parser = argparse.ArgumentParser(description="CausalMotion")
    
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt that describes the video')
    parser.add_argument('--data_root', required=True, help='dataset root')
    # TODO: First frame can be optional
    parser.add_argument('--first_frame_path', default=None, required=False, help='File path to the first frame image of the video sequence')
    parser.add_argument('--grounding_model_path', required=True, help='File path to the first frame image of the video sequence')
    parser.add_argument('--sam2_checkpoint', required=True, help='File path to the first frame image of the video sequence')
    parser.add_argument('--sam2_model_config', default="configs/sam2.1/sam2.1_hiera_l.yaml", help='File path to the first frame image of the video sequence')
    parser.add_argument('--ltx_python', default="python", help='Python executable used to run LTX inference.')
    parser.add_argument('--ltx_pipeline_config', default="LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml")
    parser.add_argument('--ltx_num_frames', type=int, default=121)
    parser.add_argument('--ltx_height', type=int, default=480)
    parser.add_argument('--ltx_width', type=int, default=720)
    parser.add_argument('--ltx_seed', type=int, default=None)
    parser.add_argument('--conditioning_strength', type=float, default=0.9)
    parser.add_argument('--adaptive_match_conditioning', action='store_true', default=False)
    parser.add_argument('--default_gap_for_unmatched', type=int, default=10)
    parser.add_argument('--first_frame_strength', type=float, default=1.0)
    parser.add_argument('--matched_strength', type=float, default=0.85)
    parser.add_argument('--unmatched_strength', type=float, default=0.5)
    parser.add_argument('--disable_trajectory_guidance', action='store_true', default=False)
    parser.add_argument('--trajectory_warp_every', type=int, default=2)
    parser.add_argument('--trajectory_alpha', type=float, default=0.3)
    parser.add_argument('--trajectory_start_ratio', type=float, default=0.2)
    parser.add_argument('--trajectory_end_ratio', type=float, default=0.75)
    parser.add_argument('--trajectory_source_shrink', type=float, default=0.8)
    parser.add_argument('--trajectory_target_expand', type=float, default=1.1)
    parser.add_argument('--trajectory_time_travel_repeat', type=int, default=1)
    parser.add_argument('--trajectory_time_travel_start_ratio', type=float, default=None)
    parser.add_argument('--trajectory_time_travel_end_ratio', type=float, default=None)
    parser.add_argument('--trajectory_time_travel_noise_scale', type=float, default=0.35)
    
    args = parser.parse_args()
    return args


def _print_component_timing(name, seconds):
    print(f"[Timing] {name}: {seconds:.2f}s")

if __name__ == "__main__":
    args = parse_args()
    total_start = time.perf_counter()
    runtime_seconds = {}
    
    exp_name = args.exp_name
    prompt = args.prompt
    data_root = args.data_root
    first_frame_path = args.first_frame_path
    
    keyframes_root = Path(data_root) / 'keyframes' / f'{exp_name}'
    output_json_path = Path(data_root) / 'json' / f'{exp_name}_results.json'
    
    
    
    # 1. Generate keyframe sequence
    print("Generating keyframe sequence...")
    reasoning_start = time.perf_counter()
    data_dict = generate_keyframe_sequence(prompt, data_root, first_frame_path, exp_name)
    data_dict["Input Prompt"] = prompt
    runtime_seconds["reasoning"] = time.perf_counter() - reasoning_start
    _print_component_timing("reasoning", runtime_seconds["reasoning"])
    
    # 2. VLM planning
    vlm_planning_start = time.perf_counter()
    data_dict['first_frame'] = Path(data_root) / 'keyframes' / f'{exp_name}' / '0.png'
        
    print("First frame used for grounding:", data_dict['first_frame'])
    sam2_model_config = args.sam2_model_config
    sam2_checkpoint = args.sam2_checkpoint
    grounding_model_path = args.grounding_model_path
    # detection_root = data_root / 'detections'
    # output_root = detection_root / f'{exp_name}'
    
    print("Loading SAM2 and Grounding models...")
    DEVICE = "cuda"
    sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(grounding_model_path)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_path).to(DEVICE)
    
    print("Detecting and segmenting objects in keyframes...")
    data_dict = detect_and_segment(data_dict, sam2_predictor=sam2_predictor, processor=processor, grounding_model=grounding_model)
    
    # 3. Generate trajectory planning
    print("Generating trajectory planning...")
    data_dict = vlm_planning(data_dict)
    data_dict["first_frame"] = str(data_dict["first_frame"])
    runtime_seconds["vlm_planning"] = time.perf_counter() - vlm_planning_start
    _print_component_timing("vlm planning", runtime_seconds["vlm_planning"])
    io_from_json(output_json_path, data_dict, io_type='w')
    print(f"Saved intermediate results to {output_json_path}")
    
    # 4. Align 
    print("Aligning keyframes to trajectory...")
    align_start = time.perf_counter()
    builder = TrajectoryBuilder()
    frames, object_appearance = builder.read_bbox_json(data_dict['vlm_planning'])
    trajectory, obj_ids = builder.build_and_interpolate(frames, object_appearance)
    
    keyframe_result = extract_bboxes_from_keyframe_folder(keyframes_root, data_dict['key_physic_object'], sam2_predictor=sam2_predictor, processor=processor, grounding_model=grounding_model)
    print("Keyframe result:", keyframe_result)
    
    mapping = align_keyframes_to_trajectory(data_dict, keyframe_result, trajectory)
    print("Alignment mapping:", mapping)
    data_dict['mapping'] = mapping
    data_dict['keyframe_result'] = keyframe_result
    io_from_json(output_json_path, data_dict, io_type='w')
    
    render_trajectory_video(trajectory, mapping, keyframes_root, Path(data_root) / 'alignment_videos' / f'{exp_name}.mp4')
    runtime_seconds["alignment"] = time.perf_counter() - align_start
    _print_component_timing("alignment", runtime_seconds["alignment"])
    
    # 5. Generate video
    latent_guidance_start = time.perf_counter()
    data_dict['exp_name'] = exp_name
    generate_video(
        data_dict,
        mapping,
        data_root,
        generation_options={
            "python": args.ltx_python,
            "pipeline_config": args.ltx_pipeline_config,
            "num_frames": args.ltx_num_frames,
            "height": args.ltx_height,
            "width": args.ltx_width,
            "seed": args.ltx_seed,
            "conditioning_strength": args.conditioning_strength,
            "adaptive_match_conditioning": args.adaptive_match_conditioning,
            "default_gap_for_unmatched": args.default_gap_for_unmatched,
            "first_frame_strength": args.first_frame_strength,
            "matched_strength": args.matched_strength,
            "unmatched_strength": args.unmatched_strength,
            "disable_trajectory_guidance": args.disable_trajectory_guidance,
            "trajectory_warp_every": args.trajectory_warp_every,
            "trajectory_alpha": args.trajectory_alpha,
            "trajectory_start_ratio": args.trajectory_start_ratio,
            "trajectory_end_ratio": args.trajectory_end_ratio,
            "trajectory_source_shrink": args.trajectory_source_shrink,
            "trajectory_target_expand": args.trajectory_target_expand,
            "trajectory_time_travel_repeat": args.trajectory_time_travel_repeat,
            "trajectory_time_travel_start_ratio": args.trajectory_time_travel_start_ratio,
            "trajectory_time_travel_end_ratio": args.trajectory_time_travel_end_ratio,
            "trajectory_time_travel_noise_scale": args.trajectory_time_travel_noise_scale,
        },
    )
    runtime_seconds["latent_guidance_generation"] = time.perf_counter() - latent_guidance_start
    _print_component_timing("latent guidance generation", runtime_seconds["latent_guidance_generation"])

    runtime_seconds["total"] = time.perf_counter() - total_start
    _print_component_timing("total pipeline", runtime_seconds["total"])
    io_from_json(output_json_path, data_dict, io_type='w')

    print("\n=== Runtime Summary (seconds) ===")
    print(f"reasoning: {runtime_seconds['reasoning']:.2f}")
    print(f"vlm planning: {runtime_seconds['vlm_planning']:.2f}")
    print(f"alignment: {runtime_seconds['alignment']:.2f}")
    print(f"latent guidance generation: {runtime_seconds['latent_guidance_generation']:.2f}")
    print(f"total: {runtime_seconds['total']:.2f}")
    
