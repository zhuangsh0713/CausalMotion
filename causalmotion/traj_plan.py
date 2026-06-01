from openai import OpenAI
import openai
import json
import os
import argparse
from pathlib import Path
from PIL import Image
import base64
from copy import deepcopy
from causalmotion.utils.template import template_library
from causalmotion.utils.helpers import encode_image, get_vlm_plan_to_json, io_from_json


openai_base_url = os.getenv("OPENAI_BASE_URL")
vlm_model = os.getenv("CAUSALMOTION_VLM_MODEL", "qwen2.5-vl-72b-instruct")

def image_bytes_to_data_url(img_bytes, format="PNG"):
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/{format.lower()};base64,{base64_str}"


def image_path_to_data_url(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return image_bytes_to_data_url(img_bytes)


def vlm_planning(data_dict):
    """
    Performs visual-language model (VLM) based motion planning using GPT-4o,
    conditioned on an input image, segmentation map, and object annotations.

    This function constructs a prompt using visual inputs and a natural language caption,
    sends it to OpenAI's GPT-4o, and parses its structured output to generate
    a frame-by-frame plan of object movements or changes.

    Parameters:
        data_dict (dict): A dictionary containing:
            - 'first_frame': Path to the original image.
            - 'seg_mask_path': Path to the segmentation map.
            - 'annotations': List of object bounding box annotations.
            - 'Consequences': Refined caption/description of the scene.
            - 'Category': Physics-based tag for selecting prompt templates.

    Returns:
        dict: Updated `data_dict` with an additional field 'vlm_planning', which contains:
            - 'Reasoning': The raw text output from GPT-4o.
            - 'Frames': Parsed frame-by-frame bounding box predictions.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize OpenAI API client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url,
    )

    # Load image paths
    first_frame = data_dict['first_frame']
    first_frame_seg = data_dict['seg_mask_path']

    # Construct bounding box list with IDs and names
    first_frame_box = []
    for idx, annotation in enumerate(data_dict['annotations']):
        box_info = {
            "id": idx,
            "name": annotation['class_name'],
            "box": annotation['bbox']
        }
        first_frame_box.append(box_info)

    # Encode images to base64 strings for use in GPT-4o image input
    # first_frame_b64_str = image_bytes_to_data_url(first_frame)
    # first_frame_seg_b64_str = image_bytes_to_data_url(first_frame_seg)

    # Retrieve natural language prompt and physics tag (e.g., "collision", "falling")
    prompt = data_dict['Consequences']
    physic_tag = data_dict['Category']

    # Load prompt template based on physics category
    # Avoid mutating global templates across repeated calls.
    messages = deepcopy(template_library[physic_tag])

    # messages[-1]['content'][0]['image'] = first_frame_b64_str
    # messages[-1]['content'][1]['image'] = first_frame_seg_b64_str

    messages[-1]["content"][0]["text"] = f"""initial_boxes: {first_frame_box}\ncaption: {prompt}"""
    messages[-1]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": image_path_to_data_url(first_frame)
        }
    })
    messages[-1]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": image_path_to_data_url(first_frame_seg)
        }
    })

    response = client.chat.completions.create(
        model=vlm_model,
        messages = messages,
    )

    # Extract the text response from GPT
    output_text = response.choices[0].message.content

    # Parse the GPT response into a structured format
    bboxs = get_vlm_plan_to_json(output_text)

    # Compose the result dictionary
    results = {
        "Reasoning": output_text,
        "Frames":  bboxs
    }
    print("VLM Planning Results:", results)

    # Add VLM planning result to the original input dictionary
    data_dict['vlm_planning'] = results

    return data_dict


def regenerate_vlm_planning_inplace(results_json_path):
    """
    Recompute vlm_planning for an existing *_results.json and overwrite it in place.
    Also removes stale 'mapping' to prevent using outdated alignment.
    """
    results_json_path = Path(results_json_path)
    info_json = io_from_json(results_json_path, io_type='r')

    required_keys = ["seg_mask_path", "annotations", "Consequences", "Category", "first_frame"]
    missing = [k for k in required_keys if k not in info_json]
    if missing:
        raise KeyError(
            f"{results_json_path} missing required keys for vlm_planning: {missing}"
        )

    info_json = vlm_planning(info_json)

    if "mapping" in info_json:
        info_json.pop("mapping", None)
        print(f"Removed stale mapping in {results_json_path.name}")

    io_from_json(results_json_path, data=info_json, io_type='w')
    return info_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate vlm_planning for existing results JSON.")
    parser.add_argument("--results-json", type=str, default=None, help="Path to existing *_results.json to update in place.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (fallback mode).")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (fallback mode).")
    
    args = parser.parse_args()

    if args.results_json:
        target_json = Path(args.results_json)
    else:
        if not args.exp_name or not args.data_root:
            raise ValueError("Use --results-json, or provide --exp_name and --data_root.")
        target_json = Path(args.data_root) / "json" / f"{args.exp_name}_results.json"

    if not target_json.exists():
        raise FileNotFoundError(f"Results JSON not found: {target_json}")

    updated = regenerate_vlm_planning_inplace(target_json)
    print(f"Updated vlm_planning in place: {target_json}")
    print(f"Num planned frames: {len(updated.get('vlm_planning', {}).get('Frames', {}))}")
