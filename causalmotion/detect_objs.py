'''Grounded-SAM-2.

The source code is adopted from:
https://github.com/IDEA-Research/Grounded-SAM-2/tree/main

'''



import argparse
import os
import re
import cv2
import json
import torch
import random
import numpy as np
import base64
from io import BytesIO
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from causalmotion.utils.helpers import CUSTOM_COLOR_MAP, PURE_COLOR_MAP, io_from_json
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from collections import defaultdict


def flatten_list(nested_list):
    """
    Flattens any arbitrarily nested list into a one-dimensional list.

    Parameters:
        nested_list (list): A list that may contain other nested lists at any depth.

    Returns:
        list: A flat list containing all the non-list elements from the original nested structure, 
              in the order they appeared.
    """
    flat = []  # Initialize an empty list to store the flattened elements
    for item in nested_list:
        if isinstance(item, list):  # If the current item is a list, recursively flatten it
            flat.extend(flatten_list(item))
        else:
            flat.append(item)  # If the item is not a list, append it directly to the result
    return flat  # Return the fully flattened list


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        box1 (tuple or list): [x1, y1, w1, h1], where (x1, y1) is the top-left corner,
                              w1 is the width, and h1 is the height of the first box.
        box2 (tuple or list): [x2, y2, w2, h2], same format for the second box.

    Returns:
        float: IoU value between 0 and 1. Returns 0 if there is no overlap.
    """
    # Convert boxes to corner coordinates (x1, y1, x2, y2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    # Compute intersection coordinates
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    # Compute intersection area (0 if no overlap)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute each box's area
    area1 = w1 * h1
    area2 = w2 * h2

    # Compute union area
    union_area = area1 + area2 - inter_area

    # Return IoU (avoid division by zero)
    return inter_area / union_area if union_area > 0 else 0


def remove_overlapping_bboxes_with_ids_and_extras(
    bbox_groups, id_groups, extra_groups1, extra_groups2, iou_threshold=0.05
):
    """
    Removes overlapping bounding boxes across all groups based on an IoU threshold,
    while keeping associated IDs and extra metadata in sync.

    Parameters:
        bbox_groups (list of list): A list of groups, each containing bounding boxes 
                                    in [x, y, w, h] format.
        id_groups (list of list): A list of groups, each containing IDs corresponding to the boxes.
        extra_groups1 (list of list): A list of groups, each containing auxiliary data 
                                      aligned with each bounding box (e.g., labels).
        extra_groups2 (list of list): A second list of auxiliary data for each bounding box.
        iou_threshold (float): Two boxes are considered overlapping if their IoU is greater than this value.

    Returns:
        tuple of lists: Flattened lists of non-overlapping bounding boxes, corresponding IDs, 
                        extra_groups1 values, and extra_groups2 values.
    """
    kept_bboxes = []

    # Initialize output containers for cleaned results
    cleaned_bbox_groups = []
    cleaned_id_groups = []
    cleaned_extra_groups1 = []
    cleaned_extra_groups2 = []

    # Iterate through each group of data
    for bboxes, ids, extras1, extras2 in zip(bbox_groups, id_groups, extra_groups1, extra_groups2):
        new_bboxes = []
        new_ids = []
        new_extras1 = []
        new_extras2 = []

        # Iterate through each item in the group
        for bbox, id_, e1, e2 in zip(bboxes, ids, extras1, extras2):
            # Keep the box only if it does not significantly overlap with any previously kept box
            if all(compute_iou(bbox, prev_bbox) <= iou_threshold for prev_bbox in kept_bboxes):
                new_bboxes.append(bbox)
                new_ids.append(id_)
                new_extras1.append(e1)
                new_extras2.append(e2)
                kept_bboxes.append(bbox)

        # Append filtered group results
        cleaned_bbox_groups.append(new_bboxes)
        cleaned_id_groups.append(new_ids)
        cleaned_extra_groups1.append(new_extras1)
        cleaned_extra_groups2.append(new_extras2)

    # Return all results as flattened lists
    return (
        flatten_list(cleaned_bbox_groups),
        flatten_list(cleaned_id_groups),
        flatten_list(cleaned_extra_groups1),
        flatten_list(cleaned_extra_groups2),
    )

def single_mask_to_rle(mask):
    """
    Converts a mask to RLE format.
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]

    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def merge_masks_with_colors(masks, background_color=(255, 255, 255)):
    """
    Merge multiple masks into a single color image, assigning a unique color
    to each object mask. Background is set to a specified color (default: white).

    Parameters:
        masks (list of np.ndarray): A list of 2D binary masks (arrays of shape H x W),
                                    where each mask corresponds to one object.
        background_color (tuple): RGB color for the background. Default is white (255, 255, 255).

    Returns:
        PIL.Image.Image: A merged color image where each object is shown in a different color.
    """
    # Define the output image size (width x height)
    image_size = (720, 480)

    # Create a blank RGB image filled with the background color
    merged_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    merged_image[:] = background_color

    # Shuffle the color map so each run gives a different set of colors
    random.shuffle(PURE_COLOR_MAP)

    # Iterate over each mask and paint it onto the merged image
    for i, mask in enumerate(masks):
        # Select a color from the color map, cycling through if there are more masks than colors
        color = PURE_COLOR_MAP[i % len(PURE_COLOR_MAP)]

        # Apply the color to the regions where the mask is non-zero
        merged_image[mask > 0] = color

    # Convert the NumPy array to a PIL Image
    merged_image_pil = Image.fromarray(merged_image)

    return merged_image_pil



def add_class_numbers(class_names):
    """
    Add numbering to repeated class names in a list to distinguish duplicates.

    For example:
        ["ball", "ball", "water", "ball"] 
        → ["ball 1", "ball 2", "water", "ball 3"]

    Parameters:
        class_names (list of str): A list of class names, possibly with duplicates.

    Returns:
        list of str: A new list where repeated class names are suffixed with an
                     incrementing number, while unique names are left unchanged.
    """
    # Count total occurrences of each class name
    total_count = defaultdict(int)
    for name in class_names:
        total_count[name] += 1

    # Track the current occurrence index while iterating
    current_count = defaultdict(int)
    result = []

    # Iterate through the original list and append numbering as needed
    for name in class_names:
        if total_count[name] > 1:
            current_count[name] += 1
            result.append(f"{name} {current_count[name]}")
        else:
            result.append(name)

    return result

def xyxy_to_xywh(box):
    """
    Convert a bounding box from [x_min, y_min, x_max, y_max] format to
    [x_min, y_min, width, height] format.

    Parameters:
        box (list or tuple): A list or tuple of four values representing the bounding box 
                             in (x_min, y_min, x_max, y_max) format.

    Returns:
        list: A list of four integers in (x_min, y_min, width, height) format.
    """
    x_min, y_min, x_max, y_max = box

    # Ensure coordinates are integers
    x_min = int(x_min)
    y_min = int(y_min)

    # Compute width and height from corner coordinates
    width = int(x_max - x_min)
    height = int(y_max - y_min)

    return [x_min, y_min, width, height]




def detect_and_segment(
    info_json,
    sam2_predictor=None,
    processor=None,
    grounding_model=None,
    DEVICE="cuda",
    box_threshold=0.3,
    text_threshold=0.15,
    min_box_area=16 * 16,
    max_box_area_ratio=0.9,
    output_dir=None,
    file_prefix=None,
):
    """
    Detects and segments key physical objects from the first frame of a video using a 
    grounding model and SAM2 (Segment Anything Model), then saves annotated outputs 
    and updates the input JSON with detection results.

    Parameters:
        info_json (dict): Dictionary containing detection input, including:
                          - 'first_frame': path to the first frame image
                          - 'key_physic_object': string prompt describing objects to detect
        output_dir (Path or str, optional): Directory to save annotation/mask images.
                                             If None, defaults to "<first_frame_parent>/_detections".
        file_prefix (str, optional): Prefix of output file names. If None, uses first frame stem.
        sam2_predictor: SAM2 predictor object used for mask generation.
        processor: Processor for preparing inputs and post-processing outputs of the grounding model.
        grounding_model: The model used to perform text-guided object detection.
        DEVICE (str): The device to run models on, e.g., "cuda" or "cpu".

    Returns:
        dict: Updated info_json dictionary containing segmentation paths and annotations.
    """

    # Load image path and detection prompts
    first_frame_path = str(info_json["first_frame"])
    detect_prompts = info_json["key_physic_object"]
    first_frame_stem = Path(first_frame_path).stem

    # Resolve output location for visualization artifacts.
    if output_dir is None:
        output_dir = Path(first_frame_path).parent / "_detections"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if file_prefix is None:
        file_prefix = first_frame_stem

    # Split multiple object prompts by commas
    DETECT_PROMPTS = [item.strip() for item in detect_prompts.split(",") if item.strip()]
    DUMP_JSON_RESULTS = True  # Whether to store results in the output JSON

    # Enable mixed precision inference and TF32 if supported
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load the first frame and prepare SAM2 predictor
    image = Image.open(first_frame_path)
    print('First frame image size:', image.size)
    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # Initialize containers for all detections
    all_masks = []
    all_boxes = []
    all_class_names = []
    all_confidences = []

    print('DETECT_PROMPTS', DETECT_PROMPTS)

    image_area = image.size[0] * image.size[1]

    # Loop through each detection prompt
    for text in DETECT_PROMPTS:
        print("Detecting objects for prompt:", text)
        query = text.strip()
        if query and not query.endswith("."):
            # GroundingDINO often works more stably with noun phrases ending in '.'
            query = f"{query}."

        # Prepare input for the grounding model
        inputs = processor(images=image, text=query, return_tensors="pt").to(DEVICE)

        # Run detection model
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        print("Grounding model outputs obtained.")
        print("outputs logits shape:", outputs.logits.shape)

        # Post-process detections
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        print("results:", results)

        # Get detected boxes
        input_boxes = results[0]["boxes"].cpu().numpy()
        if input_boxes.size == 0:
            print(f"[Warn] No boxes for prompt '{text}' with box_threshold={box_threshold}")
            continue

        # Filter degenerate / huge boxes before SAM2 to reduce noisy labels and align failures.
        wh = np.maximum(0.0, input_boxes[:, 2:4] - input_boxes[:, 0:2])
        areas = wh[:, 0] * wh[:, 1]
        valid = (areas >= float(min_box_area)) & (areas <= float(max_box_area_ratio) * float(image_area))
        if not np.any(valid):
            print(f"[Warn] All boxes filtered for prompt '{text}' by area constraints.")
            continue

        input_boxes = input_boxes[valid]
        filtered_scores = results[0]["scores"].cpu().numpy()[valid]
        raw_labels = [results[0]["labels"][i] for i in np.where(valid)[0]]

        cleaned_labels = []
        for raw_label in raw_labels:
            label = str(raw_label).replace("##", "").strip(" ,.;:")
            if not label:
                label = text.strip() if text.strip() else "object"
            cleaned_labels.append(label)

        # Predict masks using SAM2 for the detected boxes
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Remove unnecessary dimensions if present
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Collect predictions
        all_masks.append(masks)
        all_boxes.append(input_boxes)
        all_class_names.append(cleaned_labels)
        all_confidences.append(filtered_scores.tolist())

    # Remove overlapping boxes and align all associated metadata
    if all_boxes:
        all_boxes, all_class_names, all_masks, all_confidences = remove_overlapping_bboxes_with_ids_and_extras(
            all_boxes, all_class_names, all_masks, all_confidences, iou_threshold=0.1
        )

    # Convert to arrays
    masks = (
        np.stack(all_masks)
        if all_masks
        else np.zeros((0, image.size[1], image.size[0]), dtype=bool)
    )
    input_boxes = (
        np.array(all_boxes, dtype=np.float32).reshape(-1, 4)
        if all_boxes
        else np.zeros((0, 4), dtype=np.float32)
    )

    # Add numbering to duplicate class names (e.g., "ball 1", "ball 2")
    class_names = add_class_numbers(all_class_names) if all_class_names else []
    class_ids = np.array(list(range(len(class_names))))

    # Annotate and save visualizations
    img = cv2.imread(first_frame_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    # Draw bounding boxes
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    # bbox_image_path = output_root / "roundingdino_annotated_image_with_bbox.jpg"
    # cv2.imwrite(bbox_image_path, annotated_frame)

    # Draw masks on top of bounding boxes
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    mask_image_path = output_dir / f"{file_prefix}_annotated_image_with_mask.jpg"
    cv2.imwrite(mask_image_path, annotated_frame)

    # Merge all masks into a single image using distinct colors
    seg_mask = merge_masks_with_colors(masks)
    seg_mask_path = output_dir / f"{file_prefix}_merge_mask.jpg"
    seg_mask.save(seg_mask_path)

    # Optionally dump results into the JSON
    if DUMP_JSON_RESULTS:
        input_boxes = input_boxes.tolist()
        rounded_boxes = [[round(coord, 0) for coord in xyxy_to_xywh(box)] for box in input_boxes]

        info_json['seg_mask_path'] = str(seg_mask_path)
        info_json['annotations'] = [
            {
                "class_name": class_name,
                "bbox": box,
            }
            for class_name, box in zip(class_names, rounded_boxes)
        ]
        info_json['box_format'] = 'xywh'

    return info_json


def refresh_results_json_detection_fields(
    results_json_path,
    sam2_predictor,
    processor,
    grounding_model,
    DEVICE="cuda",
    box_threshold=0.3,
    text_threshold=0.15,
    min_box_area=16 * 16,
    max_box_area_ratio=0.9,
    output_dir=None,
    file_prefix=None,
):
    """
    Read an existing *_results.json and update detection-related fields in place:
      - seg_mask_path
      - annotations
      - box_format
    """
    results_json_path = Path(results_json_path)
    info_json = io_from_json(results_json_path, io_type='r')

    refreshed = detect_and_segment(
        info_json=info_json,
        sam2_predictor=sam2_predictor,
        processor=processor,
        grounding_model=grounding_model,
        DEVICE=DEVICE,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        min_box_area=min_box_area,
        max_box_area_ratio=max_box_area_ratio,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    for k in ("seg_mask_path", "annotations", "box_format"):
        info_json[k] = refreshed[k]

    io_from_json(results_json_path, data=info_json, io_type='w')
    return info_json


# For each keyframes from the first CoT step
# Extract bbox for each generated keyframes
# 1. Build info json for keyframe
def build_info_json_for_keyframe(image_path, object_prompt):
    return {
        "first_frame": image_path,
        "key_physic_object": object_prompt
    }
# 2. 
# TODO: 
def extract_bboxes_from_keyframe_folder(
    keyframe_dir,
    object_prompt,
    sam2_predictor,
    processor,
    grounding_model,
    DEVICE="cuda"
):
    """
    return dict: frame_name -> annotations
    """

    keyframe_dir = Path(keyframe_dir)
    results = {}

    image_paths = sorted([
        p for p in keyframe_dir.iterdir()
        if p.suffix.lower() == ".png"
    ])

    for img_path in image_paths:
        info_json = build_info_json_for_keyframe(
            image_path=str(img_path),
            object_prompt=object_prompt
        )

        print('Processing keyframe:', img_path)
        info_json = detect_and_segment(
            info_json=info_json,
            sam2_predictor=sam2_predictor,
            processor=processor,
            grounding_model=grounding_model,
            DEVICE=DEVICE
        )

        results[img_path.name] = info_json["annotations"]

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recaption and label video prompts.")
    parser.add_argument("--results-json", type=str, default=None, help="Path to existing *_results.json for in-place detection field refresh.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (legacy mode).")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (legacy mode).")
    parser.add_argument("--first_frame_path", type=str, default=None, help="First frame path (legacy mode).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save detection artifacts.")
    parser.add_argument("--file-prefix", type=str, default=None, help="Prefix for detection artifact file names.")
    parser.add_argument("--grounding_model_path", type=str, default="IDEA-Research/grounding-dino-base", help="GroundingDINO model path.")
    parser.add_argument("--sam2_checkpoint", type=str, default="./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint path.")
    parser.add_argument("--sam2_model_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config path.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--box-threshold", type=float, default=0.3, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.15, help="GroundingDINO text threshold.")
    parser.add_argument("--min-box-area", type=float, default=5 * 5, help="Minimum valid box area in pixels.")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.9, help="Maximum valid box area ratio against image area.")
    
    args = parser.parse_args()

    DEVICE = args.device
    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    processor = AutoProcessor.from_pretrained(args.grounding_model_path)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model_path).to(DEVICE)

    if args.results_json:
        results_json_path = Path(args.results_json)
        stem = results_json_path.stem
        default_prefix = stem[:-8] if stem.endswith("_results") else stem
        out_dir = args.output_dir
        if out_dir is None:
            out_dir = results_json_path.parent / "detections" / default_prefix

        result = refresh_results_json_detection_fields(
            results_json_path=results_json_path,
            sam2_predictor=sam2_predictor,
            processor=processor,
            grounding_model=grounding_model,
            DEVICE=DEVICE,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_box_area=args.min_box_area,
            max_box_area_ratio=args.max_box_area_ratio,
            output_dir=out_dir,
            file_prefix=args.file_prefix or default_prefix,
        )
        print("Updated detection fields in place:", results_json_path)
        print("seg_mask_path:", result.get("seg_mask_path"))
    else:
        if not args.exp_name or not args.data_root or not args.first_frame_path:
            raise ValueError(
                "Legacy mode requires --exp_name --data_root --first_frame_path, "
                "or use --results-json for in-place update."
            )

        exp_name = args.exp_name
        data_root = Path(args.data_root)
        detection_root = data_root / 'detections'
        first_frame_path = Path(args.first_frame_path)
        input_json_path = data_root / 'json' / f'{exp_name}.json'
        output_json_path = data_root / 'json' / f'{exp_name}_2.json'

        info_json = io_from_json(input_json_path, io_type='r')
        info_json['first_frame'] = str(first_frame_path)
        output_root = args.output_dir or (detection_root / f'{exp_name}')

        result = detect_and_segment(
            info_json=info_json,
            sam2_predictor=sam2_predictor,
            processor=processor,
            grounding_model=grounding_model,
            DEVICE=DEVICE,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_box_area=args.min_box_area,
            max_box_area_ratio=args.max_box_area_ratio,
            output_dir=output_root,
            file_prefix=args.file_prefix or exp_name,
        )
        print('result', result)
        io_from_json(output_json_path, result, io_type='w')
