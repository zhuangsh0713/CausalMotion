import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
from causalmotion.utils.helpers import io_from_json
from causalmotion.detect_objs import detect_and_segment


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _frame_sort_key(frame_name):
    """
    Sort keyframe names robustly, supporting:
    - integer ids (0, 1, 2)
    - filename keys ("0.png", "1.png")
    """
    if isinstance(frame_name, int):
        return frame_name
    stem = Path(str(frame_name)).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def _load_gray_resized(image_path: Path, size=(96, 96)):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read keyframe image: {image_path}")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _mean_best_iou(prev_boxes_xywh: List[List[float]], curr_boxes_xywh: List[List[float]]):
    """
    Compute mean(best IoU) from current boxes to previous boxes.
    Returns value in [0, 1].
    """
    if not prev_boxes_xywh and not curr_boxes_xywh:
        return 1.0
    if not prev_boxes_xywh or not curr_boxes_xywh:
        return 0.0

    prev_xyxy = [xywh_to_xyxy(b) for b in prev_boxes_xywh]
    curr_xyxy = [xywh_to_xyxy(b) for b in curr_boxes_xywh]

    best_ious = []
    for c in curr_xyxy:
        best = 0.0
        for p in prev_xyxy:
            best = max(best, compute_iou(c, p))
        best_ious.append(best)

    return float(np.mean(best_ious)) if best_ious else 0.0


def prune_redundant_keyframes(
    keyframe_dir: Path,
    keyframe_bboxes: Dict,
    img_diff_thr: float = 0.02,
    bbox_change_thr: float = 0.08,
):
    """
    Remove middle keyframes that are visually and spatially redundant.

    A keyframe is dropped if BOTH are satisfied against the last kept frame:
    1) mean absolute image difference < img_diff_thr
    2) bbox change (1 - mean_best_iou) < bbox_change_thr

    Always keeps first and last keyframe.

    Args:
        keyframe_dir: Directory containing keyframe images.
        keyframe_bboxes: Dict[kf_id -> annotations], each annotation has "bbox" in xywh.
        img_diff_thr: Image-difference threshold in [0, 1].
        bbox_change_thr: Bounding-box change threshold in [0, 1].

    Returns:
        pruned_bboxes: same format as input, but with redundant middle frames removed.
        keep_ids: ordered keyframe ids that are kept.
    """
    kf_ids = sorted(keyframe_bboxes.keys(), key=_frame_sort_key)
    if len(kf_ids) <= 2:
        return keyframe_bboxes, kf_ids

    keep_ids = [kf_ids[0]]

    def _get_image_path(kf_id):
        name = str(kf_id)
        if name.endswith(".png"):
            return keyframe_dir / name
        return keyframe_dir / f"{name}.png"

    for i in range(1, len(kf_ids) - 1):
        prev_kept_id = keep_ids[-1]
        curr_id = kf_ids[i]

        prev_img = _load_gray_resized(_get_image_path(prev_kept_id))
        curr_img = _load_gray_resized(_get_image_path(curr_id))
        img_diff = float(np.mean(np.abs(curr_img.astype(np.float32) - prev_img.astype(np.float32))) / 255.0)

        prev_boxes = [
            ann.get("bbox", [])
            for ann in keyframe_bboxes.get(prev_kept_id, [])
            if isinstance(ann, dict) and len(ann.get("bbox", [])) == 4
        ]
        curr_boxes = [
            ann.get("bbox", [])
            for ann in keyframe_bboxes.get(curr_id, [])
            if isinstance(ann, dict) and len(ann.get("bbox", [])) == 4
        ]
        bbox_change = 1.0 - _mean_best_iou(prev_boxes, curr_boxes)

        # Keep only if there is meaningful change; otherwise prune.
        if not (img_diff < img_diff_thr and bbox_change < bbox_change_thr):
            keep_ids.append(curr_id)

    # Always keep the last keyframe
    keep_ids.append(kf_ids[-1])

    pruned_bboxes = {kf_id: keyframe_bboxes[kf_id] for kf_id in keep_ids}
    return pruned_bboxes, keep_ids


class TrajectoryBuilder:
    def __init__(self, frame_width=720, frame_height=480):
        self.W = frame_width
        self.H = frame_height

    def clip_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = max(0, min(x1, self.W))
        y1 = max(0, min(y1, self.H))
        x2 = max(0, min(x2, self.W))
        y2 = max(0, min(y2, self.H))

        if x1 >= x2 or y1 >= y2:
            return None
        return [x1, y1, x2, y2]

    def read_bbox_json(self, sim_data):
        frames = {}
        object_appearance = {}

        for frame_str, objs in sim_data["Frames"].items():
            t = int(frame_str)
            frames[t] = []

            for obj in objs:
                x, y, w, h = obj["box"]
                bbox = self.clip_bbox([x, y, x + w, y + h])
                if bbox is None:
                    continue

                obj_id = obj["id"]
                if obj_id not in object_appearance:
                    object_appearance[obj_id] = t
                    # represent the time of first appearance

                frames[t].append({
                    "id": obj_id,
                    "bbox": bbox
                })

        return frames, object_appearance

    def build_and_interpolate(self, frames, object_appearance, T=121):  # TODO: 121 for t2v, 151 for i2v
        all_frames = sorted(frames.keys())
        # Ordered by time of first appearance
        obj_ids = sorted(object_appearance.keys())

        bboxes = np.zeros((len(all_frames), len(obj_ids), 4))
        frame_map = {f: i for i, f in enumerate(all_frames)}

        for j, oid in enumerate(obj_ids):
            for t in all_frames:
                for obj in frames[t]:
                    if obj["id"] == oid:
                        bboxes[frame_map[t], j] = obj["bbox"]

        interp = np.zeros((T, len(obj_ids), 4))
        # Use VLM frame numbers proportionally instead of uniform linspace.
        # e.g. VLM frames [0, 3, 9, 10] → map 3→36, 9→108 for T=121
        max_vlm_frame = max(all_frames) if max(all_frames) > 0 else 1
        src_idx = np.array([f / max_vlm_frame * (T - 1) for f in all_frames], dtype=float)
        tgt_idx = np.arange(T)

        for j in range(len(obj_ids)):
            for k in range(4):
                valid = np.where(bboxes[:, j, k] != 0)[0]
                if len(valid) > 1:
                    interp[:, j, k] = np.interp(
                        tgt_idx,
                        src_idx[valid],
                        bboxes[valid, j, k]
                    )

        return interp, obj_ids

def align_keyframes_to_trajectory(
    data_dict,
    keyframe_bboxes: Dict[int, List[Dict]],
    trajectory: np.ndarray,
    force_first_zero: bool = True,
):
    """
    Returns:
        mapping: List[int]
            mapping[k] = trajectory index aligned to keyframe k
            mapping is guaranteed to be ascending and at least 10 frames apart
    """
    T, N, _ = trajectory.shape

    # ensure keyframes are in order: 0,1,2,...
    kf_ids = sorted(keyframe_bboxes.keys())
    mapping: List[int] = []

    prev_t = 0  # minimum allowed t (monotonic constraint)

    for idx, kf_id in enumerate(kf_ids):

        # Force first keyframe to t=0
        if idx == 0 and force_first_zero:
            mapping.append(0)
            prev_t = 0
            continue

        best_t = prev_t
        best_iou = -1.0

        # Ensure the next frame starts at least 10 frames after the previous one
        start_t = prev_t+10
        biggest_index = T - (data_dict["num_frames"] - idx - 1)*10

        for t in range(start_t, biggest_index):
            for traj_obj_idx in range(N):
                traj_bbox = trajectory[t, traj_obj_idx]
                if np.all(traj_bbox == 0):
                    continue

                for kf_obj in keyframe_bboxes[kf_id]:
                    kf_bbox = xywh_to_xyxy(kf_obj["bbox"])
                    iou = compute_iou(kf_bbox, traj_bbox)

                    if iou > best_iou:
                        best_iou = iou
                        best_t = t
                        print("best_iou:", best_iou)

        # Fallback when IoU=0 (object positions completely disjoint):
        # use a linear interpolation estimate based on keyframe index.
        if best_iou <= 0:
            frac = idx / max(len(kf_ids) - 1, 1)
            fallback_t = int(prev_t + (biggest_index - prev_t) * frac)
            best_t = max(prev_t + 10, min(fallback_t, biggest_index - 1))
            print(f"[align] IoU=0 for keyframe {idx}, fallback to t={best_t}")

        mapping.append(best_t)
        prev_t = best_t  # enforce monotonicity and minimum gap

    return mapping




# Optional: Render video for visualization
def render_trajectory_video(
    trajectory,
    keyframe_alignment,
    keyframe_dir,
    output_path,
    W=720,
    H=480,
    fps=30
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    empty_bg = np.zeros((H, W, 3), dtype=np.uint8)

    keyframe_images = {}
    for kf_id, t in enumerate(keyframe_alignment):
        img_path = keyframe_dir / f"{kf_id}.png"
        img = cv2.imread(str(img_path))
        if img is not None:
            keyframe_images[t] = cv2.resize(img, (W, H))

    for t in range(trajectory.shape[0]):
        frame = empty_bg.copy()

        if t in keyframe_images:
            frame = keyframe_images[t].copy()

        for obj_idx in range(trajectory.shape[1]):
            bbox = trajectory[t, obj_idx]
            if np.all(bbox == 0):
                continue
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"[OK] Video saved to {output_path}")
    return output_path



if __name__ == '__main__':
    data_root = Path('./data')
    exp_name = 'example_experiment'
    sim_json_path = data_root / 'json' / f'{exp_name}.json'
    keyframe_dir = data_root / 'keyframes' / exp_name
    output_video_path = data_root / 'alignment_videos' / f'{exp_name}_alignment.mp4'

    sim_data = io_from_json(sim_json_path, io_type='r')

    builder = TrajectoryBuilder()
    frames, object_appearance = builder.read_bbox_json(sim_data)
    trajectory, obj_ids = builder.build_and_interpolate(frames, object_appearance)

    # Assume keyframe IDs are 0, 1, 2, ...
    keyframe_bboxes = {
        kf_id: frames[kf_id][0]['bbox'] for kf_id in frames if len(frames[kf_id]) > 0
    }

    alignment = align_keyframes_to_trajectory(keyframe_bboxes, trajectory, obj_idx=0)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    render_trajectory_video(
        trajectory,
        alignment,
        keyframe_dir,
        output_video_path
    )
