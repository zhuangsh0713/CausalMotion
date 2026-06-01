import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import random
from pathlib import Path
import json
import base64
import imageio
import re
import ast
import os
import rp.git.CommonSource.noise_warp as nw



def io_from_json(json_path, data=None, io_type='r'):
    """
    Reads from or writes to a JSON file depending on the specified mode.

    Parameters:
        json_path (str or Path): The path to the JSON file.
        data (dict or list, optional): The data to write to the file. Required if io_type is 'w'.
        io_type (str): Operation type. Use 'r' to read and 'w' to write. Default is 'r'.

    Returns:
        dict or list (if io_type == 'r'): The parsed JSON data when reading.
        None (if io_type == 'w'): Data is written to file and nothing is returned.

    Raises:
        FileNotFoundError: If the file doesn't exist when reading.
        ValueError: If an invalid io_type is provided.
    """
    if io_type == 'r':
        # Read JSON from file
        with open(json_path, 'r', encoding='utf-8') as f:
            print(f'Reading from {json_path}')
            return json.load(f)

    elif io_type == 'w':
        # Write JSON to file
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'Writing into {json_path}')

    else:
        raise ValueError(f"Invalid io_type '{io_type}'. Use 'r' for read or 'w' for write.")


CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


PURE_COLOR_MAP = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Dark Red
    (0, 128, 0),     # Dark Green
    (0, 0, 128),     # Dark Blue
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
]



def encode_image(image_input):
    """
    Encodes an image into a base64 string.

    Supports both image file paths and raw image bytes. This is commonly used
    for sending image data to APIs that accept base64-encoded input.

    Parameters:
        image_input (str, Path, or bytes): The input image. It can be:
            - A file path (string or Path object) to an image file.
            - Raw image bytes.

    Returns:
        str: The base64-encoded string representation of the image.

    Raises:
        FileNotFoundError: If the input is a file path and the file does not exist.
    """
    if isinstance(image_input, (str, Path)):
        # Convert to Path object and verify the file exists
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        # Read the image file as bytes
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    elif isinstance(image_input, bytes):
        # If raw bytes are provided directly
        image_bytes = image_input

    # Encode bytes to base64 and convert to UTF-8 string
    return base64.b64encode(image_bytes).decode('utf-8')


def get_vlm_plan_to_json(text):
    """
    Parses a VLM plan text into a structured dictionary mapping frame numbers to object lists.

    The input text is expected to follow the format:
        Frame 0: [{'id': 0, 'name': 'ball', 'box': [x, y, w, h]}, ...]
        Frame 1: [{'id': 0, 'name': 'ball', 'box': [x, y, w, h]}, ...]
        ...

    Parameters:
        text (str): The raw output string from a VLM (e.g., GPT-4o) containing frame-by-frame object info.

    Returns:
        dict: A dictionary where keys are frame numbers (int), and values are lists of objects with keys:
              'id', 'name', and 'box' (bounding box in [x, y, w, h] format).
    """
    frames = {}

    # Define regex pattern to capture lines like: Frame 1: [ {...}, {...} ]
    pattern = r'^Frame\s+(\d+):\s*(.*)$'

    # Split the text into lines
    lines = text.strip().splitlines()

    # Iterate through each line
    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)

        # If line matches the expected pattern
        if match:
            frame_number = int(match.group(1))        # Extract frame index
            objects_str = match.group(2)              # Extract the string representing object list
            try:
                # Safely evaluate the string into a Python list/dict
                objects = ast.literal_eval(objects_str)
            except Exception as e:
                print(f"Error parsing Frame {frame_number}: {e}")
                objects = None

            # Save parsed data into the result dictionary
            frames[frame_number] = objects

    return frames



def apply_alternating_noise(latent, noise1=0.4, noise2=0.6):
    for i in range(latent.shape[1]):
        noise = noise1 if i % 2 == 0 else noise2
        latent[:, i] = nw.mix_new_noise(latent[:, i], noise)
    return latent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def build_first_frame_prompt(scene_text, key_object):
    return f"""
You are generating the FIRST KEYFRAME (frame t=0) of a SINGLE CONTINUOUS VIDEO.

This frame DEFINES the CANONICAL CAMERA AND SCENE STATE.
All future frames MUST EXACTLY MATCH this camera and scene.

IMAGE FORMAT (STRICT):
- Resolution: 720x480 pixels
- Aspect ratio: 3:2
- No cropping, no padding, no zoom

CANONICAL CAMERA (NON-NEGOTIABLE):
- This frame establishes the ONLY camera for the entire video.
- Camera position, orientation, height, distance, focal length,
  field of view, and projection geometry are FIXED FOREVER.
- Perspective MUST remain identical in all later frames.
- NO camera movement, NO reframing, NO zoom, NO tilt, NO pan.

SCENE CANONICALIZATION (LOCKED):
- Background, environment, lighting, shadows, textures, and color tone
  MUST be fully determined in this frame and remain unchanged.
- All NON-KEY objects are STATIC and IMMUTABLE across all frames.
- No new objects may appear in later frames.

KEY OBJECT (ONLY dynamic entity in later frames):
{key_object}

KEY OBJECT RULES FOR FRAME t=0:
- The key object MUST be clearly visible and fully inside the frame.
- Its initial position, size, and orientation DEFINE its reference state.
- Object scale MUST remain physically consistent across frames
  (no resizing unless physically caused).

INITIAL FRAME CONTENT:
{scene_text}

ABSOLUTE CONSTRAINT:
This frame is the canonical reference (frame t=0).
Any deviation in camera, layout, or scene structure in later frames
will be considered INVALID.
"""


def build_next_frame_prompt(txt, key_object):
    return f"""
You are generating the NEXT KEYFRAME (frame t+1)
of the SAME SINGLE CONTINUOUS VIDEO.

This frame is a DIRECT TEMPORAL CONTINUATION
of the provided previous frame.

IMAGE FORMAT (STRICT):
- Resolution: 720x480 pixels
- Aspect ratio: 3:2
- No cropping, no padding, no resizing

ABSOLUTE CAMERA LOCK (NON-NEGOTIABLE):
- Camera viewpoint, position, orientation, height, distance,
  focal length, field of view, and projection geometry
  MUST be EXACTLY IDENTICAL to the given frame.
- Treat the camera as a fixed, immovable surveillance camera.
- NO pan, NO tilt, NO zoom, NO reframing, NO perspective change.

KEY OBJECT (DYNAMIC ENTITY):
{key_object}

KEY OBJECT MOTION RULES:
- The key object is MOVING. You MUST render it at its NEW coordinates.
- **CRITICAL:** You must REMOVE/ERASE the key object from its previous position in the last frame.
- Fill the "hole" left by the moving object with the background texture (In-painting).

SCENE IMMUTABILITY:
- The background remains locked, EXCEPT for the area where the {key_object} was previously located.
- That specific area must now show the background that was previously hidden (occluded) by the object.

NEXT FRAME CHANGE DESCRIPTION:
{txt}

ABSOLUTE CONSTRAINT:
This frame MUST look like frame t+1 rendered
from the SAME camera, not a re-imagined scene.

Any change to camera, background, lighting,
or non-key objects INVALIDATES the result.
"""
