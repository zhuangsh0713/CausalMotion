from openai import OpenAI
import base64
import os
import re
import json
import httpx
import copy
import time
from PIL import Image
from io import BytesIO
from causalmotion.utils.template import template_first_frame_message, template_next_frame_message, template_consequence_message
from causalmotion.utils.helpers import io_from_json, encode_image, build_first_frame_prompt, build_next_frame_prompt

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")
vlm_model = os.getenv("CAUSALMOTION_VLM_MODEL", "qwen2.5-vl-72b-instruct")
image_model = os.getenv("CAUSALMOTION_IMAGE_MODEL", "gpt-image-1.5")
timeout = httpx.Timeout(
    connect=60.0,
    read=None,  # 流式请求可能超过 10 分钟，禁用 read timeout
    write=60.0,
    pool=60.0,
)

client = OpenAI(
    api_key = openai_api_key,
    base_url=openai_base_url,
    http_client=httpx.Client(timeout=timeout),
)

def chat_completion_with_retry(*args, retries=3, delay=5, **kwargs):
    for i in range(retries):
        try:
            return client.chat.completions.create(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise


def _image_request_with_retry(request_fn, retries=1, delay=30):
    for i in range(retries):
        try:
            return request_fn()
        except Exception as e:
            print(f"Image attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise


def _decode_image_from_base64(image_base64):
    if not image_base64:
        raise RuntimeError("Empty image payload from image API.")
    return base64.b64decode(image_base64)


def _collect_stream_image_bytes(stream, completed_event_type):
    final_image_base64 = None
    last_partial_base64 = None
    partial_count = 0

    for event in stream:
        event_type = getattr(event, "type", "")

        if event_type.endswith(".partial_image"):
            partial_count += 1
            last_partial_base64 = getattr(event, "b64_json", None)
            print(f"[image stream] partial image #{partial_count}")
        elif event_type == completed_event_type:
            final_image_base64 = getattr(event, "b64_json", None)

    if final_image_base64:
        return _decode_image_from_base64(final_image_base64)

    if last_partial_base64:
        print("[image stream] completed event missing, fallback to last partial image.")
        return _decode_image_from_base64(last_partial_base64)

    raise RuntimeError("Image stream ended without image payload.")


def generate_or_edit_image_bytes(prompt, image=None, partial_images=0):
    """
    Generate or edit image with streaming first; fallback to non-streaming if provider
    does not support stream mode.
    """

    # def _stream_call():
    #     common_kwargs = {
    #         "model": "gpt-image-1.5",
    #         "prompt": prompt,
    #         "stream": True,
    #         "partial_images": partial_images,
    #     }
    #     if image is None:
    #         stream = client.images.generate(**common_kwargs)
    #         return _collect_stream_image_bytes(stream, completed_event_type="image_generation.completed")

    #     stream = client.images.edit(image=image, **common_kwargs)
    #     return _collect_stream_image_bytes(stream, completed_event_type="image_edit.completed")

    # try:
    #     return _image_request_with_retry(_stream_call)
    # except Exception as stream_err:
    #     print(f"[image stream] fallback to non-stream mode: {stream_err}")

    def _non_stream_call():
        common_kwargs = {
            "model": image_model,
            "prompt": prompt,
        }
        if image is None:
            response = client.images.generate(**common_kwargs)
        else:
            response = client.images.edit(image=image, **common_kwargs)
        return _decode_image_from_base64(response.data[0].b64_json)

    return _image_request_with_retry(_non_stream_call)

# 1. Without the First Frame
def generate_consequence_first_frame_txt(prompt):
    messages = copy.deepcopy(template_first_frame_message)
    messages[-1]['content'][0]['text'] = """Input Prompt:{}""".format(prompt)
    
    # List all models
    # models = client.models.list()
    # for m in models.data:
    #     print(m.id)
    t0 = time.time()
    response = chat_completion_with_retry(
        model=vlm_model,
        messages = messages
    )
    print(f"[GPT-4o] latency = {time.time() - t0:.2f}s")

    # break the whole process at this place
    # raise RuntimeError("An error occurred during the process.")
    
    result = response.choices[0].message.content
    key_object = re.search(r'Key Object:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    category = re.search(r'Category:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    consequence = re.search(r'Consequences:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    context_frame = re.search(r'Context Frame:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    concise_prompt = re.search(r'Concise Prompt:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    
    print("First Frame Generation Result:\n", result)
    
    return {
        "key_physic_object": key_object.group(1).strip() if key_object else "",
        "Category": category.group(1).strip() if category else "",
        "Context Frame": context_frame.group(1).strip() if context_frame else "",
        "Concise Prompt": concise_prompt.group(1).strip() if concise_prompt else "",
        "Consequences": consequence.group(1).strip() if consequence else ""
    }

# 2. Given the First Frame
def generate_only_consequence_text(data_root, prompt, first_frame_path, exp_name):
    # 1. Resave first frame image into data_root / keyframes / exp_name / 0.png
    if first_frame_path is not None:
        with open(first_frame_path, "rb") as f:
            img_bytes = f.read()

    # 2. Resize + format
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((720, 480))

    buf = BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # 3. Save to canonical path
    output_path = os.path.join(
        data_root, "keyframes", exp_name, "0.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(img_bytes)
        
    messages = copy.deepcopy(template_consequence_message)
    messages[-1]['content'][0]['text'] = """Input Prompt:{}""".format(prompt)
    # template_consequence_message[-1]['content'][0]['image'] = base64.b64encode(img_bytes).decode()  
    messages[-1]['content'][1]["image_url"]["url"] = image_bytes_to_data_url(img_bytes)
    
    response = chat_completion_with_retry(
        model=vlm_model,
        messages=messages,
        max_tokens=300
    )
    
    result = response.choices[0].message.content
    # print("Debug: Full response content:\n", result)
    key_object = re.search(r'Key Object:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    category = re.search(r'Category:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    consequence = re.search(r'Consequences:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    context_frame = re.search(r'Context Frame:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    
    print("Consequence Text Generation Result:\n", result)
    
    # Parse results (similar to the first function)
    # return img_bytes, {
    #     "key_physic_object": re.search(r'Key Object:\s*(.*?)(?:\n|$)', result).group(1) if "Key Object" in result else "",
    #     "Context Frame": re.search(r'Context Frame:\s*(.*?)(?:\n|$)', result).group(1) if "Context Frame" in result else "",
    #     "Category": re.search(r'Category:\s*(.*?)(?:\n|$)', result).group(1) if "Category" in result else "",
    #     "Consequences": re.search(r'Consequences:\s*(.*?)(?:\n|$)', result).group(1) if "Consequences" in result else ""
    # }
    return img_bytes, {
        "key_physic_object": key_object.group(1).strip() if key_object else "",
        "Category": category.group(1).strip() if category else "",
        "Context Frame": context_frame.group(1).strip() if context_frame else "",
        "Consequences": consequence.group(1).strip() if consequence else ""
    }


def image_bytes_to_data_url(img_bytes, format="PNG"):
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/{format.lower()};base64,{base64_str}"


def generate_next_frame_txt(prompt, previous_frames, last_txt, consequence):
    # Input Prompt, Key Frames， Hint
    # template_next_frame_message[-1]['content'][0]['text'] = (
    #     f"Input Prompt: {prompt}\nHint: {consequence}\n"
    # )
    messages = copy.deepcopy(template_next_frame_message)
    messages[-1]['content'][0]['text'] = (
        f"Input Prompt: {prompt}\nHint: {consequence}\nPrevious Frame Description: {last_txt}\n"
    )
    print("Generating next frame with prompt:", messages[-1]['content'][0]['text'])
    # raise RuntimeError("An error occurred during the process.")


    # for keyframe in previous_frames:
    #     messages[-1]['content'].append(
    #         {
    #             "type": "image_url",
    #             "image_url": {
    #                 "url" : image_bytes_to_data_url(keyframe)
    #             }
    #         }
    #     )
    last_frame = previous_frames[-1]
    messages[-1]['content'].append({
        "type": "image_url",
        "image_url": {
            "url": image_bytes_to_data_url(last_frame)
        }
    })
    
    
    response =chat_completion_with_retry(
        model=vlm_model,
        messages = messages,
    )
    
    result = response.choices[0].message.content
    next_predicted_frame = re.search(r'Next Predicted Keyframe:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    last_frame = re.search(r'Last Frame:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    
    print("Frame Index is:", len(previous_frames))
    print("Next Frame Prediction Result:\n", result)    
    return {
        "Next Predicted Keyframe": next_predicted_frame.group(1).strip() if next_predicted_frame else "",
        "Last Frame": last_frame.group(1).strip() if last_frame else ""
    }


def generate_first_frame(text, key_object, data_root, exp_name):
    prompt = build_first_frame_prompt(text, key_object)

    img_bytes = generate_or_edit_image_bytes(prompt=prompt, partial_images=0)
    # response = client.chat.completions.create(
    #     model="gpt-image-1",
    #     # model = "black-forest-labs/flux.2-max",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     extra_body = {"modalities": ["image", "text"]},
    #     # image_config = {
    #     #     "aspect_ratio": "3:2",
    #     # }
    # )

    # Resize the image to 720x480
    img = Image.open(BytesIO(img_bytes))
    img = img.resize((720, 480))
    resized_img_bytes = BytesIO()
    img.save(resized_img_bytes, format="PNG")
    img_bytes = resized_img_bytes.getvalue()

    output_path = os.path.join(
        data_root, "keyframes", exp_name, "0.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_bytes)

    return img_bytes



# def edit_following_frame(chainvis, txt, data_root, exp_name):
#     # client = OpenAI(
#     #     api_key = openai_api_key,
#     #     base_url = "https://api.openai.com/v1/images/edits"
#     # )
    
#     response = client.images.edit(
#         model = "openai/gpt-image-1", 
#         image = chainvis,
#         prompt = txt,
#     )
    
#     img_byte = base64.b64decode(response.data[0].b64_json)
#     # Save the image to the specified path
#     output_path = os.path.join(data_root, 'keyframes', f'{exp_name}', f'len(chainvis).png')
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, 'wb') as img_file:
#         img_file.write(img_byte)
        
#     return img_byte
def edit_following_frame(chainvis, prompt, data_root, exp_name):
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": prompt},
    #         ]
    #     }
    # ]

    # TODO: only consider the last frame
    # prev_img = chainvis[-1]
    # Attach previous frames
    # for img_bytes in chainvis:
    #     messages[0]["content"].append({
    #         "type": "image",
    #         "image_base64": base64.b64encode(img_bytes).decode()
    #     })

    img_bytes = generate_or_edit_image_bytes(
        prompt=prompt,
        image=chainvis[-1],
        partial_images=0
    )

    # Resize the image to 720x480
    img = Image.open(BytesIO(img_bytes))
    img = img.resize((720, 480))
    resized_img_bytes = BytesIO()
    img.save(resized_img_bytes, format="PNG")
    img_bytes = resized_img_bytes.getvalue()

    frame_idx = len(chainvis)
    output_path = os.path.join(
        data_root, "keyframes", exp_name, f"{frame_idx}.png"
    )
    with open(output_path, "wb") as f:
        f.write(img_bytes)

    return img_bytes

    

def generate_keyframe_sequence(prompt, data_root, first_frame_path, exp_name):
    if first_frame_path is not None:
        first_frame_image, result = generate_only_consequence_text(data_root, prompt, first_frame_path, exp_name)
        first_frame_caption = result["Context Frame"]
    else: 
        result = generate_consequence_first_frame_txt(prompt)
        first_frame_caption = result["Concise Prompt"] + result["Context Frame"]

        first_frame_image = generate_first_frame(first_frame_caption, result["key_physic_object"], data_root, exp_name)

    consequence = result["Consequences"]
    key_object = result["key_physic_object"]
    chainvis = [first_frame_image]
    chaintxt = [first_frame_caption]

    print("chain of texts:", chaintxt)
    # raise RuntimeError("An error occurred during the process.")

    is_last_frame = False
    MAX_FRAMES = 4

    while not is_last_frame and len(chainvis) < MAX_FRAMES:
    # while not is_last_frame:
        next_frame = generate_next_frame_txt(prompt, chainvis, chaintxt[-1], consequence)
        is_last_frame = next_frame["Last Frame"].lower() == "true"
        next_frame_caption = next_frame["Next Predicted Keyframe"]
        
        current_prompt = build_next_frame_prompt(next_frame_caption, key_object)
        next_frame_image = edit_following_frame(chainvis, current_prompt, data_root, exp_name)

        chainvis.append(next_frame_image)
        chaintxt.append(next_frame_caption)
        print("chain of texts:", chaintxt)
        print("Current number of frames:", len(chainvis))
        # raise RuntimeError("An error occurred during the process.")

    # Combine result with captions and num_frames
    return {
        **result,  # Merge the original result dictionary
        "captions": chaintxt,
        "num_frames": len(chainvis)
    }

if __name__ == "__main__":
    # Example usage
    user_prompt = "A piece of ice on a brown piece of paper sitting under the sun"
    keyframe_sequence = generate_keyframe_sequence(user_prompt)
    
    # Write into json file
    # TODO:
