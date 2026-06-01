<div align="center">

<h1>CausalMotion: Structured Physical Reasoning as Keyframe and Trajectory Guidance for Training-Free Video Generation</h1>



[![](https://img.shields.io/static/v1?label=CausalMotion&message=Project&color=purple)](https://zhuangsh0713.github.io/CausalMotion/)   [![](https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv)]()   [![](https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github)]()   

                    
<p>Sihan Zhuang</a><sup>1,2*</sup>,
<a href="https://scholar.google.com/citations?user=3fWSC8YAAAAJ">Xinyuan Chen</a><sup>2&dagger;</sup>,
<a href="https://tianfan.info/">Tianfan Xue</a><sup>3,2&dagger;</sup>,
<a href="https://wyhsirius.github.io/">Yaohui Wang</a><sup>2&dagger;</sup></p>


<span class="author-block"><sup>1</sup>ShanghaiTech University</span>
<span class="author-block"><sup>2</sup>Shanghai Artificial Intelligence Laboratory</span>
<span class="author-block"><sup>3</sup>The Chinese University of Hong Kong</span>


</div>

## Overview

CausalMotion is a training-free video generation pipeline that combines:

- structured chain-of-thought keyframe planning,
- object grounding and segmentation,
- visual-language trajectory planning, and
- latent trajectory guidance during video synthesis.

The repository contains the `causalmotion` Python package and expects two external components alongside it:

- `Grounded-SAM-2` for SAM2-based grounding / segmentation
- `LTX-Video` for final video generation

These are tracked as git submodules in this repo.

## Repository Layout

```text
CausalMotion/
├── causalmotion/
│   ├── main.py
│   ├── caption_and_keyframe.py
│   ├── detect_objs.py
│   ├── traj_plan.py
│   ├── align.py
│   ├── generate_video.py
│   └── utils/
├── requirements.txt
├── .gitmodules
└── README.md
```

At runtime, the project expects the following sibling directories to exist at the repository root:

```text
CausalMotion/
├── Grounded-SAM-2/
├── LTX-Video/
└── causalmotion/
```

## Installation

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules <your-repo-url>
cd CausalMotion
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

If you are publishing this directory as a brand-new repository by copying files manually, `.gitmodules` alone is not enough. In that case, either:

- re-add `Grounded-SAM-2` and `LTX-Video` as real git submodules in the new repo, or
- clone them manually into the repository root:

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
git clone https://github.com/zhuangsh0713/LTX-Video.git
```

### 2. Create a Python environment

Python `3.10` is recommended.

```bash
conda create -n causalmotion python=3.10 -y
conda activate causalmotion
```

### 3. Install PyTorch

Install a `torch` / `torchvision` build matching your CUDA runtime before installing the rest of the dependencies. Follow the official PyTorch instructions if you need a different CUDA version.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e ./Grounded-SAM-2
pip install -e ./LTX-Video
```

If you want the local Grounding DINO build used by the upstream `Grounded-SAM-2` project, also run:

```bash
pip install --no-build-isolation -e ./Grounded-SAM-2/grounding_dino
```

## Model Assets

### SAM2 checkpoint

Download the SAM2 checkpoints inside `Grounded-SAM-2`:

```bash
cd Grounded-SAM-2/checkpoints
bash download_ckpts.sh
cd ../..
```

For the default pipeline, the expected checkpoint path is:

```text
./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
```

### Grounding model

By default, CausalMotion uses the Hugging Face Grounding DINO model path supplied through:

```text
--grounding_model_path IDEA-Research/grounding-dino-base
```

This model is downloaded automatically by `transformers` on first use.

## API Configuration

The reasoning and keyframe generation stages require an OpenAI-compatible API.

Required:

```bash
export OPENAI_API_KEY=...
```

Optional:

```bash
export OPENAI_BASE_URL=...
export CAUSALMOTION_VLM_MODEL=qwen2.5-vl-72b-instruct
export CAUSALMOTION_IMAGE_MODEL=gpt-image-1.5
```

Notes:

- `OPENAI_BASE_URL` is only needed if you are using a non-default OpenAI-compatible provider.
- The default model names are the ones used in the original experiments.
- If you use the official OpenAI endpoint, set model names that are available in your account.

## Running the Full Pipeline

Run from the repository root:

```bash
python -m causalmotion.main \
  --exp_name demo \
  --prompt "A steel ball is dropped into water." \
  --data_root ./data/demo \
  --grounding_model_path IDEA-Research/grounding-dino-base \
  --sam2_checkpoint ./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
```

If you already have a starting image, add:

```bash
  --first_frame_path /path/to/first_frame.png
```

## Batch Runs from CSV

For dataset-style runs, use the helper script in `scripts/`:

```bash
python scripts/run_causalmotion_from_csv.py \
  --prompts-csv ./prompts.csv \
  --data-root ./data/batch_run \
  --prompt-key description \
  --skip-existing
```

The CSV script launches `python -m causalmotion.main` for each row and writes a manifest containing the resolved prompts, experiment names, and commands.

## Data Layout

`--data_root` is both the input workspace and the output workspace. CausalMotion creates the following structure automatically:

```text
data_root/
├── keyframes/
│   └── <exp_name>/
│       ├── 0.png
│       ├── 1.png
│       └── ...
├── json/
│   └── <exp_name>_results.json
├── mappings/
│   └── all_mappings.json
├── alignment_videos/
│   └── <exp_name>.mp4
├── ltx_runs/
└── results/
    └── <exp_name>.mp4
```

<!-- ## Citation

If you use this code, please cite the project paper when it becomes available.

```bibtex
@misc{causalmotion2026,
  title={CausalMotion: Structured Physical Reasoning as Keyframe and Trajectory Guidance for Training-Free Video Generation},
  author={Sihan Zhuang and Xinyuan Chen and Tianfan Xue and Yaohui Wang},
  year={2026}
}
``` -->

