# Stable Diffusion 3.5

Inference-only tiny reference implementation of SD3.5 and SD3 - everything you need for simple inference using SD3.5/SD3, excluding the weights files.

Contains code for the text encoders (OpenAI CLIP-L/14, OpenCLIP bigG, Google T5-XXL) (these models are all public), the VAE Decoder (similar to previous SD models, but 16-channels and no postquantconv step), and the core MM-DiT (entirely new).

Note: this repo is a reference library meant to assist partner organizations in implementing SD3.5/SD3. For alternate inference, use [Comfy](https://github.com/comfyanonymous/ComfyUI).

### Updates

- Oct 29, 2024 : Released inference code for SD3.5-Medium.
- Oct 24, 2024 : Updated code license to MIT License.
- Oct 22, 2024 : Released inference code for SD3.5-Large, Large-Turbo. Also works on SD3-Medium.

### Download

Download the following models from HuggingFace into `models` directory:
1. [Stability AI SD3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors) or [Stability AI SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/main/sd3.5_large_turbo.safetensors) or [Stability AI SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/sd3.5_medium.safetensors)
2. [OpenAI CLIP-L](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_l.safetensors)
3. [OpenCLIP bigG](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_g.safetensors)
4. [Google T5-XXL](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/t5xxl_fp16.safetensors)

This code also works for [Stability AI SD3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium.safetensors).

### Install

```sh
# Note: on windows use "python" not "python3"
python3 -s -m venv .sd3.5
source .sd3.5/bin/activate
# or on windows: venv/scripts/activate
python3 -s -m pip install -r requirements.txt
```

### Run

```sh
# Generate a cat using SD3.5 Large model (at models/sd3.5_large.safetensors) with its default settings
python3 sd3_infer.py --prompt "cute wallpaper art of a cat"
# Or use a text file with a list of prompts, using SD3.5 Large
python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_large.safetensors
# Generate from prompt file using SD3.5 Large Turbo with its default settings
python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_large_turbo.safetensors
# Generate from prompt file using SD3.5 Medium with its default settings, at 2k resolution
python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_medium.safetensors --width 1920 --height 1080
# Generate from prompt file using SD3 Medium with its default settings
python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3_medium.safetensors
```

Images will be output to `outputs/<MODEL>/<PROMPT>_<DATETIME>_<POSTFIX>` by default.
To add a postfix to the output directory, add `--postfix <my_postfix>`. For example,
```sh
python3 sd3_infer.py --prompt path/to/my_prompts.txt --postfix "steps100" --steps 100
```

To change the resolution of the generated image, add `--width <WIDTH> --height <HEIGHT>`.

### File Guide

- `sd3_infer.py` - entry point, review this for basic usage of diffusion model
- `sd3_impls.py` - contains the wrapper around the MMDiTX and the VAE
- `other_impls.py` - contains the CLIP models, the T5 model, and some utilities
- `mmditx.py` - contains the core of the MMDiT-X itself
- folder `models` with the following files (download separately):
    - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL/SD3, can grab a public copy)
    - `clip_g.safetensors` (openclip bigG, same as SDXL/SD3, can grab a public copy)
    - `t5xxl.safetensors` (google T5-v1.1-XXL, can grab a public copy)
    - `sd3.5_large.safetensors` or `sd3.5_large_turbo.safetensors` or `sd3.5_medium.safetensors` (or `sd3_medium.safetensors`)

### Code Origin

The code included here originates from:
- Stability AI internal research code repository (MM-DiT)
- Public Stability AI repositories (eg VAE)
- Some unique code for this reference repo written by Alex Goodwin and Vikram Voleti for Stability AI
- Some code from ComfyUI internal Stability implementation of SD3 (for some code corrections and handlers)
- HuggingFace and upstream providers (for sections of CLIP/T5 code)

### Legal

Check the LICENSE-CODE file.

#### Note

Some code in `other_impls` originates from HuggingFace and is subject to [the HuggingFace Transformers Apache2 License](https://github.com/huggingface/transformers/blob/main/LICENSE)
