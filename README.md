# Stable Diffusion 3.5

Inference-only tiny reference implementation of SD3.5 and SD3 - everything you need for simple inference using SD3.5/SD3, as well as the SD3.5 Large ControlNets, excluding the weights files.

Contains code for the text encoders (OpenAI CLIP-L/14, OpenCLIP bigG, Google T5-XXL) (these models are all public), the VAE Decoder (similar to previous SD models, but 16-channels and no postquantconv step), and the core MM-DiT (entirely new).

Note: this repo is a reference library meant to assist partner organizations in implementing SD3.5/SD3. For alternate inference, use [Comfy](https://github.com/comfyanonymous/ComfyUI).

## Updates

- Nov 26, 2024 : Released ControlNets for SD3.5-Large.
- Oct 29, 2024 : Released inference code for SD3.5-Medium.
- Oct 24, 2024 : Updated code license to MIT License.
- Oct 22, 2024 : Released inference code for SD3.5-Large, Large-Turbo. Also works on SD3-Medium.

## Download

Download the following models from HuggingFace into `models` directory:
1. [Stability AI SD3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors) or [Stability AI SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/main/sd3.5_large_turbo.safetensors) or [Stability AI SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/sd3.5_medium.safetensors)
2. [OpenAI CLIP-L](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_l.safetensors)
3. [OpenCLIP bigG](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_g.safetensors)
4. [Google T5-XXL](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/t5xxl_fp16.safetensors)

This code also works for [Stability AI SD3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium.safetensors).

### ControlNets

Optionally, download [SD3.5 Large ControlNets](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets):
- [Blur ControlNet](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/blur_8b.safetensors)
- [Canny ControlNet](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/canny_8b.safetensors)
- [Depth ControlNet](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/depth_8b.safetensors)

```py
from huggingface_hub import hf_hub_download
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_blur.safetensors", local_dir="models")
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors", local_dir="models")
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_depth.safetensors", local_dir="models")
```

## Install

```sh
# Note: on windows use "python" not "python3"
python3 -s -m venv .sd3.5
source .sd3.5/bin/activate
# or on windows: venv/scripts/activate
python3 -s -m pip install -r requirements.txt
```

## Run

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

Optionally, use [Skip Layer Guidance](https://github.com/comfyanonymous/ComfyUI/pull/5404) for potentially better struture and anatomy coherency from SD3.5-Medium.
```sh
python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_medium.safetensors --skip_layer_cfg True
```

### ControlNets

To use SD3.5 Large ControlNets, additionally download your chosen ControlNet model from the [model repository](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets), then run inference, like so:
- Blur:
```sh
python sd3_infer.py --model models/sd3.5_large.safetensors --controlnet_ckpt models/sd3.5_large_controlnet_blur.safetensors --controlnet_cond_image inputs/blur.png --prompt "generated ai art, a tiny, lost rubber ducky in an action shot close-up, surfing the humongous waves, inside the tube, in the style of Kelly Slater"
```
- Canny:
```sh
python sd3_infer.py --model models/sd3.5_large.safetensors --controlnet_ckpt models/sd3.5_large_controlnet_canny.safetensors --controlnet_cond_image inputs/canny.png --prompt "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms"
```
- Depth:
```sh
python sd3_infer.py --model models/sd3.5_large.safetensors --controlnet_ckpt models/sd3.5_large_controlnet_depth.safetensors --controlnet_cond_image inputs/depth.png --prompt "photo of woman, presumably in her mid-thirties, striking a balanced yoga pose on a rocky outcrop during dusk or dawn. She wears a light gray t-shirt and dark leggings. Her pose is dynamic, with one leg extended backward and the other bent at the knee, holding the moon close to her hand."
```

For details on preprocessing for each of the ControlNets, and examples, please review the [model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets).

## File Guide

- `sd3_infer.py` - entry point, review this for basic usage of diffusion model
- `sd3_impls.py` - contains the wrapper around the MMDiTX and the VAE
- `other_impls.py` - contains the CLIP models, the T5 model, and some utilities
- `mmditx.py` - contains the core of the MMDiT-X itself
- folder `models` with the following files (download separately):
    - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL/SD3, can grab a public copy)
    - `clip_g.safetensors` (openclip bigG, same as SDXL/SD3, can grab a public copy)
    - `t5xxl.safetensors` (google T5-v1.1-XXL, can grab a public copy)
    - `sd3.5_large.safetensors` or `sd3.5_large_turbo.safetensors` or `sd3.5_medium.safetensors` (or `sd3_medium.safetensors`)

## Code Origin

The code included here originates from:
- Stability AI internal research code repository (MM-DiT)
- Public Stability AI repositories (eg VAE)
- Some unique code for this reference repo written by Alex Goodwin and Vikram Voleti for Stability AI
- Some code from ComfyUI internal Stability implementation of SD3 (for some code corrections and handlers)
- HuggingFace and upstream providers (for sections of CLIP/T5 code)

## Legal

Check the LICENSE-CODE file.

### Note

Some code in `other_impls` originates from HuggingFace and is subject to [the HuggingFace Transformers Apache2 License](https://github.com/huggingface/transformers/blob/main/LICENSE)
