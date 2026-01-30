# Z-Image
This directory contains MFLUX's MLX implementation of **Z-Image** and **Z-Image-Turbo**.

MFLUX supports [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) from Tongyi Lab (Alibaba), released in November 2025. Z-Image is an efficient 6B-parameter image generation model with a single-stream DiT architecture. See the [technical paper](https://arxiv.org/abs/2511.22699) for more details.

| Model | Steps | CFG | Speed | Best for |
| --- | --- | --- | --- | --- |
| Z-Image (Base) | 28-50 | Yes (guidance 3-5) | Slower | Maximum quality, complex prompts, negative prompt control |
| Z-Image-Turbo | 8-9 | No (distilled) | Fast | Quick iterations, general use |

All the standard modes such as img2img, LoRA and quantizations are supported for both models.

![Z-Image-Turbo Example](../../assets/z_image_turbo_example.jpg)

## Z-Image (Base Model)

The base model supports full Classifier-Free Guidance (CFG), providing higher quality output with support for negative prompts. Recommended settings: 28-50 steps with guidance scale 3.0-5.0.

### CLI Example

```sh
mflux-generate-z-image \
  --prompt "A serene coastal landscape at sunset with dramatic clouds over the ocean" \
  --negative-prompt "blurry, distorted, low quality" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --steps 40 \
  --guidance 4.0 \
  -q 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.z_image import ZImage

model = ZImage(quantize=8)
image = model.generate_image(
    seed=42,
    prompt="A serene coastal landscape at sunset with dramatic clouds over the ocean",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=40,
    guidance=4.0,
    width=1024,
    height=1024,
)
image.save("z_image_output.png")
```
</details>

> [!NOTE]
> The base model runs the transformer twice per step (for CFG), so generation takes roughly twice as long per step compared to Turbo. However, the improved quality from CFG often allows using fewer total steps while achieving better results.

## Z-Image-Turbo (Distilled Model)

Z-Image-Turbo is a distilled model that has CFG behavior baked in, delivering high-quality images in just 8-9 steps without needing guidance scaling. This makes it one of the fastest open-source models available.

### CLI Example

```sh
mflux-generate-z-image-turbo \
  --prompt "A puffin standing on a cliff" \
  --width 1280 \
  --height 720 \
  --seed 42 \
  --steps 9 \
  -q 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    seed=42,
    prompt="A puffin standing on a cliff",
    num_inference_steps=9,
    width=1280,
    height=720,
)
image.save("z_image_turbo.png")
```
</details>

### Advanced Example with LoRA

The following uses the pre-quantized 4-bit model from [filipstrand/Z-Image-Turbo-mflux-4bit](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit) with a LoRA adapter [Technically Color](https://huggingface.co/renderartist/Technically-Color-Z-Image-Turbo) for enhanced film color:

```sh
mflux-generate-z-image-turbo \
  --model filipstrand/Z-Image-Turbo-mflux-4bit \
  --prompt "t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground 1960s style film She's holding her hand out there is a small smooth frog in her palm, she's making eye contact with the toad." \
  --width 1280 \
  --height 720 \
  --seed 456 \
  --steps 9 \
  --lora-paths renderartist/Technically-Color-Z-Image-Turbo \
  --lora-scales 0.5
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    model_config=ModelConfig.z_image_turbo(),
    model_path="filipstrand/Z-Image-Turbo-mflux-4bit",
    lora_paths=["renderartist/Technically-Color-Z-Image-Turbo"],
    lora_scales=[0.5],
)
image = model.generate_image(
    seed=456,
    prompt="t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground 1960s style film She's holding her hand out there is a small smooth frog in her palm, she's making eye contact with the toad.",
    num_inference_steps=9,
    width=1280,
    height=720,
)
image.save("z_image_turbo.png")
```
</details>

## Model Weights

> [!WARNING]
> Note: Z-Image models require downloading model weights (~31GB each), or use quantization for smaller sizes.

| Model | Full weights | Quantized |
| --- | --- | --- |
| Z-Image | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | Use `-q 4` or `-q 8` |
| Z-Image-Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | [filipstrand/Z-Image-Turbo-mflux-4bit](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit) |

## Additional Resources

*Dreambooth fine-tuning for Z-Image is not yet supported in MFLUX but is planned. In the meantime, you can train Z-Image-Turbo LoRAs using [AI Toolkit](https://github.com/ostris/ai-toolkit) - see [How to Train a Z-Image-Turbo LoRA with AI Toolkit](https://www.youtube.com/watch?v=Kmve1_jiDpQ) by Ostris AI.*

*For a Swift MLX implementation of Z-Image, see [zimage.swift](https://github.com/mzbac/zimage.swift) by [@mzbac](https://github.com/mzbac).*
