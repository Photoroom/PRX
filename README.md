<p align="center">
  <img src="assets/PRX.png" alt="PRX" width="500">
</p>
<p align="center">
  <img src="assets/mosaic.png" alt="PRX samples" width="900">
</p>


# PRX

Training framework for the PRX text-to-image diffusion models by [Photoroom](https://www.photoroom.com/).

Read the full story on the [Hugging Face blog](https://huggingface.co/blog/Photoroom/prx-open-source-t2i-model).

## Overview

PRX is a transformer-based latent diffusion model trained with flow matching. This repository contains everything needed to train and evaluate PRX models, including:

- A patchified transformer denoiser
- Support for multiple text encoders (T5, T5-Gemma2B, Qwen3) and VAEs (AutoencoderKL, DC-AE)
- Distributed training via [MosaicML Composer](https://github.com/mosaicml/composer) with FSDP
- Training algorithms: EMA, REPA/iREPA, SPRINT, TREAD, contrastive flow matching, Perceptual losses (P-DINO, LPIPS), etc.
- Evaluation metrics: FID, CMMD, DINO-MMD

## Pre-trained models

Pre-trained PRX models are available on Hugging Face and can be used directly with [diffusers](https://huggingface.co/Photoroom/prx-1024-t2i-beta).

## Installation

Requires Python 3.11+.

```bash
uv sync

# With optional dependencies
uv sync --extra streaming   # MosaicML Streaming dataset support
uv sync --extra lpips       # LPIPS perceptual loss
uv sync --all-extras        # Everything
```

## Training

Training is configured with [Hydra](https://hydra.cc/) YAML files. See [`configs/yamls/`](configs/yamls/) for examples. The repository includes all the training configurations used in the benchmarks presented in the [blog post](https://huggingface.co/blog/Photoroom/prx-part2).

```bash
composer -m prx.training.train --config-path=configs/yamls hydra/launcher=basic
```

## Data

PRX trains on [MosaicML Streaming (MDS)](https://github.com/mosaicml/streaming) datasets, organized into aspect-ratio buckets. We provide a conversion script that takes WebDataset-style tar files and produces AR-bucketed MDS shards ready for training.

### Example: fine-t2i dataset

1. **Download** the [fine-t2i](https://huggingface.co/datasets/ma-xu/fine-t2i) dataset:

```bash
HF_HUB_CACHE=/path/to/cache huggingface-cli download ma-xu/fine-t2i --repo-type dataset
```

2. **Convert** to AR-bucketed MDS (images resized to 1024-base AR buckets, 27 buckets, patch_size=32):

```bash
uv run scripts/fine-t2i-to-mds.py \
    --input /path/to/cache/hub/datasets--ma-xu--fine-t2i/snapshots/<hash> \
    --output /path/to/output/fine-t2i \
    --workers 16
```

This produces one MDS subdirectory per aspect ratio (e.g. `0.667/`, `1.000/`, `1.500/`), each containing sharded MDS files with a merged `index.json`.

3. **Train** by pointing a dataset config at the output directory. Create a dataset YAML (e.g. `configs/yamls/dataset/train_fine_t2i.yaml`):

```yaml
# @package dataset.train_dataset
_target_: prx.dataset.StreamingProcessedDataset
local:
  - /path/to/output/fine-t2i

caption_keys:
  - [prompt, 0.5]
  - [enhanced_prompt, 0.5]
has_text_latents: false
text_tower: ${diffusion_text_tower.preset_name}
cache_limit: 8tb
drop_last: true
shuffle: true
batching_method: device_per_stream
num_workers: 8
persistent_workers: true
pin_memory: true
transforms:
  - _target_: prx.dataset.transforms.ArAwareResize
    default_image_size: ${image_size}
    patch_size_pixels: ${patch_size_pixels}
transforms_targets:
  - image
```

> **Note:** `ArAwareResize` is still used at training time even though images were already resized during MDS export. The MDS conversion targets a fixed 1024-base resolution, but the training config may use a different `image_size` (e.g. 512 for early-stage training). `ArAwareResize` ensures images are resized to match the model's current resolution and patch grid, and also handles any JPEG decode size differences.

Then launch training referencing your dataset config:

```bash
composer -m prx.training.train \
    --config-path=configs/yamls \
    hydra/launcher=basic \
    dataset/train_dataset=train_fine_t2i
```

See the [JIT-benchmark configs](configs/yamls/JIT-benchmark/) for full training configuration examples.


## License

Apache 2.0

## Acknowledgments

PRX is built by the [Photoroom](https://www.photoroom.com/) machine learning team and large parts of this codebase were built on top of an existing private Photoroom codebase. The following previous team members made significant contributions to the foundations of this project and deserve credit, even though their work may not appear in the public git history:

- [Antoine d'Andigné](https://github.com/antoinedandi)
- [Quentin Desreumaux](https://github.com/quentindrx)
- [Benjamin Lefaudeux](https://github.com/blefaudeux)

PRX team: [David Bertoin](https://github.com/DavidBert), [Roman Frigg](https://github.com/photoroman), 
[Jon Almazán](https://github.com/almazan), 
[Eliot Andres](https://github.com/eliotandres)
