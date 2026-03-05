# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Logger for generated images."""

import itertools
import random
from typing import Any

import torch
from composer import Callback, Logger, State, TimeUnit
from composer.utils import dist
from torch.nn.parallel import DistributedDataParallel

from prx.dataset.constants import BatchKeys
from prx.pipeline.composer_pipeline import Pipeline 


class LogDiffusionImages(Callback):
    """Logs images generated from the evaluation prompts to a logger.

    Logs eval prompts and generated images to a table at
    the end of an evaluation batch.

    Args:
        prompt (str or list[str]): The prompt or prompts to guide the image generation.
        negative_prompt (str or list[str]): The prompt or prompts to guide the image generation away from.
        guidance_scale (float, optional): guidance_scale is defined as w of equation 2
            of the Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
            A larger guidance scale generates images that are more aligned to
            the text prompt, usually at the expense of lower image quality.
            Default: ``1.0``.
        size (int, optional): Image size to use during generation. Default: ``256``.
        seed (int, optional): Random seed to use for generation. Set a seed for reproducible generation.
            Default: `None`.
        num_inference_steps (int): Number of inference steps for the default denoiser. Default: ``50``.
        extra_denoisers_num_inference_steps (list[int], optional): Number of inference steps for each extra denoiser in
            ``extra_denoiser_names`` (same order). If not provided, extra denoisers use ``num_inference_steps``.
            If provided, its length must equal ``len(extra_denoiser_names)``.
        extra_denoisers_guidance_scales (list[float], optional): Guidance scales for each extra denoiser in
            ``extra_denoiser_names`` (same order). If not provided, extra denoisers use ``guidance_scale``.
            If provided, its length must equal ``len(extra_denoiser_names)``.
        extra_denoiser_names (list[str], optional): Additional model attribute names of denoisers to generate with
            (e.g. ["denoiser_teacher"]). Each will be run in addition to the default model denoiser and
            logged using its attribute name as the prefix. Default: ``None``.
    """

    def __init__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        guidance_scale: float = 1.0,
        size: int = 256,
        seed: int | None = None,
        num_inference_steps: int = 50,  # TODO move into a model inference config
        extra_denoiser_names: list[str] | None = None,
        extra_denoisers_num_inference_steps: list[int] | None = None,
        extra_denoisers_guidance_scales: list[float] | None = None,
        **generate_kwargs: Any,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if negative_prompt is None:
            negative_prompt = ["" for _ in prompt]
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt for _ in prompt]
        else:
            assert len(negative_prompt) == len(prompt), "negative_prompt must be the same length as prompt"

        # Calculate padding for even distribution across all GPUs across all nodes.
        # Handles less prompts than world size and more prompts than world size.
        world_size = dist.get_world_size()
        self.padding_needed = (world_size - len(prompt) % world_size) % world_size

        if self.padding_needed:
            prompt += [""] * self.padding_needed
            negative_prompt += [""] * self.padding_needed

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.size = size
        self.possible_aspect_ratio = {1.0: (size, size)}
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.generate_kwargs = generate_kwargs
        self.extra_denoiser_names = extra_denoiser_names or []
        self.extra_denoisers_num_inference_steps = extra_denoisers_num_inference_steps
        self.extra_denoisers_guidance_scales = extra_denoisers_guidance_scales

    def get_model(self, state: State) -> Pipeline:
        if isinstance(state.model, DistributedDataParallel):
            return state.model.module
        else:
            return state.model

    def batch_end(self, state: State, logger: Logger) -> None:
        # Update the list of possible aspect ratios
        model = self.get_model(state)
        h, w = model.get_image_size_from_batch(state.batch)

        if (ar := w / h) not in self.possible_aspect_ratio:
            print(f" > Adding new aspect ratio {ar:.2f} (image size:{(h, w)}) to possible aspect ratios")
            self.possible_aspect_ratio[ar] = (h, w)

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        # Only log once per eval epoch
        if state.eval_timestamp.get(TimeUnit.BATCH).value == 1:
            model = self.get_model(state)
            rank = dist.get_global_rank()
            world_size = dist.get_world_size()

            denoiser_specs = [("txt2img", None)]
            for name in self.extra_denoiser_names:
                if not hasattr(model, name):
                    raise AttributeError(f"Requested denoiser '{name}' not found on model.")
                denoiser_specs.append((name, getattr(model, name)))

            # Resolve num_inference_steps per denoiser
            extra_count = len(self.extra_denoiser_names)
            default_steps = self.num_inference_steps
            if self.extra_denoisers_num_inference_steps is None:
                extra_steps: list[int] = [default_steps] * extra_count
            else:
                if len(self.extra_denoisers_num_inference_steps) != extra_count:
                    raise ValueError(
                        f"extra_denoisers_num_inference_steps length ({len(self.extra_denoisers_num_inference_steps)}) must equal len(extra_denoiser_names) ({extra_count})."
                    )
                extra_steps = self.extra_denoisers_num_inference_steps

            default_guidance = self.guidance_scale
            if self.extra_denoisers_guidance_scales is None:
                extra_guidance: list[float] = [default_guidance] * extra_count
            else:
                if len(self.extra_denoisers_guidance_scales) != extra_count:
                    raise ValueError(
                        f"extra_denoisers_guidance_scales length ({len(self.extra_denoisers_guidance_scales)}) must equal len(extra_denoiser_names) ({extra_count})."
                    )
                extra_guidance = self.extra_denoisers_guidance_scales

            # Get this GPU's prompts using array slicing
            chunk_size = len(self.prompt) // world_size
            gpu_prompts = self.prompt[rank * chunk_size : (rank + 1) * chunk_size]
            gpu_neg_prompts = self.negative_prompt[rank * chunk_size : (rank + 1) * chunk_size]

            image_size = random.choice(list(self.possible_aspect_ratio.values()))
            seed = (self.seed or state.timestamp.batch.value) + rank
            # For each denoiser spec, generate and log
            for idx, (name_prefix, denoiser) in enumerate(denoiser_specs):
                # Generate images for assigned prompts
                gen_images: list[torch.Tensor] = []
                for i, (p, np) in enumerate(zip(gpu_prompts, gpu_neg_prompts)):
                    batch = {
                        BatchKeys.PROMPT: [p],
                        BatchKeys.NEGATIVE_PROMPT: [np],
                        BatchKeys.RESOLUTION: [image_size],
                    }

                    image: torch.Tensor = model.generate(
                        batch=batch,
                        image_size=image_size,
                        guidance_scale=(default_guidance if idx == 0 else extra_guidance[idx - 1]),
                        progress_bar=False,
                        seed=seed,
                        num_inference_steps=(default_steps if idx == 0 else extra_steps[idx - 1]),
                        denoiser=denoiser,
                        **self.generate_kwargs,
                    )
                    gen_images.append(image)

                # Gather all generated images
                gathered_images = [None] * world_size
                torch.distributed.gather_object(gen_images, gathered_images if rank == 0 else None)

                # Log images on rank 0
                if rank == 0:
                    # Flatten list of images and, if padding was added, drop the padded tail
                    all_images = list(itertools.chain.from_iterable(gathered_images))  # type: ignore
                    if self.padding_needed:
                        all_images = all_images[: -self.padding_needed]

                    prompts = self.prompt
                    if self.padding_needed:
                        prompts = prompts[: -self.padding_needed]

                    for i, (image, prompt) in enumerate(zip(all_images, prompts)):
                        truncated_prompt: str = prompt[:90] + "..." if len(prompt) > 90 else prompt
                        logger.log_images(
                            images=image.to(torch.float32),
                            name=f"{name_prefix}_{i:02}: {truncated_prompt}",
                            step=state.timestamp.batch.value,
                            use_table=False,
                        )
