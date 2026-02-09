import copy
from enum import StrEnum, auto
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric
from tqdm.auto import tqdm

from dataset.constants import BatchKeys
from schedulers.scheduler import BaseScheduler
from models.text_tower import TextTower


class ModelInputs(StrEnum):
    """Model input parameter names for the denoiser.

    This enum provides a model-agnostic interface for passing arguments
    to the denoiser, mapping from generic keys to specific parameter names.
    """
    IMAGE = auto()
    PROMPT = auto()
    PROMPT_EMBEDS = auto()
    PROMPT_ATTENTION_MASK = auto()
    IMAGE_LATENT = auto()


class PredictionType(StrEnum):
    """Noise scheduler prediction types for diffusion models."""
    FLOW_MATCHING = auto()
    X_PREDICTION_FLOW_MATCHING = auto()


class ImageSize(NamedTuple):
    height: int
    width: int


class ForwardOutput(TypedDict, total=False):
    """Output from forward/eval_forward pass."""
    prediction: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    generated_images: Dict[float, torch.Tensor]  # Only present in eval_forward



class EMAModel(torch.nn.Module):
    """Wrapper for models with EMA (Exponential Moving Average) functionality.

    This is a lightweight wrapper that holds a copy of a model for EMA purposes.
    The actual EMA weight updates are performed by the EMA Algorithm in
    algorithm/ema.py, which calls compute_ema() during training.

    This wrapper manages:
    - Activation state (whether EMA is active)
    - Model initialization and weight copying
    - Forward pass delegation to the wrapped model
    """

    def __init__(self) -> None:
        super().__init__()
        _is_active = torch.tensor(False)
        self.register_buffer("_is_active", _is_active, persistent=True)

    @property
    def is_active(self) -> bool:
        return bool(self._is_active.item())

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._is_active.fill_(value)

    def init_model(self, model: torch.nn.Module) -> None:
        """Initialize the EMA model with the weights of the provided model."""
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def copy_weights_from_source(self, source_model: torch.nn.Module, strict: bool = False) -> None:
        """Copy weights from source_model to self.model."""
        self.model.load_state_dict(source_model.state_dict(), strict=strict)

class LatentDiffusion(ComposerModel):
    """Latent Diffusion pipeline for training and inference.

    Extends ComposerModel for integration with the Composer training framework.

    Args:
        denoiser: The denoising model (e.g., UNet, DiT).
        vae: Variational autoencoder for encoding/decoding images to/from latent space.
        text_tower: Text encoder for processing prompts into embeddings.
        noise_scheduler: Scheduler for adding noise during training.
        inference_noise_scheduler: Scheduler for denoising during inference.
        p_drop_caption: Probability of dropping text conditioning for classifier-free guidance training.
        train_metrics: List of metrics to track during training.
        val_metrics: List of metrics to track during validation.
        val_seed: Random seed for reproducible validation sampling.
        val_guidance_scales: List of guidance scales to use during validation generation.
        loss_bins: Timestep bins for computing per-bin losses, as (start, end) tuples.
        negative_prompt: Default negative prompt for classifier-free guidance.
    """

    _NULL_PROMPT_EMBEDDING_ATTR = "null_prompt_embedding"

    def __init__(
        self,
        denoiser: torch.nn.Module,
        vae: torch.nn.Module,
        text_tower: TextTower,
        noise_scheduler: BaseScheduler,
        inference_noise_scheduler: BaseScheduler,
        p_drop_caption: float = 0.1,
        train_metrics: Optional[List[str]] = None,
        val_metrics: Optional[List[str]] = None,
        val_seed: int = 1138,
        val_guidance_scales: Optional[List[float]] = None,
        loss_bins: Optional[List[Tuple[int, int]]] = None,
        negative_prompt: str = "",
    ):
        super().__init__()

        self.denoiser = denoiser
        self.ema_denoiser = EMAModel()
        self.vae = vae
        self.text_tower = text_tower
        self.noise_scheduler: BaseScheduler = noise_scheduler
        self.inference_scheduler: BaseScheduler = inference_noise_scheduler

        if not 0.0 <= p_drop_caption <= 1.0:
            raise ValueError(f"p_drop_caption must be between 0.0 and 1.0, got {p_drop_caption}")
        self.p_drop_caption = p_drop_caption

        self.sampled_timesteps = torch.tensor([])
        self.val_seed = val_seed

        self.vae_scale_factor = self.vae.vae_scale_factor
        self.vae_channels = self.vae.vae_channels

        self.negative_prompt = negative_prompt

        train_metrics = train_metrics or [MeanSquaredError()]
        val_guidance_scales = val_guidance_scales if val_guidance_scales is not None else [0.0]
        loss_bins = loss_bins or [(0, 1)]

        self.train_metrics = train_metrics
        self.val_guidance_scales = val_guidance_scales

        self.val_metrics = {}
        self.val_metrics["MeanSquaredError"] = MeanSquaredError()

        for bin in loss_bins:
            new_metric = MeanSquaredError()
            new_metric.loss_bin = bin  # type: ignore
            self.val_metrics[f"MeanSquaredError-bin-{bin[0]}-to-{bin[1]}".replace(".", "p")] = new_metric


    @property
    def denoiser_device(self) -> torch.device:
        """Returns the denoiser device"""
        return next(self.denoiser.parameters()).device

    @property
    def denoiser_dtype(self) -> torch.dtype:
        """Returns the denoiser dtype"""
        return next(self.denoiser.parameters()).dtype

    # ============================================================
    # BATCH PROCESSING (Training & Inference Shared)
    # ============================================================

    def prepare_batch(self, batch: Dict[BatchKeys, Any]) -> Dict[ModelInputs, Any]:
        """
        Return the right model arguments for the model from the batch.
        Classifier free guidance is not done here.

        Return Image latent and text embedding.
        """
        denoiser_kwargs = {}
        denoiser_kwargs.update(self.get_image_latents(batch))
        denoiser_kwargs.update(self.get_text_embedding(batch))
        return denoiser_kwargs

    @torch.inference_mode()  # type: ignore
    def get_image_latents(self, batch: Dict[BatchKeys, Any]) -> Dict[ModelInputs, torch.Tensor]:
        """Return the image latent from the batch. Check for precomputed latents first"""
        if BatchKeys.IMAGE_LATENT in batch:
            image_latent = self.scale_image_latent(batch[BatchKeys.IMAGE_LATENT])
        elif BatchKeys.IMAGE in batch:
            image_latent = self.image_to_latent(batch[BatchKeys.IMAGE])
        else:
            raise ValueError(
                "Could not get the image latent from the batch. "
                "Batch must contain either 'image_latent' or 'image' key."
            )

        return {ModelInputs.IMAGE_LATENT: image_latent}

    def encode_single_text(self, text: str, device: torch.device) -> Dict[str, Any]:
        """Encode a single text string to embeddings.

        Note: The text_tower already implements internal caching for efficiency.
        """
        return {k: v.to(device) for k, v in self.text_tower([text]).items()}

    def encode_texts(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        outputs = [self.encode_single_text(text, device) for text in texts]
        # list of dict to dict
        return {key: torch.vstack([i[key] for i in outputs]) for key in outputs[0]}

    @torch.no_grad()  # type: ignore
    def get_text_embedding(self, batch: Dict[BatchKeys, Any]) -> Dict[ModelInputs, torch.Tensor]:
        """Return the text latent from the batch. Check for precomputed latents first"""
        if BatchKeys.PROMPT_EMBEDDING in batch:
            text_mask = batch.get(BatchKeys.PROMPT_EMBEDDING_MASK) if self.text_tower.use_attn_mask else None
            if self.text_tower.use_attn_mask and text_mask is None:
                raise ValueError("Could not get the text embeddings masks from the batch.")
            text_embedding = batch[BatchKeys.PROMPT_EMBEDDING]
        elif BatchKeys.PROMPT in batch:
            text_tower_output = self.encode_texts(batch[BatchKeys.PROMPT], device=self.denoiser_device)
            text_mask = text_tower_output.get("attention_mask", None)
            text_embedding = text_tower_output["text_embed"]
        else:
            raise ValueError(
                "Could not get the text embeddings from the batch. "
                "Batch must contain either 'prompt_embedding' or 'prompt' key."
            )

        output = {ModelInputs.PROMPT_EMBEDS: text_embedding.to(self.denoiser_dtype)}
        if text_mask is not None:
            output[ModelInputs.PROMPT_ATTENTION_MASK] = text_mask
        return output

    # ============================================================
    # CONDITIONING (CFG & Dropout)
    # ============================================================

    def make_cfg_batch(
        self, denoiser_kwargs: Dict[ModelInputs, Any], batch: Dict[BatchKeys, Any]
    ) -> Dict[ModelInputs, Any]:
        """
        Method to duplicate all the model arguments to do classifier free guidance.
        All arguments are duplicated except:
          - the image latent: the generate function takes care of duplicated the latent after each denoising step
          - prompt : use the negative prompt or ""
        """
        do_not_duplicate_key = [
            ModelInputs.IMAGE_LATENT,  # The generate method takes care of this
            ModelInputs.PROMPT_EMBEDS,  # set negative prompts
            ModelInputs.PROMPT_ATTENTION_MASK,  #  goes with negative prompt
        ]

        denoiser_kwargs.update(
            {k: torch.concat([v] * 2, dim=0) for k, v in denoiser_kwargs.items() if k not in do_not_duplicate_key}
        )
        # Negative Prompts
        batch_size = len(denoiser_kwargs[ModelInputs.PROMPT_EMBEDS])
        negative_prompt = batch.get(BatchKeys.NEGATIVE_PROMPT, [self.negative_prompt] * batch_size)
        device = denoiser_kwargs[ModelInputs.IMAGE_LATENT].device
        uncond_text_tower_output = self.encode_texts(negative_prompt, device=device)
        denoiser_kwargs[ModelInputs.PROMPT_EMBEDS] = torch.concat(
            [uncond_text_tower_output["text_embed"], denoiser_kwargs[ModelInputs.PROMPT_EMBEDS]], dim=0
        )
        if ModelInputs.PROMPT_ATTENTION_MASK in denoiser_kwargs:
            denoiser_kwargs[ModelInputs.PROMPT_ATTENTION_MASK] = torch.concat(
                [uncond_text_tower_output["attention_mask"], denoiser_kwargs[ModelInputs.PROMPT_ATTENTION_MASK]], dim=0
            )

        return denoiser_kwargs

    def get_denoiser_kwargs(
        self, batch: Dict[BatchKeys, Any], do_cfg: bool = False
    ) -> Dict[ModelInputs, Any]:
        """Build the denoiser argument from the batch. Shared method for inference and training."""
        denoiser_kwargs = self.prepare_batch(batch)

        if self.training:
            denoiser_kwargs = self.random_drop_conditionings(denoiser_kwargs)

        if do_cfg is True:
            denoiser_kwargs = self.make_cfg_batch(denoiser_kwargs, batch)

        return denoiser_kwargs

    def random_drop_conditionings(self, denoiser_kwargs: Dict[BatchKeys, Any]) -> Dict[BatchKeys, Any]:
        """Randomly drop the conditionings."""

        # Inplace
        self.drop_text_conditioning(denoiser_kwargs)

        return denoiser_kwargs

    # ============================================================
    # TRAINING FORWARD PASS
    # ============================================================

    def sample_timesteps(self, size: int, device: torch.device) -> torch.Tensor:
        """Return timestep arrays for training."""
        return self.noise_scheduler.sample_timesteps(size, device)

    def compile(
        self, compile_denoiser: bool = False, compile_vae: bool = False, compile_denoiser_teacher: bool = False
    ) -> None:
        def compile_model(model: torch.nn.Module, **compile_kwargs: Any) -> None:
            if hasattr(model, "experts"):
                for expert in model.experts:
                    compile_model(expert, **compile_kwargs)
                return

            compiled_model = torch.compile(model, **compile_kwargs)
            model = compiled_model._orig_mod
            model.forward = compiled_model.dynamo_ctx(model.forward)

        if compile_denoiser:
            print("> Compiling the denoiser")
            compile_model(self.denoiser)

        if compile_vae:
            print("> Compiling the vae encoder")
            # Adapted from the end of the Trainer.__init__
            compile_model(self.vae.vae.encoder, dynamic=True)


    def forward(self, batch: Dict[BatchKeys, Any], use_ema: bool = False) -> ForwardOutput:
        denoiser_kwargs = self.get_denoiser_kwargs(batch)
        latents = denoiser_kwargs.pop(ModelInputs.IMAGE_LATENT)
        # Sample the diffusion timesteps
        self.sampled_timesteps = self.sample_timesteps(latents.shape[0], latents.device)
        # Add noise to the inputs (forward diffusion)
        noise = self.get_latent_noise(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, self.sampled_timesteps)
        # Forward through the denoiser model
        if use_ema:
            assert self.ema_denoiser.is_active, "EMA denoiser is not initialized."
            denoiser = self.ema_denoiser
        else:
            denoiser = self.denoiser
        prediction = denoiser(image_latent=noised_latents, timestep=self.sampled_timesteps, **denoiser_kwargs)
        target = self.get_target(latents, noise, self.sampled_timesteps)

        if self.noise_scheduler.config.prediction_type == PredictionType.X_PREDICTION_FLOW_MATCHING:
            prediction, target = self.convert_x_to_v(prediction, target, noised_latents, self.sampled_timesteps)
            
        return {
            "prediction": prediction.contiguous(),
            "target": target,
            "timesteps": self.sampled_timesteps,
        }
    # from "Back to Basics: Let Denoising Generative Models Denoise"
    # https://arxiv.org/abs/2511.13720   
    def convert_x_to_v(self, prediction: torch.Tensor, target: torch.Tensor, noised_latents: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert x-prediction to v-prediction space"""
        t = timesteps.view(-1, 1, 1, 1)
        t = torch.clamp(t, min=0.05)
        prediction = (noised_latents - prediction) / t
        target = (noised_latents - target) / t
        return prediction, target

    def init_null_text_conditioning(self, device: torch.device) -> None:
        null_conditioning = self.encode_single_text(self.negative_prompt, device=device)
        self.null_prompt_embedding = null_conditioning["text_embed"]
        if "attention_mask" in null_conditioning:
            self.null_prompt_mask = null_conditioning["attention_mask"]

    def drop_text_conditioning(self, model_inputs: Dict[ModelInputs, torch.Tensor]) -> None:
        """
        Drop the text conditioning with probability `p_drop_caption` for each sample in the batch.
        Inplace operation.
        """
        conditioning = model_inputs[ModelInputs.PROMPT_EMBEDS]
        if not hasattr(self, self._NULL_PROMPT_EMBEDDING_ATTR):
            self.init_null_text_conditioning(conditioning.device)

        drop_mask = torch.rand(conditioning.shape[0], device=conditioning.device) < self.p_drop_caption

        model_inputs[ModelInputs.PROMPT_EMBEDS][drop_mask] = self.null_prompt_embedding.to(conditioning.dtype)
        if ModelInputs.PROMPT_ATTENTION_MASK in model_inputs:
            model_inputs[ModelInputs.PROMPT_ATTENTION_MASK][drop_mask] = self.null_prompt_mask

    def get_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.noise_scheduler.config.prediction_type == PredictionType.X_PREDICTION_FLOW_MATCHING:
            return latents
        if self.noise_scheduler.config.prediction_type == PredictionType.FLOW_MATCHING:
            return noise - latents
        else:
            raise ValueError(
                f"prediction_type given as {self.noise_scheduler.config.prediction_type} "
                f"must be one of {PredictionType.FLOW_MATCHING} or {PredictionType.X_PREDICTION_FLOW_MATCHING}"
            )

    def get_latent_noise(self, latents: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(latents)
        return noise

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[BatchKeys, Any]) -> torch.Tensor:
        """Loss between denoiser output and added noise, typically mse."""
        loss = F.mse_loss(outputs["prediction"], outputs["target"].clone(), reduction="none")
        loss = loss.mean()
        self.logger.log_metrics({"loss/train/mse": loss.detach().cpu()})

        return loss

    # ============================================================
    # VAE ENCODING/DECODING
    # ============================================================

    @torch.inference_mode()  # type: ignore
    def scale_image_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.scale_latent(latent)

    @torch.inference_mode()  # type: ignore
    def image_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(image)
        return latent.to(self.denoiser_dtype)

    @torch.inference_mode()  # type: ignore
    def latent_to_image(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)

    # ============================================================
    # UTILITIES
    # ============================================================

    def get_image_size_from_batch(self, batch: Dict[BatchKeys, Any]) -> ImageSize:
        """Return ImageSize (height, width) from a batch."""
        if BatchKeys.IMAGE in batch:
            shape = batch[BatchKeys.IMAGE].shape[-2:]
            return ImageSize(height=int(shape[0]), width=int(shape[1]))
        else:
            h, w = batch[BatchKeys.IMAGE_LATENT].shape[-2:]
            return ImageSize(height=int(h * self.vae_scale_factor), width=int(w * self.vae_scale_factor))

    def get_image_latent_size_from_batch(self, batch: Dict[BatchKeys, Any]) -> ImageSize:
        """Return ImageSize (height, width) of image latents from a batch."""
        if BatchKeys.IMAGE_LATENT in batch:
            shape = batch[BatchKeys.IMAGE_LATENT].shape[-2:]
            return ImageSize(height=int(shape[0]), width=int(shape[1]))
        else:
            h, w = batch[BatchKeys.IMAGE].shape[-2:]
            return ImageSize(height=h // self.vae_scale_factor, width=w // self.vae_scale_factor)

    def get_batch_size_from_batch(self, batch: Dict[BatchKeys, Any]) -> int:
        """Return the batch size from a batch dict."""
        for key in (BatchKeys.IMAGE_LATENT, BatchKeys.IMAGE, BatchKeys.PROMPT):
            if key in batch:
                return len(batch[key])
        raise ValueError("Cannot infer batch size: batch contains none of IMAGE_LATENT, IMAGE, or PROMPT")

    # ============================================================
    # EVALUATION & METRICS
    # ============================================================

    def eval_forward(self, batch: Dict[BatchKeys, Any], outputs: ForwardOutput | None = None) -> ForwardOutput:
        """For stable diffusion, eval forward computes denoiser outputs as well as some samples."""
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        # Get denoiser outputs
        outputs = self.forward(batch, use_ema=self.ema_denoiser.is_active)
        # Generate samples
        generated_images = {}
        for guidance_scale in self.val_guidance_scales:
            gen_images = self.generate(
                batch=batch,
                image_size=(
                    outputs["prediction"].shape[-2] * self.vae_scale_factor,
                    outputs["prediction"].shape[-1] * self.vae_scale_factor
                ),
                guidance_scale=guidance_scale,
                seed=self.val_seed,
                num_inference_steps=20,
                progress_bar=False,
                denoiser=self.ema_denoiser if self.ema_denoiser.is_active else self.denoiser,
            )
            generated_images[guidance_scale] = gen_images

        outputs["generated_images"] = generated_images
        return outputs

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            return {metric.__class__.__name__: metric for metric in self.train_metrics}
        return self.val_metrics

    def update_metric(self, batch: Dict[BatchKeys, Any], outputs: Dict[str, Any], metric: Metric) -> None:
        """Update MSE metric - either for a specific timestep bin or for all timesteps."""
        if isinstance(metric, MeanSquaredError) and hasattr(metric, "loss_bin"):
            # Update metric for a specific timestep bin
            loss_bin = metric.loss_bin
            bin_indices = torch.where(
                (outputs["timesteps"] >= loss_bin[0]) & (outputs["timesteps"] < loss_bin[1])
            )
            metric.update(outputs["prediction"][bin_indices], outputs["target"][bin_indices])
        else:
            # Update metric for all timesteps
            metric.update(outputs["prediction"], outputs["target"])

    # ============================================================
    # INFERENCE GENERATION
    # ============================================================

    def _initialize_latents(
        self,
        batch_size: int,
        image_size: Tuple[int, int],
        init_latents: Optional[torch.Tensor],
        seed: int | list[int] | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Initialize latents for generation.

        Handles three cases:
        1. Pre-provided latents (init_latents)
        2. Per-sample seeds (list of seeds)
        3. Single seed or random initialization

        Args:
            batch_size: Number of samples to generate
            image_size: Target image size (H, W)
            init_latents: Optional pre-initialized latents
            seed: Random seed(s) for reproducibility
            device: Device for tensor allocation

        Returns:
            Initialized latents tensor
        """
        rng_generator = torch.Generator(device=device)

        if init_latents is not None:
            return init_latents
        elif seed and isinstance(seed, list):
            # Per-sample seeds
            return torch.cat(
                [
                    torch.randn(
                        (1, self.vae_channels, image_size[0] // self.vae_scale_factor, image_size[1] // self.vae_scale_factor),
                        device=self.denoiser_device,
                        dtype=self.denoiser_dtype,
                        generator=rng_generator.manual_seed(_seed),
                    )
                    for _seed in seed
                ],
                dim=0,
            )
        else:
            # Single seed or random
            if seed is not None:
                rng_generator = rng_generator.manual_seed(seed)
            return torch.randn(
                (batch_size, self.vae_channels, image_size[0] // self.vae_scale_factor, image_size[1] // self.vae_scale_factor),
                device=self.denoiser_device,
                dtype=self.denoiser_dtype,
                generator=rng_generator,
            )

    def _compute_cfg_guidance(
        self,
        model_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Compute classifier-free guidance from conditional and unconditional predictions.

        Supports two modes:
        - Standard CFG: output = uncond + scale * (cond - uncond)
        - CFG-Zero: Uses alignment-based weighting (arXiv:2503.18886)

        Args:
            model_output: Concatenated [uncond, cond] model predictions
            guidance_scale: Guidance strength

        Returns:
            Guided model output
        """
        model_output_uncond, model_output_text = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_text - model_output_uncond)

    @torch.no_grad()  # type: ignore
    def generate(
        self,
        batch: Dict[BatchKeys, Any],
        image_size: Tuple[int, int],
        num_inference_steps: int | list[int] = 50,
        guidance_scale: float = 7.0,
        seed: int | list[int] | None = None,
        progress_bar: bool = False,
        init_latents: torch.Tensor | None = None,
        denoiser: torch.nn.Module | None = None,
        decode_latents: bool = True,
    ) -> torch.Tensor:
        """Generate images from noise using the diffusion model.

        Args:
            batch: Input batch with prompts/conditions
            image_size: Target image size (H, W)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance strength
            seed: Random seed(s) for reproducibility
            progress_bar: Show progress bar
            init_latents: Optional pre-initialized latents
            denoiser: Optional denoiser override (defaults to EMA if active)
            decode_latents: Decode to image space (vs return latents)

        Returns:
            Generated images or latents
        """

        # 1. Setup
        device = self.vae.device
        denoiser = denoiser or (self.ema_denoiser if self.ema_denoiser.is_active else self.denoiser)
        batch_size = self.get_batch_size_from_batch(batch)

        # 2. Initialize latents
        latents = self._initialize_latents(batch_size, image_size, init_latents, seed, device)

        # 3. Prepare denoiser inputs
        do_cfg = (guidance_scale > 1.0)
        if BatchKeys.IMAGE_LATENT not in batch:
            batch[BatchKeys.IMAGE_LATENT] = latents
        denoiser_kwargs = self.get_denoiser_kwargs(batch=batch, do_cfg=do_cfg)
        denoiser_kwargs.pop(ModelInputs.IMAGE_LATENT)  # Will be updated per step

        # 4. Setup timesteps
        if hasattr(denoiser, "set_timesteps"):
            denoiser.set_timesteps(num_inference_steps, self.inference_scheduler)
        else:
            self.inference_scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.inference_scheduler.init_noise_sigma

        # 5. Denoising loop
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            # Prepare input
            latent_input = torch.cat([latents] * (1 + int(do_cfg)))
            latent_input = self.inference_scheduler.scale_model_input(latent_input, t)

            # Predict model output
            model_output = denoiser(
                image_latent=latent_input,
                timestep=t.repeat(len(latent_input)).to(device=self.denoiser_device, dtype=self.denoiser_dtype),
                **denoiser_kwargs,
            )

            # Apply guidance
            if do_cfg:
                model_output_uncond, model_output_text = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * (model_output_text - model_output_uncond)
            # Scheduler step
            latents = self.inference_scheduler.step(model_output, t, latents, generator=None)
        # 6. Decode if requested
        return self.latent_to_image(latents).detach() if decode_latents else latents

