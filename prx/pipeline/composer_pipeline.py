from typing import Any

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric

from prx.dataset.constants import BatchKeys
from prx.models.text_tower import TextTower
from prx.schedulers.scheduler import BaseScheduler

# Re-export types from fm_pipeline
from .fm_pipeline import (
    EMAModel,
    FMPipeline,
    ForwardOutput,
    ImageSize,
    ModelInputs,
    PredictionType,
)


class ComposerFMPipeline(FMPipeline, ComposerModel):
    """Composer-integrated flow-matching pipeline.

    Extends FMPipeline with Composer-specific functionality:
    loss logging, evaluation, and metrics tracking.

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

    def __init__(
        self,
        denoiser: torch.nn.Module,
        vae: torch.nn.Module,
        text_tower: TextTower,
        noise_scheduler: BaseScheduler,
        inference_noise_scheduler: BaseScheduler,
        p_drop_caption: float = 0.1,
        train_metrics: list[str] | None = None,
        val_metrics: list[str] | None = None,
        val_seed: int = 1138,
        val_guidance_scales: list[float] | None = None,
        loss_bins: list[tuple[int, int]] | None = None,
        negative_prompt: str = "",
        noise_scale: float = 1.0,
    ):
        super().__init__(
            denoiser=denoiser,
            vae=vae,
            text_tower=text_tower,
            noise_scheduler=noise_scheduler,
            inference_noise_scheduler=inference_noise_scheduler,
            p_drop_caption=p_drop_caption,
            val_seed=val_seed,
            val_guidance_scales=val_guidance_scales,
            negative_prompt=negative_prompt,
            noise_scale=noise_scale,
        )

        # Metrics (Composer-specific)
        train_metrics = train_metrics or [MeanSquaredError()]
        loss_bins = loss_bins or [(0, 1)]

        self.train_metrics = train_metrics

        self.val_metrics = {}
        self.val_metrics["MeanSquaredError"] = MeanSquaredError()

        for bin in loss_bins:
            new_metric = MeanSquaredError()
            new_metric.loss_bin = bin  # type: ignore
            self.val_metrics[f"MeanSquaredError-bin-{bin[0]}-to-{bin[1]}".replace(".", "p")] = new_metric

    # ============================================================
    # COMPOSER INTERFACE: Loss, Eval, Metrics
    # ============================================================

    def loss(self, outputs: dict[str, torch.Tensor], batch: dict[BatchKeys, Any]) -> torch.Tensor:
        """Loss between denoiser output and added noise, typically mse."""
        loss = F.mse_loss(outputs["prediction"], outputs["target"].clone(), reduction="none")
        loss = loss.mean()
        self.logger.log_metrics({"loss/train/mse": loss.detach().cpu()})

        return loss

    def eval_forward(self, batch: dict[BatchKeys, Any], outputs: ForwardOutput | None = None) -> ForwardOutput:
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

    def get_metrics(self, is_train: bool = False) -> dict[str, Metric]:
        if is_train:
            return {metric.__class__.__name__: metric for metric in self.train_metrics}
        return self.val_metrics

    def update_metric(self, batch: dict[BatchKeys, Any], outputs: dict[str, Any], metric: Metric) -> None:
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

