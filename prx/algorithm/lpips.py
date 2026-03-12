"""LPIPS Perceptual Loss Algorithm for Composer.

Adds an auxiliary LPIPS loss between the denoiser's raw x0 prediction and
the ground-truth clean latents (both RGB images in [-1, 1] with IdentityVAE).

The algorithm uses a forward hook on the denoiser to capture its raw output
before the x-to-v conversion, then computes LPIPS in the wrapped loss method.
"""

import logging
from typing import Any, Literal

import lpips
import torch
import torch.nn.functional as F
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

logger = logging.getLogger(__name__)


class LPIPS(Algorithm):
    """Adds LPIPS perceptual loss between x0_pred and x0_gt during training.

    Args:
        lpips_weight: Scalar weight for the LPIPS loss term.
        lpips_net: Backbone for LPIPS (\"vgg\" or \"alex\").
        resize_factor: Spatial downscale factor applied before LPIPS (ignored when crop_size > 0).
        crop_size: If > 0, randomly crop x0_pred and x0_gt to this size before LPIPS.
            Takes priority over resize_factor.
        t_threshold: Only apply LPIPS for timesteps below this value (1.0 = all samples).
    """

    def __init__(
        self,
        lpips_weight: float = 1.0,
        lpips_net: Literal["vgg", "alex"] = "vgg",
        resize_factor: float = 1.0,
        crop_size: int = 0,
        t_threshold: float = 1.0,
    ):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.lpips_net = lpips_net
        self.resize_factor = resize_factor
        self.crop_size = crop_size
        self.t_threshold = t_threshold
        if crop_size > 0 and resize_factor != 1.0:
            logger.warning(
                f"LPIPS: both crop_size={crop_size} and resize_factor={resize_factor} are set. "
                "Only cropping will be applied; resize_factor is ignored."
            )
        self.lpips_fn: lpips.LPIPS | None = None
        self._stashed_x0_pred: torch.Tensor | None = None
        self._modules_added = False

    def add_new_pipeline_modules(self, model: torch.nn.Module) -> None:
        """Called from train.py before optimizer creation.

        Creates a frozen LPIPS network and attaches it to the model
        so FSDP/DDP can see it (all params are frozen).
        """
        if self._modules_added:
            return
        self.lpips_fn = lpips.LPIPS(net=self.lpips_net).eval().requires_grad_(False).to(torch.bfloat16)
        self.lpips_fn = torch.compile(self.lpips_fn)
        model.lpips_fn = self.lpips_fn
        self._modules_added = True
        logger.info(f"LPIPS: Added frozen {self.lpips_net} network in bf16 (weight={self.lpips_weight})")

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, composer_logger: Logger) -> None:
        model = state.model.module if hasattr(state.model, "module") else state.model

        # Guard: LPIPS requires x-prediction mode (denoiser outputs x0, not velocity)
        from prx.pipeline.composer_pipeline import PredictionType
        pred_type = model.noise_scheduler.config.prediction_type
        if pred_type != PredictionType.X_PREDICTION_FLOW_MATCHING:
            raise ValueError(
                f"LPIPS requires prediction_type='x_prediction_flow_matching' "
                f"(denoiser must output x0), but got '{pred_type}'"
            )

        self.lpips_fn = model.lpips_fn
        device = next(model.denoiser.parameters()).device
        self.lpips_fn = self.lpips_fn.to(device=device)

        # Hook: stash raw denoiser output (x0_pred before v-conversion)
        algo = self

        def hook_fn(module: torch.nn.Module, input: Any, output: torch.Tensor) -> None:
            algo._stashed_x0_pred = output

        self._hook_handle = model.denoiser.register_forward_hook(hook_fn)

        # Wrap loss
        self._wrap_loss_method(state)

        composer_logger.log_hyperparameters(
            {
                "lpips/weight": self.lpips_weight,
                "lpips/net": self.lpips_net,
                "lpips/resize_factor": self.resize_factor,
                "lpips/crop_size": self.crop_size,
                "lpips/t_threshold": self.t_threshold,
            }
        )
        logger.info("LPIPS: Registered denoiser hook and wrapped loss()")

    def _wrap_loss_method(self, state: State) -> None:
        original_loss_fn = state.model.loss
        lpips_fn = self.lpips_fn
        lpips_weight = self.lpips_weight
        resize_factor = self.resize_factor
        crop_size = self.crop_size
        t_threshold = self.t_threshold
        algo = self
        composer_model = state.model

        def augmented_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
            base_loss = original_loss_fn(outputs, batch)

            t = outputs["timesteps"]  # [B]

            # Per-sample mask: only apply LPIPS when t < threshold (less noisy samples)
            mask = (t < t_threshold).float()

            x0_pred = algo._stashed_x0_pred
            if x0_pred is None:
                raise RuntimeError("Forward hook not yet invoked. _stashed_x0_pred not captured.")
            algo._stashed_x0_pred = None  # release reference

            # Recover x0_gt from v-space target: x0 = noised_latents - t * v
            t_expanded = t.view(-1, 1, 1, 1).clamp(min=0.05)
            x0_gt = outputs["noised_latents"] - t_expanded * outputs["target"]
            x0_gt_detached = x0_gt.detach()

            # Spatial preprocessing before LPIPS: random crop or resize
            if crop_size > 0:
                _, _, h, w = x0_pred.shape
                if h > crop_size and w > crop_size:
                    top = torch.randint(0, h - crop_size, (1,)).item()
                    left = torch.randint(0, w - crop_size, (1,)).item()
                else:
                    top, left = 0, 0
                x0_pred = x0_pred[:, :, top : top + crop_size, left : left + crop_size]
                x0_gt_detached = x0_gt_detached[:, :, top : top + crop_size, left : left + crop_size]
            elif resize_factor != 1.0:
                x0_pred = F.interpolate(x0_pred, scale_factor=resize_factor, mode="bilinear", align_corners=False)
                x0_gt_detached = F.interpolate(
                    x0_gt_detached, scale_factor=resize_factor, mode="bilinear", align_corners=False
                )

            # Always run lpips_fn to keep FSDP2 collectives consistent across ranks
            lpips_per_sample = lpips_fn(x0_pred.to(torch.bfloat16), x0_gt_detached.to(torch.bfloat16)).view(-1)  # [B]
            mask_count = mask.sum().clamp(min=1)
            lpips_val = (lpips_per_sample * mask).sum() / mask_count
            composer_model.logger.log_metrics({"loss/train/lpips": lpips_val.detach().cpu()})

            return base_loss + lpips_weight * lpips_val

        state.model.loss = augmented_loss
