"""P-DINO (Perceptual DINO) Loss Algorithm for Composer.

Adds an auxiliary loss that aligns patch-level DINO features between the denoiser's
raw x0 prediction and the ground-truth clean latents.

Loss: L_P-DINO = (1/|P|) * sum_{p in P} (1 - cos(f^p_DINO(x0_pred), f^p_DINO(x0_gt)))
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import torchvision
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

logger = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

SUPPORTED_MODELS = {
    "dinov2_vitb14_reg": ("facebookresearch/dinov2", "dinov2_vitb14_reg", 14),
    "dinov2_vitl14_reg": ("facebookresearch/dinov2", "dinov2_vitl14_reg", 14),
    "dinov3_vitl16": ("facebookresearch/dinov3", "dinov3_vitl16", 16),
}


class PDINOEncoder(torch.nn.Module):
    """Lightweight DINOv2 wrapper for P-DINO perceptual loss."""

    def __init__(self, model: str = "dinov2_vitb14_reg"):
        super().__init__()
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not supported. Choose from: {list(SUPPORTED_MODELS.keys())}")
        repo_dir, model_name, self.patch_size_pixels = SUPPORTED_MODELS[model]
        self.model = torch.hub.load(repo_dir, model_name).eval().to(torch.bfloat16)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_dim = self.model.embed_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Inputs come from x0 prediction/target in [-1, 1]. DINO expects ImageNet-normalized [0, 1].
        img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
        img = torchvision.transforms.functional.normalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        return self.model.forward_features(img)["x_norm_patchtokens"]


class PerceptualDINO(Algorithm):
    """Adds P-DINO perceptual loss between x0_pred and x0_gt during training.

    Args:
        pdino_weight: Scalar weight for the P-DINO loss term.
        encoder: DINO encoder model ("dinov2_vitb14_reg").
        compile_encoder: Whether to torch.compile the encoder.
        t_threshold: Only apply P-DINO loss when timestep t < threshold.
        resize_resolution: Resize images to this resolution before DINO (default: 256 for memory efficiency).
        crop_size: If > 0, randomly crop x0_pred and x0_gt to this size before DINO.
            Takes priority over resize_resolution.
    """

    def __init__(
        self,
        pdino_weight: float = 1.0,
        encoder: str = "dinov2_vitb14_reg",
        compile_encoder: bool = True,
        t_threshold: float = 0.0,
        resize_resolution: int = 256,
        crop_size: int = 0,
    ):
        super().__init__()
        self.pdino_weight = pdino_weight
        self.encoder_name = encoder
        self.compile_encoder = compile_encoder
        self.t_threshold = t_threshold
        self.resize_resolution = resize_resolution
        self.crop_size = crop_size
        self.encoder: Optional[PDINOEncoder] = None
        self._stashed_x0_pred: Optional[torch.Tensor] = None
        self._modules_added = False

    def add_new_pipeline_modules(self, model: torch.nn.Module) -> None:
        """Called from train.py before optimizer creation.

        Creates a new frozen DINO encoder.
        """
        if self._modules_added:
            return

        self.encoder = PDINOEncoder(self.encoder_name).eval().requires_grad_(False)
        if self.compile_encoder:
            self.encoder = torch.compile(self.encoder)
        model.pdino_encoder = self.encoder
        self._modules_added = True
        compiled_str = " (compiled)" if self.compile_encoder else ""
        logger.info(f"P-DINO: Created frozen {self.encoder_name} encoder{compiled_str} (weight={self.pdino_weight})")

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        model = state.model.module if hasattr(state.model, "module") else state.model

        # Guard: P-DINO requires x-prediction mode (denoiser outputs x0, not velocity)
        from pipeline.pipeline import PredictionType
        pred_type = model.noise_scheduler.config.prediction_type
        if pred_type != PredictionType.X_PREDICTION_FLOW_MATCHING:
            raise ValueError(
                f"P-DINO requires prediction_type='x_prediction_flow_matching' "
                f"(denoiser must output x0), but got '{pred_type}'"
            )

        self.encoder = model.pdino_encoder
        device = next(model.denoiser.parameters()).device
        self.encoder = self.encoder.to(device=device)

        # Hook: stash raw denoiser output (x0_pred before v-conversion)
        algo = self

        def hook_fn(module: torch.nn.Module, input: Any, output: torch.Tensor) -> None:
            algo._stashed_x0_pred = output

        model.denoiser.register_forward_hook(hook_fn)

        # Wrap loss
        self._wrap_loss_method(state)

        logger.log_hyperparameters(
            {
                "pdino/weight": self.pdino_weight,
                "pdino/encoder": self.encoder_name,
                "pdino/t_threshold": self.t_threshold,
                "pdino/resize_resolution": self.resize_resolution,
                "pdino/crop_size": self.crop_size,
            }
        )
        logger.info("P-DINO: Registered denoiser hook and wrapped loss()")

    def _wrap_loss_method(self, state: State) -> None:
        original_loss_fn = state.model.loss
        composer_model = state.model
        encoder = self.encoder
        pdino_weight = self.pdino_weight
        t_threshold = self.t_threshold
        resize_resolution = self.resize_resolution
        crop_size = self.crop_size
        algo = self

        def augmented_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
            base_loss = original_loss_fn(outputs, batch)

            t = outputs["timesteps"]  # [B]

            # Per-sample mask: only apply P-DINO when t < threshold (less noisy samples)
            mask = (t < t_threshold).float()

            x0_pred = algo._stashed_x0_pred  # raw denoiser output (x0 in x-pred mode)
            algo._stashed_x0_pred = None  # release reference
            if x0_pred is None:
                return base_loss

            # Recover x0_gt from v-space target: x0 = noised_latents - t * v
            t_expanded = t.view(-1, 1, 1, 1).clamp(min=0.05)
            x0_gt = outputs["noised_latents"] - t_expanded * outputs["target"]
            x0_gt_detached = x0_gt.detach()

            # Spatial preprocessing before DINO: random crop or resize
            if crop_size > 0:
                _, _, h, w = x0_pred.shape
                if h > crop_size and w > crop_size:
                    top = torch.randint(0, h - crop_size, (1,)).item()
                    left = torch.randint(0, w - crop_size, (1,)).item()
                else:
                    top, left = 0, 0
                x0_pred_resized = x0_pred[:, :, top : top + crop_size, left : left + crop_size]
                x0_gt_resized = x0_gt_detached[:, :, top : top + crop_size, left : left + crop_size]
            else:
                x0_pred_resized = F.interpolate(
                    x0_pred, size=(resize_resolution, resize_resolution), mode="bilinear", align_corners=False
                )
                x0_gt_resized = F.interpolate(
                    x0_gt_detached, size=(resize_resolution, resize_resolution), mode="bilinear", align_corners=False
                )

            # Always compute DINO features to keep FSDP2 collectives consistent across ranks
            features_pred = encoder(x0_pred_resized.to(torch.bfloat16))  # [B, P, D]
            with torch.no_grad():
                features_gt = encoder(x0_gt_resized.to(torch.bfloat16))  # [B, P, D]

            # L_P-DINO = (1/|P|) * sum_{p in P} (1 - cos(f^p_pred, f^p_gt))
            # cosine_similarity with dim=-1 gives [B, P], then mean over patches
            cos_sim = F.cosine_similarity(features_pred.float(), features_gt.float(), dim=-1)  # [B, P]
            pdino_per_sample = (1 - cos_sim).mean(dim=-1)  # [B]

            # Apply per-sample mask
            mask_count = mask.sum().clamp(min=1)
            pdino_val = (pdino_per_sample * mask).sum() / mask_count
            composer_model.logger.log_metrics({"loss/train/pdino": pdino_val.detach().cpu()})

            return base_loss + pdino_weight * pdino_val

        state.model.loss = augmented_loss
