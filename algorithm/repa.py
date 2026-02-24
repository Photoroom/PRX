"""REPA (https://arxiv.org/abs/2410.06940) and iREPA (https://arxiv.org/pdf/2512.10794) Algorithms for Composer.

These algorithms add an auxiliary loss that aligns intermediate denoiser features
with frozen pretrained encoder (e.g., DINOv3 or DINOv2) representations via a trainable projector.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchvision
from torch import nn, Tensor
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

from dataset.constants import BatchKeys

log = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer(x)
        x = self.silu(x)
        x = self.hidden_layer(x)
        x = self.silu(x)
        x = self.out_layer(x)
        return x


class AttentionProjector(nn.Module):
    """Projector with RoPE positional encoding and linear output for iREPA."""

    def __init__(self, embed_dim: int, out_dim: int, num_heads: int = 4, theta: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.theta = theta

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.output = nn.Linear(embed_dim, out_dim)

    def rope(self, pos: Tensor, dim: int, theta: float) -> Tensor:
        """Compute RoPE embeddings from positions.

        Args:
            pos: Position coordinates [B, N]
            dim: Dimension for RoPE (should be even)
            theta: RoPE theta parameter

        Returns:
            RoPE embeddings [B, N, dim//2, 2, 2]
        """
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)  # (B,N,1) * (1,D//2) -> B, N, D//2
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        out = out.view(*out.shape[:-1], 2, 2)
        return out.float()

    def apply_rope(self, xq: Tensor, freqs_cis: Tensor) -> Tensor:
        """Apply RoPE rotation to query or key tensor.

        Args:
            xq: Query or key tensor [B, H, N, D_h]
            freqs_cis: RoPE embeddings [B, N, D_h//2, 2, 2]

        Returns:
            Rotated tensor [B, H, N, D_h]
        """
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        freqs_cis = freqs_cis.unsqueeze(1)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq)

    def forward(self, x: Tensor, h_coords: Tensor, w_coords: Tensor) -> Tensor:
        """Forward pass with RoPE positional encoding.

        Args:
            x: Input tensor [B, N, D]
            h_coords: Height coordinates [B, N]
            w_coords: Width coordinates [B, N]

        Returns:
            Output tensor [B, N, out_dim]
        """
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D_h]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        rope_h = self.rope(h_coords, self.head_dim // 2, self.theta)  # [B, N, D_h//4, 2, 2]
        rope_w = self.rope(w_coords, self.head_dim // 2, self.theta)  # [B, N, D_h//4, 2, 2]

        freqs_cis = torch.cat([rope_h, rope_w], dim=2)  # [B, N, D_h//2, 2, 2]

        q = self.apply_rope(q, freqs_cis)
        k = self.apply_rope(k, freqs_cis)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # [B, H, N, D_h]

        attn = attn.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        out = self.out_proj(attn)

        return self.output(out)


class DinoWrapper(torch.nn.Module):
    def __init__(self, model: str = "dinov3_vitl16"):
        super().__init__()
        if model == "dinov3_vitl16":
            repo_dir = "facebookresearch/dinov3"
            model_name = "dinov3_vitl16"
            self.patch_size_pixels = 16
        elif model == "dinov2_vitl14_reg":
            repo_dir = "facebookresearch/dinov2"
            model_name = "dinov2_vitl14_reg"
            self.patch_size_pixels = 14
        else:
            raise ValueError(f"Model {model} not supported")
        try:
            self.model = torch.hub.load(repo_dir, model_name).eval().to(torch.bfloat16)
        except (RuntimeError, FileNotFoundError, ConnectionError, ImportError) as e:
            if model == "dinov3_vitl16":
                import os

                torch_home = os.environ.get("TORCH_HOME")
                if torch_home is None:
                    raise ValueError("You need to set the TORCH_HOME environment to use the dinov3 weights")
                repo_dir = os.path.join(torch_home, "hub", "facebookresearch_dinov3_main")
                assert os.path.exists(repo_dir), f"Repository directory {repo_dir} does not exist"
                weights_dir = os.path.join(
                    torch_home, "hub", "checkpoints", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
                )
                assert os.path.exists(weights_dir), f"Weights file {weights_dir} does not exist"
            raise ValueError(f"Model {model} not found. Original error: {e}")
        # Explicitly set all parameters to not require gradients (needed for FSDP/DDP)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_dim = self.model.embed_dim

    def forward(self, img: torch.Tensor, denoiser_downsampling_ratio: int) -> torch.Tensor:
        # resize image
        resize_factor = self.patch_size_pixels / denoiser_downsampling_ratio
        img = torch.nn.functional.interpolate(img, scale_factor=resize_factor)
        # normalize
        img = torchvision.transforms.functional.normalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        return self.model.forward_features(img)["x_norm_patchtokens"]  # [B, num_patches, hidden_dim]


class REPALoss(torch.nn.Module):
    def __init__(
        self,
        denoiser_hidden_dim: int,
        lambda_weight: float = 0.5,
        layer_index: int = 8,
        encoder: str = "dinov3_vitl16",
        compile_encoder: bool = True,
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.layer_index = layer_index
        self.encoder = self.build_encoder(encoder) if encoder else None
        if compile_encoder and self.encoder is not None:
            self.encoder = torch.compile(self.encoder)
        self.projector = self._build_projector(denoiser_hidden_dim)
        self.activations: Optional[torch.Tensor] = None

    def _build_projector(self, denoiser_hidden_dim: int) -> nn.Module:
        """Build the projector module. Override in subclasses for different projector types."""
        return MLP(denoiser_hidden_dim, self.encoder.hidden_dim).to(torch.bfloat16)

    @property
    def mlp(self) -> nn.Module:
        """Alias for projector for backward compatibility."""
        return self.projector

    def build_encoder(self, model: str) -> torch.nn.Module:
        if model.lower() in ["dinov2_vitl14_reg", "dinov3_vitl16"]:
            return DinoWrapper(model)
        else:
            raise ValueError("Only dinov2_vitl14_reg or dinov3_vitl16 encoder is supported.")

    def forward(
        self,
        target_feature: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        tread_original_num_tokens: Optional[int] = None,
        tread_visible_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_feature is None:
            if image is None:
                raise ValueError("Either target_feature or image must be provided")

            # Use original number of tokens if TREAD is active
            if tread_original_num_tokens is not None:
                num_denoiser_tokens = tread_original_num_tokens
            else:
                num_denoiser_tokens = self.activations.shape[-2]

            denoiser_downsampling_ratio = (image.shape[-1] * image.shape[-2] / num_denoiser_tokens) ** 0.5
            with torch.inference_mode():
                target_feature = self.encoder(image.to(torch.bfloat16), denoiser_downsampling_ratio)

        # If TREAD is active, select only visible tokens from target features
        if tread_visible_idx is not None:
            # self.activations: [B, num_visible, hidden_dim]
            # target_feature: [B, num_original, hidden_dim]
            # visible_idx: [B, num_visible]

            hidden_dim = target_feature.shape[-1]
            # Gather visible tokens
            expanded_idx = tread_visible_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
            target_feature = target_feature.gather(1, expanded_idx)
            # Now target_feature: [B, num_visible, hidden_dim] matching self.activations

        loss = torch.nn.functional.cosine_similarity(
            self.projector(self.activations), target_feature, dim=2, eps=1e-8
        ).mean()

        return -self.lambda_weight * loss  # maximise

    def prepare_denoiser(self, denoiser: torch.nn.Module) -> None:
        """Add forward hooks to save intermediate features. Assumes the denoiser has a nn.ModuleList `blocks`"""

        def get_activation() -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
            def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
                self.activations = output

            return hook

        denoiser.blocks[self.layer_index].register_forward_hook(get_activation())


class iREPALoss(REPALoss):
    """iREPA loss with convolutional projection and spatial normalization on encoder features."""

    def __init__(
        self,
        denoiser_hidden_dim: int,
        lambda_weight: float = 0.5,
        layer_index: int = 8,
        encoder: str = "dinov3_vitl16",
        compile_encoder: bool = True,
        use_attention_projector: bool = False,
        num_attention_heads: int = 4,
        rope_theta: float = 10000.0,
    ):
        self.use_attention_projector = use_attention_projector
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        super().__init__(
            denoiser_hidden_dim=denoiser_hidden_dim,
            lambda_weight=lambda_weight,
            layer_index=layer_index,
            encoder=encoder,
            compile_encoder=compile_encoder,
        )

    def _build_projector(self, denoiser_hidden_dim: int) -> nn.Module:
        """Build iREPA projector: AttentionProjector for TREAD compatibility or Conv2d."""
        if self.use_attention_projector:
            return AttentionProjector(
                embed_dim=denoiser_hidden_dim,
                out_dim=self.encoder.hidden_dim,
                num_heads=self.num_attention_heads,
                theta=self.rope_theta,
            ).to(torch.bfloat16)
        else:
            return nn.Conv2d(denoiser_hidden_dim, self.encoder.hidden_dim, kernel_size=3, padding=1).to(
                torch.bfloat16
            )

    def _infer_spatial_dimensions(
        self,
        image: torch.Tensor,
        tread_original_num_tokens: Optional[int],
        current_num_tokens: int,
    ) -> Tuple[int, int]:
        """Infer original spatial dimensions (H, W) in token space.

        Args:
            image: Original image tensor [B, 3, H_image, W_image]
            tread_original_num_tokens: Original number of tokens before TREAD routing
            current_num_tokens: Current number of tokens

        Returns:
            (H_orig, W_orig) in token space
        """
        # Use TREAD's original token count if available, else current count
        num_tokens = tread_original_num_tokens if tread_original_num_tokens is not None else current_num_tokens

        H_image, W_image = image.shape[-2:]
        downsampling_ratio = (H_image * W_image / num_tokens) ** 0.5
        H_latent = int(H_image / downsampling_ratio)
        W_latent = int(W_image / downsampling_ratio)

        return H_latent, W_latent

    def forward(
        self,
        target_feature: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        tread_original_num_tokens: Optional[int] = None,
        tread_visible_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("image is required for iREPALoss to infer spatial dimensions")

        if target_feature is None:
            # Use original number of tokens if TREAD is active
            if tread_original_num_tokens is not None:
                num_denoiser_tokens = tread_original_num_tokens
            else:
                num_denoiser_tokens = self.activations.shape[-2]

            denoiser_downsampling_ratio = (image.shape[-1] * image.shape[-2] / num_denoiser_tokens) ** 0.5
            with torch.inference_mode():
                target_feature = self.encoder(image.to(torch.bfloat16), denoiser_downsampling_ratio)

        # Spatial normalization on encoder features [B, T, D]
        # Note: dim=2 corresponds to the feature dimension D
        gamma = target_feature.mean(dim=2, keepdim=True)
        target_feature = target_feature - gamma * target_feature.mean(dim=1, keepdim=True)
        target_feature = target_feature / (target_feature.std(dim=1, keepdim=True) + 1e-6)

        # If TREAD is active, select only visible tokens from target features
        if tread_visible_idx is not None:
            # self.activations: [B, num_visible, hidden_dim]
            # target_feature: [B, num_original, hidden_dim]
            # visible_idx: [B, num_visible]

            hidden_dim = target_feature.shape[-1]
            # Gather visible tokens
            expanded_idx = tread_visible_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
            target_feature = target_feature.gather(1, expanded_idx)
            # Now target_feature: [B, num_visible, hidden_dim] matching self.activations

        # Hybrid projection: Conv2d when all tokens are visible case, attention for TREAD case
        B, T, D = self.activations.shape
        H_orig, W_orig = self._infer_spatial_dimensions(image, tread_original_num_tokens, T)

        if tread_visible_idx is not None and not self.use_attention_projector:
            raise ValueError("TREAD is active but attention projector is not used. This is not supported.")

        if self.use_attention_projector:
            # Attention path: TREAD-compatible
            if tread_visible_idx is not None:
                # Convert flat indices to 2D coordinates for RoPE
                h_coords = (tread_visible_idx // W_orig).float()  # [B, num_visible]
                w_coords = (tread_visible_idx % W_orig).float()  # [B, num_visible]

            else:
                # No TREAD: use full grid coordinates
                h_coords = (
                    torch.arange(H_orig, device=self.activations.device)
                    .repeat_interleave(W_orig)
                    .unsqueeze(0)
                    .expand(B, -1)
                    .float()
                )
                w_coords = (
                    torch.arange(W_orig, device=self.activations.device)
                    .repeat(H_orig)
                    .unsqueeze(0)
                    .expand(B, -1)
                    .float()
                )

            projected = self.projector(self.activations, h_coords=h_coords, w_coords=w_coords)

        else:
            # Conv2d path: spatial inductive bias
            # Reshape to spatial format [B, T, D] -> [B, D, H, W]
            activations_spatial = self.activations.permute(0, 2, 1).reshape(B, D, H_orig, W_orig)
            # Apply conv projection
            projected = self.projector(activations_spatial)
            # Reshape back to [B, T, D_out]
            projected = projected.reshape(B, self.encoder.hidden_dim, -1).permute(0, 2, 1)

        loss = torch.nn.functional.cosine_similarity(projected, target_feature, dim=2, eps=1e-8).mean()

        return -self.lambda_weight * loss  # maximise

class REPA(Algorithm):
    """
    Representation Alignment (REPA) Algorithm.

    Adds an auxiliary loss that aligns intermediate denoiser features with
    frozen pretrained encoder (e.g., DINOv3) representations via a trainable MLP.

    The algorithm:
    1. add_new_pipeline_modules() creates REPALoss module before optimizer creation
    2. Optimizer automatically includes MLP parameters (no manual add_param_group needed)
    3. During INIT event: registers forward hooks and wraps model.loss() method
    4. Module stored on state.model for TREAD discovery

    Args:
        lambda_weight: Weight for REPA loss term (default: 0.5)
        layer_index: Which denoiser block to capture activations from (default: 8)
        encoder: Encoder model name, e.g., "dinov3_vitl16" or "dinov2_vitl14_reg" (default: "dinov3_vitl16")
        compile_encoder: Whether to torch.compile the encoder (default: True)

    Example:
        In your YAML config:

        algorithms:
          repa:
            _target_: algorithm.repa.REPA
            lambda_weight: 0.5
            layer_index: 7
            encoder: "dinov3_vitl16"
            compile_encoder: true
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        layer_index: int = 8,
        encoder: str = "dinov3_vitl16",
        compile_encoder: bool = True,
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.layer_index = layer_index
        self.encoder = encoder
        self.compile_encoder = compile_encoder
        self.repa_loss: Optional[REPALoss] = None
        self._hooks_registered = False
        self._modules_added = False

    def _get_log_prefix(self) -> str:
        """Return the logging prefix for this algorithm ('repa' or 'irepa')."""
        return "repa"

    def _get_init_message(self) -> str:
        """Return the initialization message to print."""
        return (
            f" > REPA: Initializing hooks and loss wrapper (lambda_weight={self.lambda_weight}, "
            f"layer_index={self.layer_index}, encoder={self.encoder})"
        )

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters to log."""
        prefix = self._get_log_prefix()
        return {
            f"{prefix}/lambda_weight": self.lambda_weight,
            f"{prefix}/layer_index": self.layer_index,
            f"{prefix}/encoder": self.encoder,
            f"{prefix}/compile_encoder": self.compile_encoder,
        }

    def _build_loss_module(self, denoiser_hidden_dim: int) -> REPALoss:
        """Build the loss module. Override in subclasses for different loss types."""
        return REPALoss(
            denoiser_hidden_dim=denoiser_hidden_dim,
            lambda_weight=self.lambda_weight,
            layer_index=self.layer_index,
            encoder=self.encoder,
            compile_encoder=self.compile_encoder,
        )

    def add_new_pipeline_modules(self, model: torch.nn.Module) -> None:
        """
        Add projector module to the model before optimizer creation.

        This method is called from train.py before the optimizer is instantiated,
        ensuring that the projector parameters are automatically included in the optimizer.

        Args:
            model: The pipeline model (will be wrapped by DDP/FSDP later)
        """
        prefix = self._get_log_prefix().upper()
        if self._modules_added:
            log.info(f" > {prefix}: Modules already added, skipping")
            return

        # Resolve denoiser from model
        denoiser = model.denoiser
        denoiser_hidden_dim = denoiser.hidden_size

        log.info(f" > {prefix}: Adding projector module (lambda_weight={self.lambda_weight}, "
              f"layer_index={self.layer_index}, encoder={self.encoder})")

        # Create loss module with projector
        self.repa_loss = self._build_loss_module(denoiser_hidden_dim)

        # Attach to model (not denoiser) to ensure it's accessible for TREAD
        model.repa_loss = self.repa_loss

        # Move to bfloat16 (loss module uses bfloat16 internally)
        self.repa_loss = self.repa_loss.to(dtype=torch.bfloat16)

        num_params = sum(p.numel() for p in self.repa_loss.projector.parameters())
        log.info(f" > {prefix}: Added projector with {num_params:,} parameters")

        self._modules_added = True

    def match(self, event: Event, state: State) -> bool:
        """Match only INIT event for setup."""
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """
        Apply algorithm logic during INIT event:
        1. Retrieve pre-existing REPALoss module (added via add_new_pipeline_modules)
        2. Register forward hooks on denoiser
        3. Wrap model.loss() method

        Note: The REPALoss module and MLP are created in add_new_pipeline_modules(),
        which is called from train.py before optimizer creation. This ensures the MLP
        parameters are automatically included in the optimizer without manual add_param_group().
        """
        if event == Event.INIT:
            # Resolve denoiser (handle DDP/FSDP wrapping)
            denoiser = self._resolve_denoiser(state.model)

            # Retrieve the pre-existing REPALoss module
            # It should have been added by add_new_pipeline_modules() before FSDP wrapping
            if hasattr(state.model, 'module'):
                # Model is wrapped (DDP/FSDP)
                model_unwrapped = state.model.module
            else:
                model_unwrapped = state.model

            prefix = self._get_log_prefix().upper()
            if not hasattr(model_unwrapped, 'repa_loss'):
                raise RuntimeError(
                    f"{prefix} module not found on model. Did you call add_new_pipeline_modules() "
                    "in train.py before creating the optimizer?"
                )

            self.repa_loss = model_unwrapped.repa_loss

            log.info(self._get_init_message())

            # Move to same device as denoiser (dtype already set to bfloat16 in add_new_pipeline_modules)
            device = next(denoiser.parameters()).device
            self.repa_loss = self.repa_loss.to(device=device)

            # Register forward hook on denoiser to capture activations
            if not self._hooks_registered:
                self.repa_loss.prepare_denoiser(denoiser)
                self._hooks_registered = True

            # Wrap the model's loss method to inject REPA loss
            self._wrap_loss_method(state)

            # Log hyperparameters
            logger.log_hyperparameters(self._get_hyperparameters())

    def _resolve_denoiser(self, model: torch.nn.Module) -> torch.nn.Module:
        """Resolve denoiser from (possibly wrapped) model."""
        # Handle DDP/FSDP wrapping
        if hasattr(model, 'module'):
            model = model.module
        return model.denoiser

    def _wrap_loss_method(self, state: State) -> None:
        """
        Wrap the model's loss() method to inject REPA loss computation.

        The original loss method computes MSE + contrastive loss.
        Our wrapper adds REPA auxiliary loss on top.
        """
        original_loss_fn = state.model.loss
        repa_loss_module = self.repa_loss

        def augmented_loss(outputs: Dict[str, torch.Tensor], batch: Dict[BatchKeys, Any]) -> torch.Tensor:
            # Compute base loss (MSE + contrastive if enabled)
            base_loss = original_loss_fn(outputs, batch)

            # Compute REPA auxiliary loss
            # Note: TREAD will inject routing metadata via pre-hook if active
            repa_loss_value = repa_loss_module(
                target_feature=batch.get(BatchKeys.TARGET_REPRESENTATION, None),
                image=batch.get(BatchKeys.IMAGE, None),
            )

            # Log REPA loss separately
            state.model.logger.log_metrics({"loss/train/repa": repa_loss_value.detach().cpu()})

            # Return combined loss
            return base_loss + repa_loss_value

        # Replace the loss method
        state.model.loss = augmented_loss
        log.info(" > REPA: Wrapped model.loss() method to inject REPA loss computation")


class iREPA(REPA):
    """
    Improved Representation Alignment (iREPA) Algorithm.

    Extends REPA with:
    - Spatial normalization on encoder features
    - Choice between Conv2d or Attention projector
    - TREAD-compatible attention projector with RoPE positional encoding

    Args:
        lambda_weight: Weight for iREPA loss term (default: 0.5)
        layer_index: Which denoiser block to capture activations from (default: 8)
        encoder: Encoder model name, e.g., "dinov3_vitl16" or "dinov2_vitl14_reg" (default: "dinov3_vitl16")
        compile_encoder: Whether to torch.compile the encoder (default: True)
        use_attention_projector: Use attention projector instead of Conv2d (default: True)
        num_attention_heads: Number of attention heads for attention projector (default: 4)
        rope_theta: RoPE theta parameter (default: 10000.0)

    Example:
        In your YAML config:

        algorithms:
          irepa:
            _target_: algorithm.repa.iREPA
            lambda_weight: 0.5
            layer_index: 7
            encoder: "dinov3_vitl16"
            use_attention_projector: true
            num_attention_heads: 4
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        layer_index: int = 8,
        encoder: str = "dinov3_vitl16",
        compile_encoder: bool = True,
        use_attention_projector: bool = True,
        num_attention_heads: int = 4,
        rope_theta: float = 10000.0,
    ):
        # Call parent __init__ with base REPA parameters
        super().__init__(
            lambda_weight=lambda_weight,
            layer_index=layer_index,
            encoder=encoder,
            compile_encoder=compile_encoder,
        )
        # Add iREPA-specific parameters
        self.use_attention_projector = use_attention_projector
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta

    def _get_log_prefix(self) -> str:
        """Return the logging prefix for iREPA."""
        return "irepa"

    def _get_init_message(self) -> str:
        """Return the initialization message for iREPA."""
        projector_type = "Attention" if self.use_attention_projector else "Conv2d"
        return (
            f" > iREPA: Initializing hooks and loss wrapper (lambda_weight={self.lambda_weight}, "
            f"layer_index={self.layer_index}, encoder={self.encoder}, projector={projector_type})"
        )

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters to log, including iREPA-specific params."""
        # Get base hyperparameters from parent
        hyperparams = super()._get_hyperparameters()
        # Add iREPA-specific hyperparameters
        prefix = self._get_log_prefix()
        hyperparams.update({
            f"{prefix}/use_attention_projector": self.use_attention_projector,
            f"{prefix}/num_attention_heads": self.num_attention_heads,
            f"{prefix}/rope_theta": self.rope_theta,
        })
        return hyperparams

    def _build_loss_module(self, denoiser_hidden_dim: int) -> iREPALoss:
        """Build iREPALoss module with the appropriate projector type."""
        return iREPALoss(
            denoiser_hidden_dim=denoiser_hidden_dim,
            lambda_weight=self.lambda_weight,
            layer_index=self.layer_index,
            encoder=self.encoder,
            compile_encoder=self.compile_encoder,
            use_attention_projector=self.use_attention_projector,
            num_attention_heads=self.num_attention_heads,
            rope_theta=self.rope_theta,
        )

