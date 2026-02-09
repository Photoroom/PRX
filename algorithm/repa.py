"""REPA (Representation Alignment) Algorithm for Composer (https://arxiv.org/abs/2410.06940).

This algorithm adds an auxiliary loss that aligns intermediate denoiser features
with frozen pretrained encoder (e.g., DINOv3) representations via a trainable MLP.
"""

from typing import Any, Callable, Dict, Optional

import torch
import torchvision
from torch import nn, Tensor
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

from dataset.constants import BatchKeys


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

    def forward(self, img: torch.Tensor, denoiser_downsampling_ratio: int) -> Dict[str, torch.Tensor]:
        # resize image
        resize_factor = self.patch_size_pixels / denoiser_downsampling_ratio
        img = torch.nn.functional.interpolate(img, scale_factor=resize_factor)
        # normalize
        img = torchvision.transforms.functional.normalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        features = self.model.forward_features(img)
        return {
            "cls_token": features["x_norm_clstoken"],      # [B, hidden_dim]
            "patch_tokens": features["x_norm_patchtokens"],  # [B, num_patches, hidden_dim]
        }


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
        # Shared MLP - convert to same dtype as encoder
        self.mlp = MLP(denoiser_hidden_dim, self.encoder.hidden_dim).to(torch.bfloat16)
        self.activations: Optional[torch.Tensor] = None

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
                features = self.encoder(image.to(torch.bfloat16), denoiser_downsampling_ratio)
                target_feature = features["patch_tokens"]  # Use patch tokens for REPA

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

        if self.activations is None:
            raise RuntimeError("Forward hook not yet invoked. Activations not captured.")

        loss = 0.0
        loss += torch.nn.functional.cosine_similarity(
            self.mlp(self.activations), target_feature, dim=2, eps=1e-8
        ).mean()

        return -self.lambda_weight * loss  # maximise

    def prepare_denoiser(self, denoiser: torch.nn.Module) -> None:
        """Add forward hooks to save intermediate features. Assumes the denoiser has a nn.ModuleList `blocks`"""

        def get_activation() -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
            def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
                self.activations = output

            return hook

        denoiser.blocks[self.layer_index].register_forward_hook(get_activation())

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

    def add_new_pipeline_modules(self, model: torch.nn.Module) -> None:
        """
        Add REPA MLP module to the model before optimizer creation.

        This method is called from train.py before the optimizer is instantiated,
        ensuring that the MLP parameters are automatically included in the optimizer.

        Args:
            model: The pipeline model (will be wrapped by DDP/FSDP later)
        """
        if self._modules_added:
            print(" > REPA: Modules already added, skipping")
            return

        # Resolve denoiser from model
        denoiser = model.denoiser
        denoiser_hidden_dim = denoiser.hidden_size

        print(f" > REPA: Adding MLP module (lambda_weight={self.lambda_weight}, "
              f"layer_index={self.layer_index}, encoder={self.encoder})")

        # Create REPALoss module with MLP
        self.repa_loss = REPALoss(
            denoiser_hidden_dim=denoiser_hidden_dim,
            lambda_weight=self.lambda_weight,
            layer_index=self.layer_index,
            encoder=self.encoder,
            compile_encoder=self.compile_encoder,
        )

        # Attach to model (not denoiser) to ensure it's accessible for TREAD
        # Use the same attachment point as the original implementation
        model.repa_loss = self.repa_loss

        # Move to bfloat16 (REPALoss uses bfloat16 internally)
        self.repa_loss = self.repa_loss.to(dtype=torch.bfloat16)

        num_params = sum(p.numel() for p in self.repa_loss.mlp.parameters())
        print(f" > REPA: Added MLP module with {num_params:,} parameters")

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

            if not hasattr(model_unwrapped, 'repa_loss'):
                raise RuntimeError(
                    "REPA module not found on model. Did you call add_new_pipeline_modules() "
                    "in train.py before creating the optimizer?"
                )

            self.repa_loss = model_unwrapped.repa_loss

            print(f" > REPA: Initializing hooks and loss wrapper (lambda_weight={self.lambda_weight}, "
                  f"layer_index={self.layer_index}, encoder={self.encoder})")

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
            logger.log_hyperparameters({
                "repa/lambda_weight": self.lambda_weight,
                "repa/layer_index": self.layer_index,
                "repa/encoder": self.encoder,
                "repa/compile_encoder": self.compile_encoder,
            })

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
        print(" > REPA: Wrapped model.loss() method to inject REPA loss computation")

