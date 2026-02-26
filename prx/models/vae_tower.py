from typing import Dict

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, AutoencoderKL
from transformers.modeling_utils import ModuleUtilsMixin


class IdentityVAE(nn.Module):
    """Identity VAE that passes inputs through unchanged (for pixel-space diffusion).

    Args:
        channels: Number of input/output channels. Defaults to 3.
        torch_dtype: Data type for the device tracker buffer. Defaults to torch.float32.

    Attributes:
        scale_factor: Spatial scale factor (always 1 for identity).
        spatial_compression_ratio: Spatial compression ratio (always 1 for identity).
        config: Configuration object with scaling_factor, shift_factor, and latent_channels.
        encoder: Identity encoder module (passes inputs unchanged).
        decoder: Identity decoder module (passes inputs unchanged).
        _device_tracker: Buffer tensor used for device/dtype detection.
    """

    def __init__(self, channels: int = 3, torch_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.scale_factor = 1
        self.spatial_compression_ratio = 1
        self.config = type(
            "Config",
            (),
            {
                "scaling_factor": 1.0,
                "shift_factor": 0.0,
                "latent_channels": channels,
            },
        )()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.register_buffer("_device_tracker", torch.zeros(1, dtype=torch_dtype))

    def parameters(self, recurse: bool = True):
        """Yield the device tracker as a parameter for device/dtype detection."""
        yield self._device_tracker

    @property
    def dtype(self) -> torch.dtype:
        return self._device_tracker.dtype

    def encode(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode image - returns identity."""
        return {"latent": image}

    def decode(self, latent: torch.Tensor):
        """Decode latent - returns identity."""

        class DecodeOutput:
            def __init__(self, sample: torch.Tensor) -> None:
                self.sample = sample

        return DecodeOutput(latent)


class VaeTower(torch.nn.Module, ModuleUtilsMixin):
    """VAE Tower wrapper that provides a consistent interface for different VAE types."""

    def __init__(
        self,
        model_name: str = "flux-dev",
        model_class: str = "AutoencoderKL",
        default_channels: int = 16,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.torch_dtype = torch_dtype
        self.model_class = model_class
        self.default_channels = default_channels

        self.vae = self.create_model(model_name)

        self._set_properties()
        self.eval()

    def _set_properties(self) -> None:
        """Set consistent properties regardless of VAE type."""
        # Spatial compression ratio / scale factor
        if hasattr(self.vae, "spatial_compression_ratio"):
            self.spatial_compression_ratio = int(self.vae.spatial_compression_ratio)
        elif hasattr(self.vae, "scale_factor"):
            self.spatial_compression_ratio = int(self.vae.scale_factor)
        else:
            self.spatial_compression_ratio = 8  # default

        # Scaling and shift factors for latent normalization
        if hasattr(self.vae, "config"):
            self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
            self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
            if self.shift_factor is None:  # this is the case for dc-ae models
                self.shift_factor = 0.0
        else:
            self.scaling_factor = 0.18215  # default
            self.shift_factor = 0.0

        # Number of latent channels
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "latent_channels"):
            self.latent_channels = int(self.vae.config.latent_channels)
        elif hasattr(self.vae, "config") and hasattr(self.vae.config, "in_channels"):
            self.latent_channels = int(self.vae.config.in_channels)
        else:
            # Use default_channels from YAML config (passed to __init__)
            self.latent_channels = self.default_channels

    @property
    def encoder(self) -> torch.nn.Module:
        """Return the encoder component of the underlying VAE."""
        return self.vae.encoder

    @property
    def decoder(self) -> torch.nn.Module:
        """Return the decoder component of the underlying VAE."""
        return self.vae.decoder

    @property
    def vae_scale_factor(self) -> int:
        """Alias for spatial_compression_ratio for compatibility."""
        return self.spatial_compression_ratio

    @property
    def vae_channels(self) -> int:
        """Alias for latent_channels for compatibility."""
        return self.latent_channels

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying VAE."""
        return next(self.vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying VAE."""
        return next(self.vae.parameters()).dtype

    def create_model(self, model_name: str) -> torch.nn.Module:
        """Create VAE model based on config."""
        # Use the model_class from __init__ parameter (from YAML config)
        model_class = self.model_class

        # Handle identity VAE for pixel-space diffusion
        if model_class == "IdentityVAE":
            return IdentityVAE(
                channels=self.default_channels,
                torch_dtype=self.torch_dtype
            )

        vae_class = AutoencoderDC if model_class == "AutoencoderDC" else AutoencoderKL

        try:
            vae = vae_class.from_pretrained(model_name, torch_dtype=torch.float32)
        except Exception:
            # Try subfolder if direct loading fails
            try:
                vae = vae_class.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Could not load VAE model {model_name}: {e}")

        vae.to(self.torch_dtype)
        return vae

    def scale_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale latent using VAE's scaling and shift factors."""
        return self.scaling_factor * (latent - self.shift_factor)

    def unscale_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Unscale latent for decoding."""
        return (1.0 / self.scaling_factor) * latent + self.shift_factor

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space.

        Args:
            image: Input image tensor in [0, 1] range

        Returns:
            Scaled latent tensor
        """
        # Scale image from [0, 1] to [-1, 1] for VAE
        image = image * 2.0 - 1.0
        with torch.inference_mode():
            latent = self.vae.encode(image.to(self.vae.dtype))

        # Handle different VAE output formats
        latent_dist = latent.get("latent_dist")
        if latent_dist is not None:  # AutoencoderKL
            latent_tensor = latent_dist.sample().data
        else:
            latent_tensor = latent.get("latent")
            if latent_tensor is not None:  # AutoencoderDC / IdentityVAE
                latent_tensor = latent_tensor.data.contiguous()
            else:
                raise ValueError("No latent found in VAE output")

        # Apply scaling
        return self.scale_latent(latent_tensor)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            latent: Scaled latent tensor

        Returns:
            Image tensor in [0, 1] range
        """
        # Unscale latent for decoding
        unscaled_latent = self.unscale_latent(latent)

        # Decode
        with torch.inference_mode():
            image = self.vae.decode(unscaled_latent.to(self.vae.dtype)).sample

        # Scale back to [0, 1]
        image = image.add_(1.0).div_(2.0).clamp_(0, 1)
        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass - encode image to latents."""
        return self.encode(image)
