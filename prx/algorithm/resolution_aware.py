"""Resolution-Aware conditioning algorithm for Composer.

Injects image resolution information (H_latent, W_latent) into the denoiser
so the model learns resolution-specific behavior during multi-resolution training.

Two modes:
- "vec": Adds resolution embedding to the timestep vec (uniform modulation).
- "token": Prepends resolution tokens to the text sequence (selective attention).
"""

import logging
from typing import Any

import torch
from torch import nn, Tensor
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

from prx.models.prx_layers import MLPEmbedder, timestep_embedding

log = logging.getLogger(__name__)


class ResolutionEmbedder(nn.Module):
    """Embeds (H, W) into resolution conditioning.

    In 'vec' mode: returns a single vector [B, hidden_size] to add to timestep vec.
    In 'token' mode: returns two tokens [B, 2, hidden_size] to prepend to text.
    """

    def __init__(self, hidden_size: int, mode: str = "token", max_period: int = 10000):
        super().__init__()
        self.mode = mode
        self.max_period = max_period
        if mode == "vec":
            self.mlp = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        elif mode == "token":
            self.h_embedder = MLPEmbedder(in_dim=128, hidden_dim=hidden_size)
            self.w_embedder = MLPEmbedder(in_dim=128, hidden_dim=hidden_size)
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected 'vec' or 'token'")

    def forward(self, height: Tensor, width: Tensor) -> Tensor:
        h_emb = timestep_embedding(height, dim=128, max_period=self.max_period, time_factor=1.0)
        w_emb = timestep_embedding(width, dim=128, max_period=self.max_period, time_factor=1.0)
        if self.mode == "vec":
            return self.mlp(torch.cat([h_emb, w_emb], dim=-1))  # [B, hidden_size]
        else:  # token
            h_tok = self.h_embedder(h_emb)  # [B, hidden_size]
            w_tok = self.w_embedder(w_emb)  # [B, hidden_size]
            return torch.stack([h_tok, w_tok], dim=1)  # [B, 2, hidden_size]


class ResolutionAware(Algorithm):
    """Resolution-Aware conditioning algorithm.

    Injects (H_latent, W_latent) into the denoiser via hooks.

    Args:
        mode: "vec" (add to timestep vec) or "token" (prepend to text sequence).
        max_period: Controls sinusoidal embedding frequency range.
    """

    def __init__(self, mode: str = "token", max_period: int = 10000):
        super().__init__()
        self.mode = mode
        self.max_period = max_period
        self.resolution_embedder: ResolutionEmbedder | None = None
        self._modules_added = False
        self._hooks_registered = False
        # Mutable state captured by hooks
        self._hook_state: dict[str, Any] = {}

    def add_new_pipeline_modules(self, model: nn.Module) -> None:
        """Attach ResolutionEmbedder to the model before optimizer creation."""
        if self._modules_added:
            log.info(" > ResolutionAware: Modules already added, skipping")
            return
        denoiser = model.denoiser if hasattr(model, "denoiser") else model
        hidden_size = denoiser.hidden_size
        embedder = ResolutionEmbedder(hidden_size, self.mode, self.max_period)
        model.resolution_embedder = embedder
        num_params = sum(p.numel() for p in embedder.parameters())
        log.info(f" > ResolutionAware: Attached embedder (mode={self.mode}, params={num_params:,})")
        self._modules_added = True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        # Resolve denoiser (handle DDP/FSDP wrapping)
        if hasattr(state.model, "module"):
            model_unwrapped = state.model.module
        else:
            model_unwrapped = state.model

        if not hasattr(model_unwrapped, "resolution_embedder"):
            raise RuntimeError(
                "ResolutionAware module not found on model. "
                "Did you call add_new_pipeline_modules() in train.py before creating the optimizer?"
            )

        denoiser = model_unwrapped.denoiser
        self.resolution_embedder = model_unwrapped.resolution_embedder
        self._register_hooks(denoiser)

        logger.log_hyperparameters({
            "resolution_aware/mode": self.mode,
            "resolution_aware/max_period": self.max_period,
        })
        log.info(f" > ResolutionAware: Registered hooks (mode={self.mode})")

    def _register_hooks(self, denoiser: nn.Module) -> None:
        """Register forward hooks on the denoiser."""
        if self._hooks_registered:
            return
        self._hooks_registered = True
        hook_state = self._hook_state
        embedder = self.resolution_embedder

        # Pre-hook on denoiser.forward: capture (B, H, W, device) from image_latent
        def denoiser_pre_hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            # image_latent may be positional or keyword
            image_latent = args[0] if args else kwargs["image_latent"]
            B, _C, H, W = image_latent.shape
            hook_state["B"] = B
            hook_state["H"] = H
            hook_state["W"] = W
            hook_state["device"] = image_latent.device

        denoiser.register_forward_pre_hook(denoiser_pre_hook, with_kwargs=True)

        if self.mode == "vec":
            # Post-hook on denoiser.time_in: add resolution embedding to vec
            def time_in_post_hook(module: nn.Module, input: Any, output: Tensor) -> Tensor:
                B = hook_state["B"]
                H = hook_state["H"]
                W = hook_state["W"]
                device = hook_state["device"]
                h = torch.full((B,), H, device=device, dtype=torch.float32)
                w = torch.full((B,), W, device=device, dtype=torch.float32)
                res_emb = embedder(h, w).to(output.dtype)
                return output + res_emb

            denoiser.time_in.register_forward_hook(time_in_post_hook)

        elif self.mode == "token":
            # Wrap forward_transformers to prepend resolution tokens
            original_forward_transformers = denoiser.forward_transformers

            def wrapped_forward_transformers(
                image_latent: Tensor,
                prompt_embeds: Tensor,
                *args: Any,
                attention_mask: Tensor | None = None,
                **kwargs: Any,
            ) -> Tensor:
                B = hook_state["B"]
                H = hook_state["H"]
                W = hook_state["W"]
                device = hook_state["device"]

                h = torch.full((B,), H, device=device, dtype=torch.float32)
                w = torch.full((B,), W, device=device, dtype=torch.float32)
                res_tokens = embedder(h, w).to(prompt_embeds.dtype)  # [B, 2, hidden_size]

                # Prepend resolution tokens to prompt_embeds
                prompt_embeds = torch.cat([res_tokens, prompt_embeds], dim=1)

                # Extend attention_mask if present
                if attention_mask is not None:
                    mask_prefix = torch.ones(B, 2, device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([mask_prefix, attention_mask], dim=1)

                return original_forward_transformers(
                    image_latent, prompt_embeds, *args,
                    attention_mask=attention_mask, **kwargs,
                )

            denoiser.forward_transformers = wrapped_forward_transformers
