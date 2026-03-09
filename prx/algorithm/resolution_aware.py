"""Resolution-Aware conditioning algorithm for Composer.

Injects image resolution information (H_latent, W_latent) into the denoiser
so the model learns resolution-specific behavior during multi-resolution training.

Two modes:
- "vec": Adds resolution embedding to the timestep vec (uniform modulation).
- "token": Prepends resolution tokens to the text sequence (selective attention).
"""

import logging
from typing import Any, Literal

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

    def __init__(self, hidden_size: int, mode: Literal["vec", "token"] = "token", max_period: int = 10000):
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

    def forward(self, height: Tensor, width: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        h_emb = timestep_embedding(height, dim=128, max_period=self.max_period, time_factor=1.0)
        w_emb = timestep_embedding(width, dim=128, max_period=self.max_period, time_factor=1.0)
        if dtype is not None:
            h_emb = h_emb.to(dtype)
            w_emb = w_emb.to(dtype)
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

    def __init__(self, mode: Literal["vec", "token"] = "token", max_period: int = 10000):
        super().__init__()
        self.mode = mode
        self.max_period = max_period
        self._modules_added = False
        # Mutable state captured by hooks (shared — only one denoiser runs at a time)
        self._hook_state: dict[str, Any] = {}

    def add_new_pipeline_modules(self, model: nn.Module) -> None:
        """Attach ResolutionEmbedder to the denoiser before optimizer creation.

        Attaching to the denoiser (not the pipeline) ensures that EMA's deepcopy
        of the denoiser includes the embedder, and compute_ema averages its weights.
        """
        if self._modules_added:
            log.info(" > ResolutionAware: Modules already added, skipping")
            return
        denoiser = model.denoiser if hasattr(model, "denoiser") else model
        hidden_size = denoiser.hidden_size
        embedder = ResolutionEmbedder(hidden_size, self.mode, self.max_period)
        denoiser.resolution_embedder = embedder
        num_params = sum(p.numel() for p in embedder.parameters())
        log.info(f" > ResolutionAware: Attached embedder (mode={self.mode}, params={num_params:,})")
        self._modules_added = True

    def _unwrap_model(self, state: State) -> torch.nn.Module:
        return state.model.module if hasattr(state.model, "module") else state.model

    def match(self, event: Event, state: State) -> bool:
        return event in (Event.INIT, Event.FIT_START)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        model_unwrapped = self._unwrap_model(state)

        if event == Event.INIT:
            denoiser = model_unwrapped.denoiser
            if not hasattr(denoiser, "resolution_embedder"):
                raise RuntimeError(
                    "ResolutionEmbedder not found on denoiser. "
                    "Did you call add_new_pipeline_modules() in train.py before creating the optimizer?"
                )
            self._register_hooks_on_denoiser(denoiser)
            logger.log_hyperparameters({
                "resolution_aware/mode": self.mode,
                "resolution_aware/max_period": self.max_period,
            })
            log.info(f" > ResolutionAware: Registered hooks (mode={self.mode})")

        if event == Event.FIT_START:
            # Register hooks on EMA denoiser if the EMA algorithm initialized it.
            # By FIT_START all INIT events have fired, so ema.model exists iff EMA is active.
            ema = getattr(model_unwrapped, "ema_denoiser", None)
            if ema is not None and hasattr(ema, "model"):
                self._register_hooks_on_denoiser(ema.model)
                log.info(" > ResolutionAware: Also registered hooks on EMA denoiser")

    def _register_hooks_on_denoiser(self, denoiser: nn.Module) -> None:
        """Register forward hooks on a denoiser instance.

        Called once per denoiser (training + EMA). Each denoiser uses its own
        resolution_embedder (attached in add_new_pipeline_modules / deepcopy).
        The hook_state dict is shared since only one denoiser runs at a time.
        """
        hook_state = self._hook_state
        embedder = denoiser.resolution_embedder

        # Pre-hook on denoiser.forward: capture (B, H, W, device) from image_latent
        def denoiser_pre_hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            if not args and "image_latent" not in kwargs:
                raise ValueError("denoiser.forward() called without image_latent as positional or keyword argument")
            image_latent = args[0] if args else kwargs["image_latent"]
            B, _C, H, W = image_latent.shape
            hook_state["B"] = B
            hook_state["H"] = H
            hook_state["W"] = W
            hook_state["device"] = image_latent.device

        denoiser.register_forward_pre_hook(denoiser_pre_hook, with_kwargs=True)

        if self.mode == "vec":
            def time_in_post_hook(module: nn.Module, input: Any, output: Tensor) -> Tensor:
                B = hook_state["B"]
                H = hook_state["H"]
                W = hook_state["W"]
                device = hook_state["device"]
                h = torch.full((B,), H, device=device, dtype=torch.float32)
                w = torch.full((B,), W, device=device, dtype=torch.float32)
                res_emb = embedder(h, w, dtype=output.dtype)
                return output + res_emb

            denoiser.time_in.register_forward_hook(time_in_post_hook)

        elif self.mode == "token":
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
                res_tokens = embedder(h, w, dtype=prompt_embeds.dtype)

                prompt_embeds = torch.cat([res_tokens, prompt_embeds], dim=1)

                if attention_mask is not None:
                    mask_prefix = torch.ones(B, 2, device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([mask_prefix, attention_mask], dim=1)

                return original_forward_transformers(
                    image_latent, prompt_embeds, *args,
                    attention_mask=attention_mask, **kwargs,
                )

            denoiser.forward_transformers = wrapped_forward_transformers
