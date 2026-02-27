from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from algorithm.tread import Tread


class SPRINT(Tread):
    """
    SPRINT-style routing on top of TREAD:
      - stash dense (pre-drop) tokens (full sequence before routing)
      - run middle blocks on visible tokens
      - restore full length with [MASK] for dropped tokens
      - fuse: Linear([dense || restored_deep]) -> hidden
            img [B,N,C]         pe [B,1,N,...]
             │                    │
             ▼                    ▼
        ┌─── pre-hook: gather ────────┐
        │                             │
   visible_tokens    dense_tokens  visible_pe
   [B,Nv,C]         [B,N,C]          │
        │               │            │
        │               ▼            │
        │            STASH           │
        │               │            │
        ▼               │            ▼
   ┌────────────────────│──────────────┐
   │ Blocks start → end │  (Nv tokens) │
   └────────┬───────────│──────────────┘
            │           │
            ▼           ▼
        ┌── post-hook: fuse ──────────┐
        │                             │
        │  deep = [MASK] × (B,N,C)    │
        │  deep[visible_idx] = output │
        │                             │
        │  fused = Linear(            │
        │    cat[dense, deep], dim=-1 │
        │  )  # (B,N,2C) → (B,N,C)    │
        └─────────┬───────────────────┘
                  ▼
           img [B,N,C]  (fused)

    """

    def __init__(
        self,
        *args,
        fuse_name: str = "sprint_fuse",
        mask_name: str = "sprint_mask_token",
        learnable_mask: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fuse_name = fuse_name
        self.mask_name = mask_name
        self.learnable_mask = learnable_mask

        # Cached refs (set in add_new_pipeline_modules / FIT_START)
        self._denoiser: nn.Module | None = None
        self._fuse: nn.Module | None = None

    def add_new_pipeline_modules(self, model: nn.Module) -> None:
        """
        Called in train.py before optimizer creation.
        Adds fusion + mask to model.denoiser so optimizer picks them up.
        """
        denoiser = self._resolve_attr_path(model, "denoiser")
        self._denoiser = denoiser

        # Infer hidden size C
        hidden = getattr(denoiser, "hidden_size", None)
        if hidden is None:
            img_in = getattr(denoiser, "img_in", None)
            if isinstance(img_in, nn.Linear):
                hidden = img_in.out_features
        if hidden is None:
            raise ValueError("Could not infer denoiser hidden size for SPRINT fusion layer.")

        # Fusion: Linear(2C -> C)
        if not hasattr(denoiser, self.fuse_name):
            denoiser.add_module(self.fuse_name, nn.Linear(2 * hidden, hidden, bias=True))

        # Mask token: (1,1,C) buffer or parameter
        if not hasattr(denoiser, self.mask_name):
            init = torch.zeros(1, 1, hidden)
            if self.learnable_mask:
                denoiser.register_parameter(self.mask_name, nn.Parameter(init))
            else:
                denoiser.register_buffer(self.mask_name, init, persistent=True)

        self._fuse = getattr(denoiser, self.fuse_name)

    def _pre_route_start(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Pre-hook at route_start:
          - Always stash dense tokens (full sequence) for fusion.
          - If training: route tokens with routing_probability.
          - If eval: routing_p=0 (keep all tokens), but still run fusion later.
        """
        if not self._enabled:
            return args, kwargs

        # Clear previous forward's routing state (important when hooks stay registered)
        if self._active:
            self._reset_routing_state()

        img: Tensor = kwargs["img"]
        pe: Tensor | None = kwargs.get("pe", None)
        batch_size, num_tokens, hidden_dim = img.shape

        # Deterministic sampling (same as TREAD)
        self._ensure_generator(img.device)
        if self._step_seed is not None and self._random_generator is not None:
            self._random_generator.manual_seed(int(self._step_seed))

        routing_p = self.routing_probability if _module.training else 0.0 # this is needed for eval during training

        # Sample indices (if routing_p==0, this returns routed_idx empty and visible_idx=arange(N))
        routed_idx, visible_idx = self._sample_indices(
            batch_size, num_tokens, routing_p, img.device, self._random_generator
        )

        visible_tokens = self._gather_batch_tokens(img, visible_idx)

        # Visible PE MUST match visible tokens for RoPE correctness
        visible_pe = None
        if pe is not None:
            # For PRX-style PE [B, 1, N, ...], gather along token dimension (2)
            visible_pe = self._gather_positional_encoding(pe, visible_idx)
            self._visible_pe = visible_pe
        else:
            self._visible_pe = None


        # Stash minimal state needed for post hook
        self._stash = {
            "routed_idx": routed_idx,
            "visible_idx": visible_idx,
            "num_tokens": num_tokens,
            "hidden_dim": hidden_dim,
            "dense_tokens": img,
        }
        self._active = True

        # Replace kwargs with visible subset for middle blocks
        kwargs["img"] = visible_tokens
        if self._visible_pe is not None:
            kwargs["pe"] = self._visible_pe

        return args, kwargs

    def _pre_middle_layers(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Pre-hook for middle layers:
        PRX re-supplies the original full `pe` every block call via **block_kwargs,
        so we MUST overwrite it whenever routing is active.
        """
        if not self._enabled or not self._active:
            return args, kwargs

        if self._visible_pe is not None:
            kwargs["pe"] = self._visible_pe

        return args, kwargs

    def _post_route_end(self, _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> Any:
        """
        Post-hook at route_end:
          - Build padded deep stream with [MASK] at dropped positions.
          - Concatenate with dense stream and apply fusion projection.
          - Return full-length tensor to feed into decoder blocks.
        """
        if not self._enabled or not self._active:
            return output

        stash = self._stash
        if "dense_tokens" not in stash or "visible_idx" not in stash or "num_tokens" not in stash:
            self._reset_routing_state()
            return output

        if not isinstance(output, Tensor):
            self._reset_routing_state()
            return output

        img_out: Tensor = output  # (B, N_visible, C)
        batch_size, _, hidden_dim_got = img_out.shape

        num_tokens = int(stash["num_tokens"])
        hidden_dim = int(stash["hidden_dim"])
        if hidden_dim_got != hidden_dim:
            self._reset_routing_state()
            return output

        denoiser = self._denoiser
        if denoiser is None or not hasattr(denoiser, self.fuse_name) or not hasattr(denoiser, self.mask_name):
            self._reset_routing_state()
            return output

        fuse = getattr(denoiser, self.fuse_name)
        mask_tok: Tensor = getattr(denoiser, self.mask_name)  # (1,1,C) parameter or buffer

        visible_idx: Tensor = stash["visible_idx"]
        dense: Tensor = stash["dense_tokens"]  # (B, N, C)

        # Build padded deep stream: start with mask everywhere, scatter visible outputs
        out_full = (
            mask_tok.to(device=img_out.device, dtype=img_out.dtype)
            .expand(batch_size, num_tokens, hidden_dim)
            .clone()
        )
        out_full = self._scatter_batch_tokens(out_full, visible_idx, img_out)

        fused_in = torch.cat([dense, out_full], dim=-1)  # (B, N, 2C)
        fused = fuse(fused_in)  # (B, N, C)

        return fused
