import contextlib
from collections.abc import Callable, Generator
from typing import Any

import torch
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from torch import Tensor, nn


class Tread(Algorithm):
    """
    TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training
    (https://arxiv.org/pdf/2501.04765v3)
            img [B,N,C]         pe [B,1,N,...]
             │                    │
             ▼                    ▼
        ┌─── pre-hook: gather ────────┐
        │                             │
   visible_tokens  routed_tokens  visible_pe
   [B,Nv,C]        [B,Nr,C]          │
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
        ┌── post-hook: scatter ───────┐
        │                             │
        │  out[visible_idx] = visible │
        │  out[routed_idx]  = stash   │
        └─────────┬───────────────────┘
                  ▼
           img [B,N,C]  (restored)


    Assumptions:
      - The Composer model exposes `model.denoiser.blocks` as an `nn.ModuleList`.
      - Each block is called with `img=...` (B, N_img, C) and optionally `pe=...` in kwargs.

    Compatibility:
      - PRX-style: `pe` has shape [B, 1, N_img, ...].
    """

    _GOLDEN_RATIO_MIX = 0x9E3779B97F4A7C15
    _LCG_MULTIPLIER = 1664525

    def __init__(
        self,
        route_start: int,
        route_end: int,
        routing_probability: float,
        detach: bool = False,
        seed: int | None = None,
        train_only: bool = True,
        self_guidance: bool = False,
    ) -> None:
        super().__init__()
        assert 0 <= route_start < route_end, "Require 0 <= route_start < route_end"
        assert 0 <= routing_probability <= 1, "Probability must be between 0 and 1"

        self.route_start = int(route_start)
        self.route_end = int(route_end)
        self.routing_probability = float(routing_probability)
        self.detach = bool(detach)
        self.seed = seed
        self.train_only = bool(train_only)
        self.self_guidance = bool(self_guidance)

        self._enabled: bool = True
        self._handles: list[Any] = []
        self._hooks_registered: bool = False
        self._active: bool = False
        self._stash: dict[str, Tensor] = {}
        self._visible_pe: Tensor | None = None
        self._random_generator: torch.Generator | None = None
        self._seed_base: int = int(seed if (seed is not None) else 0)
        self._step_seed: int | None = None
        self._blocks: nn.ModuleList | None = None
        self._denoiser: nn.Module | None = None
        self._denoiser_hook_handle: Any | None = None
        self._repa_loss: nn.Module | None = None
        self._repa_hook_handle: Any | None = None
        # Self-guidance state
        self._original_generate: Callable[..., Any] | None = None
        self._guidance_ctx: Any | None = None

    def match(self, event: Event, state: State) -> bool:
        if event in (Event.FIT_START, Event.FIT_END):
            return True
        if self.train_only and event in (Event.EVAL_START, Event.EVAL_END):
            return True
        if event is Event.BATCH_START:
            return True
        return False

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event is Event.FIT_START:
            if self._blocks is None:
                denoiser = self._resolve_attr_path(state.model, path="denoiser")
                self._denoiser = denoiser  # Store denoiser reference
                self._blocks = self._get_blocks(denoiser)
                depth = len(self._blocks)
                if not (0 <= self.route_start < self.route_end < depth):
                    raise ValueError(f"Route ({self.route_start}->{self.route_end}) out of range depth={depth}")
            # Discover REPA loss if it exists and needs routing metadata
            if self._repa_loss is None:
                self._repa_loss = getattr(state.model, "repa_loss", None)
                if self._repa_loss is not None:
                    # Check if REPA's layer is in routing range
                    repa_layer = self._repa_loss.layer_index
                    if not (self.route_start <= repa_layer <= self.route_end):
                        # REPA is outside routing range, no metadata needed
                        self._repa_loss = None
            if not self._hooks_registered:
                self._register_hooks()
            self._enabled = True

        elif event is Event.FIT_END:
            if self._hooks_registered:
                self._teardown_hooks()

        elif self.train_only and event is Event.EVAL_START:
            self._enabled = False
            if self.self_guidance:
                self._wrap_generate_for_self_guidance(state)

        elif self.train_only and event is Event.EVAL_END:
            if self.self_guidance:
                self._unwrap_generate_for_self_guidance(state)
            self._enabled = True

        # per-batch seed, for activation checkpointing determinism
        elif event is Event.BATCH_START:
            rank = self._get_rank()
            step_idx = int(getattr(state.timestamp, "batch", 0))
            self._step_seed = self._compute_batch_seed(step_idx, rank)

    @staticmethod
    def _get_rank() -> int:
        """Get the current process rank in distributed training, or 0 if not distributed."""
        try:
            import torch.distributed as dist
        except ImportError:
            return 0

        rank: int = dist.get_rank() if dist.is_initialized() else 0
        return rank

    def _compute_batch_seed(self, step_idx: int, rank: int) -> int:
        """Compute a deterministic per-batch seed for reproducible token sampling."""
        return (self._seed_base + self._GOLDEN_RATIO_MIX + self._LCG_MULTIPLIER * step_idx + rank) % (2**63 - 1)

    def _reset_routing_state(self) -> None:
        """Reset all routing state (active flag, stash, and cached positional encodings)."""
        self._active = False
        self._stash.clear()
        self._visible_pe = None

    def _split_positional_encoding(
        self, pe: Tensor | None, routed_idx: Tensor, visible_idx: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """
        Split positional encodings into visible and routed subsets based on indices.

        This function assumes `pe` only contains image tokens (PRX-style).
        For Flux, where `pe` = [txt | img], we explicitly split text/image in
        `_pre_route_start` and call this only on the image part.
        """
        if pe is None:
            return None, None
        visible_pe = self._gather_positional_encoding(pe, visible_idx)
        routed_pe = self._gather_positional_encoding(pe, routed_idx)
        return visible_pe, routed_pe

    @staticmethod
    def _resolve_attr_path(root: nn.Module, path: str) -> nn.Module:
        """Resolve a dotted attribute path from a (possibly DDP/FSDP-wrapped) model."""
        obj = root
        for part in path.split("."):
            if part:
                obj = getattr(obj, part)
        return getattr(obj, "module", obj)

    @staticmethod
    def _get_blocks(model: nn.Module) -> nn.ModuleList:
        """Extract the ModuleList of transformer blocks from the model."""
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise ValueError("Expected model to have `blocks` as nn.ModuleList")
        return blocks

    def _inject_compute_mask_flag(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Inject compute_attn_mask_in_block flag for Flux models when routing will be active.

        Note: This is called before blocks execute, so we set the flag proactively
        when TREAD is enabled (routing_probability > 0). The actual routing happens
        in _pre_route_start hook on the first routed block.

        Only injects for Flux models, not PRX.
        """
        if self._enabled and self.routing_probability > 0:
            # Only inject for Flux models (check if model accepts this parameter)
            # PRX models don't have **kwargs in forward and will error
            model_class_name = _module.__class__.__name__
            if "Flux" in model_class_name:
                kwargs["compute_attn_mask_in_block"] = True
        return args, kwargs

    def _inject_repa_routing_info(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Inject routing metadata into REPA's forward kwargs when routing is active.

        Provides:
        - tread_original_num_tokens: Original number of tokens before routing
        - tread_visible_idx: Indices of visible tokens [B, num_visible]
        """
        if not self._enabled or not self._active:
            return args, kwargs

        # Inject routing metadata from stash
        if self._stash:
            kwargs["tread_original_num_tokens"] = self._stash["num_tokens"]
            kwargs["tread_visible_idx"] = self._stash["visible_idx"]

        return args, kwargs

    def _register_hooks(self) -> None:
        """Register forward hooks on transformer blocks to implement token routing."""
        assert self._blocks is not None
        if self._hooks_registered:
            return

        # Register denoiser hook to inject compute_attn_mask_in_block for Flux models
        if self._denoiser is not None and hasattr(self._denoiser, "forward"):
            self._denoiser_hook_handle = self._denoiser.register_forward_pre_hook(
                self._inject_compute_mask_flag, with_kwargs=True
            )

        # Register REPA hook to inject routing metadata
        if self._repa_loss is not None:
            self._repa_hook_handle = self._repa_loss.register_forward_pre_hook(
                self._inject_repa_routing_info, with_kwargs=True
            )

        # route_start pre-hook: reduce img+pe
        self._handles.append(
            self._blocks[self.route_start].register_forward_pre_hook(self._pre_route_start, with_kwargs=True)
        )
        # mid pre-hooks (including route_end): ensure reduced pe flows while active
        for i in range(self.route_start + 1, self.route_end + 1):
            self._handles.append(self._blocks[i].register_forward_pre_hook(self._pre_middle_layers, with_kwargs=True))
        # route_end post-hook: rebuild full sequence
        self._handles.append(self._blocks[self.route_end].register_forward_hook(self._post_route_end, with_kwargs=True))
        self._hooks_registered = True

    def _teardown_hooks(self) -> None:
        """Remove all registered hooks and reset routing state."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        # Remove denoiser hook if it exists
        if self._denoiser_hook_handle is not None:
            self._denoiser_hook_handle.remove()
            self._denoiser_hook_handle = None

        # Remove REPA hook if it exists
        if self._repa_hook_handle is not None:
            self._repa_hook_handle.remove()
            self._repa_hook_handle = None

        self._repa_loss = None
        self._hooks_registered = False
        self._reset_routing_state()
        self._random_generator = None
        self._step_seed = None

    def _ensure_generator(self, device: torch.device) -> None:
        """Initialize or reuse a random generator on the specified device with rank-aware seeding."""
        if self._random_generator is not None and getattr(self._random_generator, "device", None) == device:
            return
        rank = self._get_rank()
        seed_rank = (self._seed_base + rank) % (2**63 - 1)
        self._random_generator = torch.Generator(device=device)
        self._random_generator.manual_seed(seed_rank)

    @staticmethod
    def _sample_indices(
        batch_size: int,
        num_tokens: int,
        routing_probability: float,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """Sample indices for tokens to route away vs keep visible."""
        if routing_probability <= 0:
            num_routed = 0
        else:
            num_routed = int(round(num_tokens * routing_probability))
            num_routed = min(max(num_routed, 1), num_tokens - 1)

        if num_routed == 0:
            routed_idx = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            visible_idx = torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
            return routed_idx, visible_idx

        probs = torch.rand(batch_size, num_tokens, device=device, generator=generator)
        _, routed_idx = probs.topk(k=num_routed, dim=1, largest=False, sorted=True)

        all_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        keep_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device)
        keep_mask.scatter_(1, routed_idx, False)
        visible_idx = all_indices.masked_select(keep_mask).view(batch_size, num_tokens - num_routed)
        return routed_idx, visible_idx

    @staticmethod
    def _gather_batch_tokens(x: Tensor, idx: Tensor) -> Tensor:
        """Gather tokens from batch×tokens×hidden tensor using token indices."""
        batch_size, _, hidden_dim = x.shape
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, hidden_dim))

    @staticmethod
    def _scatter_batch_tokens(out: Tensor, idx: Tensor, vals: Tensor) -> Tensor:
        """Scatter tokens into batch×tokens×hidden tensor at specified indices."""
        return out.scatter(1, idx.unsqueeze(-1).expand(-1, -1, vals.shape[-1]), vals)

    @staticmethod
    def _gather_positional_encoding(pe: Tensor, idx: Tensor) -> Tensor:
        """Gather positional encodings at specified token indices (handles higher-dimensional PE)."""
        expand = idx.unsqueeze(1).unsqueeze(-1)
        for _ in range(pe.dim() - expand.dim()):
            expand = expand.unsqueeze(-1)
        expand = expand.expand(-1, 1, -1, *pe.shape[3:])
        return torch.gather(pe, 2, expand)

    def _pre_route_start(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Pre-hook at route_start: sample and split tokens into visible and routed subsets.

        For PRX:
            `pe` shape is [B, 1, N_img, ...] → we route all tokens in `pe`.

        For Flux:
            `pe` shape is [B, 1, N_txt + N_img, ...]; text tokens come first.
            We only route *image* tokens and keep the text part intact.
        """
        if not self._enabled:
            return args, kwargs

        # If routing is still active from previous forward pass, that means loss computation
        # has completed and we're starting a new forward pass. Clear the old routing state.
        if self._active:
            self._reset_routing_state()

        img: Tensor = kwargs["img"]
        pe: Tensor | None = kwargs.get("pe", None)
        batch_size, num_tokens, hidden_dim = img.shape

        self._ensure_generator(img.device)

        if self._step_seed is not None and self._random_generator is not None:
            self._random_generator.manual_seed(int(self._step_seed))

        routed_idx, visible_idx = self._sample_indices(
            batch_size, num_tokens, self.routing_probability, img.device, self._random_generator
        )
        visible_tokens = self._gather_batch_tokens(img, visible_idx)
        routed_tokens = self._gather_batch_tokens(img, routed_idx)
        if self.detach:
            routed_tokens = routed_tokens.detach()

        visible_pe = None
        routed_pe = None
        if pe is not None:
            # pe: [B, 1, T_total, ...]
            total_pe_tokens = pe.shape[2]

            if total_pe_tokens == num_tokens:
                # PRX-style: PE only for image tokens
                visible_pe, routed_pe = self._split_positional_encoding(pe, routed_idx, visible_idx)
                self._visible_pe = visible_pe

            elif total_pe_tokens > num_tokens:
                # FLUX-style: PE = [txt | img]; txt part is never routed
                n_txt = total_pe_tokens - num_tokens

                pe_txt = pe[:, :, :n_txt]  # [B, 1, N_txt, ...]
                pe_img = pe[:, :, n_txt:]  # [B, 1, N_img,  ...]

                visible_pe_img, routed_pe_img = self._split_positional_encoding(pe_img, routed_idx, visible_idx)

                # New PE seen by blocks: [txt | visible_img]
                self._visible_pe = torch.cat([pe_txt, visible_pe_img], dim=2)
                routed_pe = routed_pe_img

            else:
                # Unexpected: PE shorter than img token count
                raise ValueError(
                    f"Unexpected PE shape: total_pe_tokens={total_pe_tokens} < num_tokens={num_tokens}. "
                    f"PE should have at least as many tokens as the image sequence."
                )

        # Stash state for post hook
        self._stash = {
            "routed_idx": routed_idx,
            "visible_idx": visible_idx,
            "routed_tokens": routed_tokens,
            "routed_pe": routed_pe,
            "num_tokens": num_tokens,
            "hidden_dim": hidden_dim,
            "num_visible": num_tokens - routed_idx.shape[1],
        }
        self._active = True

        kwargs["img"] = visible_tokens
        if self._visible_pe is not None:
            kwargs["pe"] = self._visible_pe
        return args, kwargs

    def _pre_middle_layers(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Pre-hook for middle layers: ensure visible positional encodings flow through."""
        if not self._enabled or not self._active:
            return args, kwargs
        if self._visible_pe is not None:
            kwargs["pe"] = self._visible_pe
        return args, kwargs

    def _post_route_end(self, _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> Any:
        """
        Post-hook at route_end: reconstruct full image sequence by merging visible and routed tokens.

        Handles two cases:
          - PRX blocks: output is a Tensor (img only).
          - Flux blocks:   output is a tuple (img, txt, ...). We route only the img part
                           and pass txt (and any extra elements) through unchanged.
        """
        if not self._enabled or not self._active:
            return output

        # 1) Normalize output to (img_out, extra) where extra can be () or (txt, ...)
        is_tensor_only = isinstance(output, Tensor)
        if is_tensor_only:
            img_out = output
            extra = ()
        elif isinstance(output, (tuple, list)):
            if len(output) == 0:
                # Nothing to do
                self._reset_routing_state()
                return output
            img_out = output[0]
            extra = tuple(output[1:])
        else:
            # Unknown output structure: don't touch it
            self._reset_routing_state()
            return output

        # 2) Apply routing reconstruction to the image part
        batch_size, num_visible_got, hidden_dim_got = img_out.shape
        stash = self._stash
        num_tokens = stash["num_tokens"]
        hidden_dim = stash["hidden_dim"]
        num_visible_exp = stash["num_visible"]

        if num_visible_got != num_visible_exp:
            self._reset_routing_state()
            return output

        assert hidden_dim_got == hidden_dim, "Hidden size must match."

        out_full = img_out.new_zeros(batch_size, num_tokens, hidden_dim)
        out_full = self._scatter_batch_tokens(out_full, stash["visible_idx"], img_out)
        out_full = self._scatter_batch_tokens(out_full, stash["routed_idx"], stash["routed_tokens"])

        # 3) Do NOT reset routing state here - REPA (and other loss modules) may need
        #    the routing metadata after denoiser forward completes but before loss computation.
        #    State will be cleared at the start of the next forward pass in _pre_route_start.

        # 4) Rebuild final output with the same structure as the original
        if is_tensor_only:
            return out_full
        else:
            # Preserve type (tuple vs list)
            if isinstance(output, tuple):
                return (out_full, *extra)
            else:  # list
                return [out_full, *extra]

    # ============================================================
    # Self Guidance: Token-routing as guidance during inference
    # ============================================================
    #
    #   Instead of classical CFG (null-prompt vs real-prompt), use TREAD's
    #   token routing as the degradation mechanism:
    #
    #     full pass  (all N tokens, real prompt)  →  "conditional"
    #     routed pass (N_vis tokens, real prompt) →  "unconditional"
    #
    #     output = routed + scale × (full − routed)
    #
    #   At EVAL_START we monkey-patch model.generate so the existing pipeline
    #   and callbacks pick it up transparently.  At EVAL_END we restore.
    # ============================================================

    @contextlib.contextmanager
    def self_guidance_context(self, denoiser: nn.Module) -> Generator["Tread", None, None]:
        """Register temporary routing hooks on *denoiser* and yield self for _enabled toggling.

        If the inference denoiser is the EMA copy (separate blocks), we register
        temporary hooks.  If it is the training denoiser (same blocks — e.g. when
        EMA is not used), we reuse the existing training hooks to avoid duplicates
        and ``torch.compile`` issues.
        """
        blocks = self._get_blocks(denoiser)
        handles: list[Any] = []

        # Only register new hooks when the inference denoiser has different
        # blocks from the training denoiser (i.e. EMA copy).  When they are the
        # same object the training hooks are already present (and compiled into
        # the torch.compile graph), so adding a second set would cause double-
        # routing and shape mismatches.
        reuse_training_hooks = blocks is self._blocks

        if not reuse_training_hooks:
            handles.append(
                blocks[self.route_start].register_forward_pre_hook(self._pre_route_start, with_kwargs=True)
            )
            for i in range(self.route_start + 1, self.route_end + 1):
                handles.append(blocks[i].register_forward_pre_hook(self._pre_middle_layers, with_kwargs=True))
            handles.append(
                blocks[self.route_end].register_forward_hook(self._post_route_end, with_kwargs=True)
            )

        prev_enabled = self._enabled
        self._enabled = False
        self._reset_routing_state()

        try:
            yield self
        finally:
            for h in handles:
                h.remove()
            self._enabled = prev_enabled
            self._reset_routing_state()

    def set_inference_seed(self, seed: int) -> None:
        """Set a deterministic seed for routing during inference."""
        self._step_seed = self._compute_batch_seed(seed, self._get_rank())

    def enable(self) -> None:
        """Enable token routing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable token routing."""
        self._enabled = False

    def _make_self_guidance_step(self, denoiser: Any) -> Callable:
        """Build a per-step guidance callable for use with ``Pipeline.generate``."""
        actual_denoiser = getattr(denoiser, "model", denoiser)
        ctx = self.self_guidance_context(actual_denoiser)
        routing = ctx.__enter__()
        self._guidance_ctx = ctx  # prevent GC, cleaned up in _unwrap

        def step_fn(
            denoiser: Any, latent_input: Tensor, timestep: Tensor,
            denoiser_kwargs: dict, guidance_scale: float,
        ) -> Tensor:
            routing.disable()
            full_output = denoiser(image_latent=latent_input, timestep=timestep, **denoiser_kwargs)
            routing.enable()
            routed_output = denoiser(image_latent=latent_input, timestep=timestep, **denoiser_kwargs)
            return routed_output + guidance_scale * (full_output - routed_output)

        return step_fn

    def _wrap_generate_for_self_guidance(self, state: State) -> None:
        """Replace ``state.model.generate`` with a thin wrapper that injects ``guidance_func``."""
        model = state.model
        self._original_generate = model.generate
        algo = self
        original = self._original_generate

        denoiser = model.ema_denoiser if model.ema_denoiser.is_active else model.denoiser
        step_fn = algo._make_self_guidance_step(denoiser)

        def wrapped(*, seed: Any = None, **kwargs: Any) -> Tensor:
            if seed is not None:
                algo.set_inference_seed(seed if isinstance(seed, int) else seed[0])
            return original(seed=seed, guidance_func=step_fn, **kwargs)

        model.generate = wrapped  # type: ignore[assignment]

    def _unwrap_generate_for_self_guidance(self, state: State) -> None:
        """Restore original ``model.generate`` at EVAL_END."""
        if self._original_generate is not None:
            state.model.generate = self._original_generate  # type: ignore[assignment]
            self._original_generate = None
        if self._guidance_ctx is not None:
            self._guidance_ctx.__exit__(None, None, None)
            self._guidance_ctx = None
