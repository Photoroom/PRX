"""Optimizer construction utilities.

Provides factory functions for creating optimizers from Hydra configs,
including support for muon_fsdp2.Muon (Newton-Schulz orthogonalization with
FSDP2-optimized communication) with automatic parameter group splitting.
"""

import importlib
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig, OmegaConf


def _resolve_torch_optimizer(target: str):
    """Resolve an optimizer class from a string target.

    Accepts both short names (e.g. "AdamW") resolved from torch.optim,
    and fully-qualified paths (e.g. "some_pkg.SomeOptimizer").
    """
    if "." not in target:
        return getattr(torch.optim, target)

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def create_muon_optimizer(
    model: torch.nn.Module,
    config: DictConfig,
) -> torch.optim.Optimizer:
    """Create a muon_fsdp2.Muon optimizer with proper param groups.

    Uses ndim-based filtering:
    - ndim == 2 (matrices) in denoiser blocks -> Muon (Newton-Schulz)
    - Everything else -> Adam (internal to muon_fsdp2.Muon)

    The muon_fsdp2.Muon class handles both param groups internally,
    so no wrapper is needed.
    """
    from muon_fsdp2 import Muon

    muon_cfg = OmegaConf.to_container(config.muon_config)
    adam_cfg = OmegaConf.to_container(config.adam_config)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    muon_params = []
    adam_params = []

    for name, param in trainable:
        if param.ndim == 2 and "blocks" in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    print(
        f"[Muon] Hidden weights (Muon): {len(muon_params)} params, "
        f"{sum(p.numel() for p in muon_params):_} elements"
    )
    print(
        f"[Muon] Other params (Adam): {len(adam_params)} params, "
        f"{sum(p.numel() for p in adam_params):_} elements"
    )

    muon_cfg = {k: v for k, v in muon_cfg.items() if v is not None}
    adam_cfg = {k: v for k, v in adam_cfg.items() if v is not None}

    if "betas" in adam_cfg and isinstance(adam_cfg["betas"], list):
        adam_cfg["betas"] = tuple(adam_cfg["betas"])

    param_groups = [
        dict(params=muon_params, use_muon=True, **muon_cfg),
        dict(params=adam_params, use_muon=False, **adam_cfg),
    ]
    return Muon(param_groups)


def create_optimizer(
    model: torch.nn.Module,
    optimiser_config: dict[str, Any],
    optimizer_name: str = "optimizer",
) -> torch.optim.Optimizer:
    """Create an optimizer with optional parameter filtering."""
    key_name_filter = optimiser_config.pop("parameter_name_filter", None)
    freeze_name_filter = optimiser_config.pop("parameter_freeze_name_filter", None)

    def selected(name: str) -> bool:
        if key_name_filter and not any(k in name for k in key_name_filter):
            return False
        if freeze_name_filter and any(k in name for k in freeze_name_filter):
            return False
        return True

    trainable = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and selected(n)
    ]

    if key_name_filter or freeze_name_filter:
        print(f"[{optimizer_name}] Trainable parameters:")
        print("\n".join(f"  {n}" for n, _ in trainable))

    print(f"[{optimizer_name}] -> Layers to train: {len(trainable)}")
    print(
        f"[{optimizer_name}] -> Parameters to train: "
        f"{sum(p.numel() for _, p in trainable):_}"
    )

    opt_target = optimiser_config.pop("_target_")
    opt_class = (
        _resolve_torch_optimizer(opt_target)
        if isinstance(opt_target, str)
        else opt_target
    )

    optimiser_config = {
        k: v for k, v in optimiser_config.items() if v is not None
    }

    return opt_class(params=[p for _, p in trainable], **optimiser_config)


def create_optimizers(
    model: torch.nn.Module,
    config: DictConfig,
) -> torch.optim.Optimizer:
    """Create optimizer from config, supporting multiple modes.

    Checks for 'muon_optimizer' key first (muon_fsdp2.Muon with param
    group splitting), falls back to standard single optimizer via
    'optimizer'.
    """
    if "muon_optimizer" in config and config.muon_optimizer is not None:
        print("\n=== Creating muon_fsdp2.Muon optimizer ===")
        return create_muon_optimizer(model, config.muon_optimizer)
    else:
        optimiser_config = OmegaConf.to_container(config.optimizer)
        return create_optimizer(model, optimiser_config)
