import importlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _resolve_class(target: str):
    """Import and return the class identified by *target*.

    Accepts short names resolved from ``torch.optim`` (e.g. ``"AdamW"``)
    or fully-qualified paths (e.g. ``"some_pkg.SomeOptimizer"``).
    """
    if "." not in target:
        return getattr(torch.optim, target)

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _collect_trainable(
    model: torch.nn.Module,
    parameter_name_filter: list[str] | None = None,
    parameter_freeze_name_filter: list[str] | None = None,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return ``(name, param)`` pairs for all trainable parameters."""

    def selected(name: str) -> bool:
        if parameter_name_filter and not any(k in name for k in parameter_name_filter):
            return False
        if parameter_freeze_name_filter and any(k in name for k in parameter_freeze_name_filter):
            return False
        return True

    trainable = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and selected(n)
    ]

    if parameter_name_filter or parameter_freeze_name_filter:
        logger.info("Trainable parameters:\n%s", "\n".join(f"  {n}" for n, _ in trainable))

    logger.info("Layers to train: %d", len(trainable))
    logger.info("Parameters to train: %s", f"{sum(p.numel() for _, p in trainable):_}")
    return trainable


# ── Standard optimizer factory ──────────────────────────────────────


def create_standard_optimizer(
    model: torch.nn.Module,
    optimizer_class: str,
    parameter_name_filter: list[str] | None = None,
    parameter_freeze_name_filter: list[str] | None = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create a standard PyTorch optimizer.

    Args:
        model: The model whose parameters will be optimized.
        optimizer_class: Optimizer class path (e.g. ``"torch.optim.AdamW"``
            or just ``"AdamW"``).
        parameter_name_filter: If set, only train parameters whose names
            contain at least one of these substrings.
        parameter_freeze_name_filter: If set, freeze parameters whose names
            contain any of these substrings.
        **kwargs: Passed directly to the optimizer constructor (lr, betas, …).
    """
    trainable = _collect_trainable(model, parameter_name_filter, parameter_freeze_name_filter)
    opt_class = _resolve_class(optimizer_class)
    # Drop None values so unset Hydra config fields fall back to the optimizer's own defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return opt_class(params=[p for _, p in trainable], **kwargs)


# ── Muon optimizer factory ──────────────────────────────────────────


def create_muon_optimizer(
    model: torch.nn.Module,
    muon_config: dict[str, Any],
    adam_config: dict[str, Any],
    muon_name_filter: str = "blocks",
    parameter_name_filter: list[str] | None = None,
    parameter_freeze_name_filter: list[str] | None = None,
    **_kwargs: Any,
) -> torch.optim.Optimizer:
    """Create a ``muon_fsdp2.Muon`` optimizer with param group splitting.

    Splits trainable parameters into two groups:
    - **Muon group**: 2-D weight matrices matching *muon_name_filter* → Newton-Schulz.
    - **Adam group**: everything else → Adam.

    Args:
        model: The model whose parameters will be optimized.
        muon_config: Hyperparameters for the Muon (Newton-Schulz) group
            (lr, momentum, nesterov, ns_steps, …).
        adam_config: Hyperparameters for the Adam fallback group
            (lr, betas, eps, …).
        muon_name_filter: Substring that must appear in a parameter's name
            for it to be assigned to the Muon group (default ``"blocks"``).
        parameter_name_filter: If set, only train parameters whose names
            contain at least one of these substrings.
        parameter_freeze_name_filter: If set, freeze parameters whose names
            contain any of these substrings.
    """
    from muon_fsdp2 import Muon

    trainable = _collect_trainable(model, parameter_name_filter, parameter_freeze_name_filter)

    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []

    for name, param in trainable:
        if param.ndim == 2 and muon_name_filter in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    logger.info(
        "Hidden weights (Muon): %d params, %s elements",
        len(muon_params), f"{sum(p.numel() for p in muon_params):_}",
    )
    logger.info(
        "Other params (Adam): %d params, %s elements",
        len(adam_params), f"{sum(p.numel() for p in adam_params):_}",
    )

    muon_cfg = {k: v for k, v in muon_config.items() if v is not None}
    adam_cfg = {k: v for k, v in adam_config.items() if v is not None}

    if "betas" in adam_cfg and isinstance(adam_cfg["betas"], list):
        adam_cfg["betas"] = tuple(adam_cfg["betas"])

    param_groups = [
        dict(params=muon_params, use_muon=True, **muon_cfg),
        dict(params=adam_params, use_muon=False, **adam_cfg),
    ]
    return Muon(param_groups)
