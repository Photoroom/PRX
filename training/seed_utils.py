import logging
import random
from typing import Any

import torch
import torch.distributed as dist
from composer.utils import reproducibility
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def broadcast_from_rank0(value: int) -> int:
    """
    Broadcast a int value from rank 0 to all other ranks.

    Args:
        value: The value to broadcast from rank 0

    Returns:
        The broadcast value (same on all ranks)
    """
    if dist.is_initialized() is False:
        raise RuntimeError(
            "Distributed environment is not initialized. Please initialize it before calling this function."
        )

    tensor = torch.tensor(value, dtype=torch.long, device=dist.get_node_local_rank())
    dist.broadcast(tensor, src=0)
    value = tensor.item()
    return value


def _set_shuffle_seed_if_needed(name: str, dataset_config: dict[str, Any], is_distributed: bool) -> None:
    """Set shuffle seed for a dataset if needed.

    Args:
        name: Name of the dataset for logging purposes
        dataset_config: Dataset configuration dictionary
        is_distributed: Whether running in distributed mode
    """
    if dataset_config.get("shuffle_seed", None) is None and dataset_config.get("shuffle", False) is True:
        # Ensure same seed across all ranks
        seed = random.randint(0, 2**31 - 1)
        if is_distributed:
            seed = broadcast_from_rank0(seed)
        dataset_config["shuffle_seed"] = seed
        logger.info(f"Setting {name} shuffle_seed to {dataset_config['shuffle_seed']}")


def set_seeds(config: DictConfig, is_distributed: bool) -> DictConfig:
    """Set the random seeds for reproducibility."""
    config = OmegaConf.to_container(config, resolve=True)

    # Main seed
    if config.get("seed", None) is None:
        seed = random.randint(0, 2**31 - 1)
        if is_distributed:
            seed = broadcast_from_rank0(seed)
        config["seed"] = seed

    reproducibility.seed_all(config["seed"])

    # Dataset shuffle seeds
    ds_cfg = config["dataset"]

    _set_shuffle_seed_if_needed("train_dataset", ds_cfg["train_dataset"], is_distributed)

    if "eval_dataset" in ds_cfg:
        _set_shuffle_seed_if_needed("eval_dataset", ds_cfg["eval_dataset"], is_distributed)

    if "evaluators" in ds_cfg:
        for evaluator_name, evaluator_config in ds_cfg["evaluators"].items():
            eval_dataset_cfg = evaluator_config["eval_dataset"]
            _set_shuffle_seed_if_needed(f"evaluator {evaluator_name}", eval_dataset_cfg, is_distributed)

    return OmegaConf.create(config)
