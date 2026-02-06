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


def _set_shuffle_seed_if_needed(
    name: str, dataset_config: dict[str, Any], rng: random.Random
) -> None:
    """Set shuffle seed for a dataset if needed.

    Args:
        name: Name of the dataset for logging purposes
        dataset_config: Dataset configuration dictionary
        rng: Random number generator initialized from the main seed
    """
    if dataset_config.get("shuffle_seed", None) is None and dataset_config.get("shuffle", False) is True:
        # Generate deterministic shuffle seed from the provided RNG
        seed = rng.randint(0, 2**31 - 1)
        dataset_config["shuffle_seed"] = seed
        logger.info(f"Setting {name} shuffle_seed to {dataset_config['shuffle_seed']}")


def set_seeds(config: DictConfig, is_distributed: bool) -> DictConfig:
    """Set the random seeds for reproducibility."""
    config = OmegaConf.to_container(config, resolve=True)

    # Main seed
    if config.get("seed", None) is None:
        # Use a fixed default seed for deterministic behavior across runs
        seed = 0
        if is_distributed:
            seed = broadcast_from_rank0(seed)
        config["seed"] = seed

    reproducibility.seed_all(config["seed"])

    # Use a dedicated RNG so that all subsequent random numbers are derived
    # deterministically from the main seed, without relying on global state.
    rng = random.Random(config["seed"])

    # Dataset shuffle seeds
    ds_cfg = config["dataset"]

    _set_shuffle_seed_if_needed("train_dataset", ds_cfg["train_dataset"], rng)

    if "eval_dataset" in ds_cfg:
        _set_shuffle_seed_if_needed("eval_dataset", ds_cfg["eval_dataset"], rng)

    if "evaluators" in ds_cfg:
        for evaluator_name, evaluator_config in ds_cfg["evaluators"].items():
            eval_dataset_cfg = evaluator_config["eval_dataset"]
            _set_shuffle_seed_if_needed(f"evaluator {evaluator_name}", eval_dataset_cfg, rng)

    return OmegaConf.create(config)
