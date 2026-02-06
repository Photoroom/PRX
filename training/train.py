"""Training script for PRX models using Composer and Hydra.

This module provides the main training loop and utilities for training diffusion models
with MosaicML Composer framework. It handles distributed training setup, optimizer
configuration, algorithm registration, and model compilation.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Union
import importlib

import hydra
import streaming
import torch
import torch._functorch.config as functorch_config
from composer import Algorithm, Callback, ComposerModel, DataSpec, Trainer
from composer.loggers import LoggerDestination
from composer.utils import dist
from omegaconf import DictConfig, OmegaConf
from streaming.base.distributed import maybe_init_dist
from torch import distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel

from seed_utils import set_seeds


def clean_up_mosaic() -> None:
    """Clean up stale shared memory segments used by MosaicML Streaming.

    This ensures node stability and prevents potential conflicts or resource leaks
    from stale shared memory segments.
    """
    print(
        " > MosaicML: Initiating cleanup of stale shared memory segments. "
        "This ensures node stability and prevents potential resource leaks."
    )
    streaming.base.util.clean_stale_shared_memory()


def _resolve_torch_optimizer(target: str):
    # Accept "AdamW" or "torch.optim.AdamW"
    if "." not in target:
        return getattr(torch.optim, target)

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def create_optimizer(
    model: torch.nn.Module,
    optimiser_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create an optimizer with optional parameter filtering.

    Args:
        model: The model to optimize
        optimiser_config: Optimizer configuration dict with optional filters:
            - parameter_name_filter: Only train params containing these strings
            - parameter_freeze_name_filter: Freeze params containing these strings
            - _target_: Optimizer class (e.g., torch.optim.AdamW)

    Returns:
        Configured optimizer instance
    """
    key_name_filter = optimiser_config.pop("parameter_name_filter", None)
    freeze_name_filter = optimiser_config.pop("parameter_freeze_name_filter", None)

    def selected(name: str) -> bool:
        """Check if parameter should be trainable based on name filters."""
        if key_name_filter and not any(k in name for k in key_name_filter):
            return False
        if freeze_name_filter and any(k in name for k in freeze_name_filter):
            return False
        return True

    trainable = [(n, p) for n, p in model.named_parameters() if selected(n)]

    if key_name_filter or freeze_name_filter:
        print("Trainable parameters:")
        print("\n".join(n for n, _ in trainable))

    print(f"-> Layers to train: {len(trainable)}")
    print(f"-> Parameters to train: {sum(p.numel() for _, p in trainable):_}")

    opt_target = optimiser_config.pop("_target_")  # likely "torch.optim.AdamW"
    opt_class = _resolve_torch_optimizer(opt_target) if isinstance(opt_target, str) else opt_target

    return opt_class(params=[p for _, p in trainable], **optimiser_config)


def train(config: DictConfig) -> None:
    """Train a model using Composer framework.

    Args:
        config: Hydra configuration containing model, dataset, trainer, and algorithm settings
    """
    # Initialize distributed training
    global_destroy_dist = maybe_init_dist()
    clean_up_mosaic()
    config = set_seeds(config, is_distributed=global_destroy_dist)

    if dist.get_global_rank() == 0:
        print(config)

    # Configure activation checkpointing if specified
    if "activation_memory_budget" in config:
        functorch_config.activation_memory_budget = config.activation_memory_budget

    # Initialize loggers
    logger: List[LoggerDestination] = []
    if "logger" in config:
        for log_name, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                if log_name == "wandb":
                    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                    wandb_logger = hydra.utils.instantiate(lg_conf, _partial_=True)
                    logger.append(wandb_logger(init_kwargs={"config": container}))
                else:
                    logger.append(hydra.utils.instantiate(lg_conf))

    # Instantiate model
    model: ComposerModel = hydra.utils.instantiate(config.model)

    # Build algorithms before optimizer creation
    # This allows algorithms to add modules to the pipeline before optimization
    algorithms: List[Algorithm] = []
    if config.get("algorithms", None) is not None:
        for _, ag_conf in config.algorithms.items():
            if "_target_" in ag_conf:
                print(f"Instantiating algorithm <{ag_conf._target_}>")
                algorithms.append(hydra.utils.instantiate(ag_conf))

    # Allow algorithms to register new modules before optimizer creation
    for algorithm in algorithms:
        if hasattr(algorithm, "add_new_pipeline_modules"):
            algorithm.add_new_pipeline_modules(model)

    # Create optimizer (includes any modules added by algorithms)
    optimiser_config = OmegaConf.to_container(config.optimizer)
    optimizer = create_optimizer(model, optimiser_config)

    # Create dataloaders
    train_dataloader: Union[Iterable[Any], DataSpec, Dict[str, Any]] = hydra.utils.instantiate(
        config.dataset.train_dataset,
        batch_size=config.global_batch_size // dist.get_world_size(),
    )

    eval_set = hydra.utils.instantiate(
        config.dataset.eval_dataset,
        batch_size=config.device_eval_microbatch_size,
    )

    # Build callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, call_conf in config.callbacks.items():
            if call_conf and "_target_" in call_conf:
                print(f"Instantiating callback <{call_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(call_conf))

    # Create learning rate scheduler
    scheduler = hydra.utils.instantiate(config.scheduler) if "scheduler" in config else None

    # Save config file in the checkpoint folder
    with open(Path(config.trainer.save_folder) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_set,
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
        callbacks=callbacks,
    )

    def compile_model(module_name: str, **compile_kwargs: Any) -> None:
        """Compile a model module using torch.compile.

        Args:
            module_name: Dot-separated path to the module (e.g., "denoiser" or "vae.encoder")
            **compile_kwargs: Arguments to pass to torch.compile
        """
        pipeline = trainer.state.model
        if isinstance(pipeline, DistributedDataParallel):
            pipeline = pipeline.module

        target_module = pipeline
        for attr in module_name.split("."):
            target_module = getattr(target_module, attr)


        compiled_model = torch.compile(target_module, **compile_kwargs)
        target_module = compiled_model._orig_mod
        target_module.forward = compiled_model.dynamo_ctx(target_module.forward)

    # Apply model compilation if configured
    if config.get("compile_denoiser", False):
        print("> Compiling the denoiser")
        compile_model("denoiser")

    if config.get("compile_vae", False):
        print("> Compiling the VAE encoder")
        compile_model("vae.encoder", dynamic=True)


    def eval_and_then_train() -> None:
        """Run initial evaluation (if configured) followed by training."""
        if config.get("eval_first", True):
            if hasattr(config.trainer, "eval_subset_num_batches"):
                trainer.eval(subset_num_batches=config.trainer.eval_subset_num_batches)
            else:
                trainer.eval()

        trainer.fit()

        # Cleanup distributed process group
        if global_destroy_dist:
            torch_dist.destroy_process_group()

    return eval_and_then_train()


@hydra.main(version_base=None, config_path="yamls", config_name="PRX-JIT-1024")
def main(config: DictConfig) -> None:
    """Entry point for training with Hydra configuration management.

    Args:
        config: Hydra configuration automatically composed from config files
    """
    return train(config)


if __name__ == "__main__":
    main()