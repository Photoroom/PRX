
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Union

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
    """Clean up stale shared memory segments used by MosaicML Streaming."""
    print(
        " > MosaicML: initiating cleanup of stale shared memory segments used by MosaicML Streaming. This is to ensure node stability and prevent potential conflicts or resource leaks."
    )
    streaming.base.util.clean_stale_shared_memory()


def create_optimizer(
    model: torch.nn.Module,
    optimiser_config: dict[str, Any],
) -> torch.optim.Optimizer:
    key_name_filter = optimiser_config.pop("parameter_name_filter", None)
    freeze_name_filter = optimiser_config.pop("parameter_freeze_name_filter", None)

    def selected(name: str) -> bool:
        if key_name_filter and not any(k in name for k in key_name_filter):
            return False
        if freeze_name_filter and any(k in name for k in freeze_name_filter):
            return False
        return True

    trainable = [(n, p) for n, p in model.named_parameters() if selected(n)]

    if key_name_filter or freeze_name_filter:
        print("Trainable parameters:")
        print("\n".join(n for n, _ in trainable))

    print(f"-> # layers to train: {len(trainable)}")
    print(f"-> # params to train: {sum(p.numel() for _, p in trainable):_}")

    opt_class = optimiser_config.pop("_target_")  # e.g. torch.optim.AdamW (class)
    return opt_class(params=[p for _, p in trainable], **optimiser_config)


def train(config: DictConfig) -> None:
    """Train a model.

    Args:
        config (DictConfig): Configuration composed by Hydra
    Returns:
        Optional[float]: Metric score for hyperparameter optimization
    """
    global_destroy_dist = maybe_init_dist()
    clean_up_mosaic()
    config = set_seeds(config, is_distributed=global_destroy_dist)

    if dist.get_global_rank() == 0:
        print(config)

    if "activation_memory_budget" in config:
        # activation checkpointing
        functorch_config.activation_memory_budget = config.activation_memory_budget

    logger: List[LoggerDestination] = []
    if "logger" in config:
        for log, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                if log == "wandb":
                    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                    # use _partial_ so it doesn't try to init everything
                    wandb_logger = hydra.utils.instantiate(lg_conf, _partial_=True)
                    logger.append(wandb_logger(init_kwargs={"config": container}))
                else:
                    logger.append(hydra.utils.instantiate(lg_conf))

    model: ComposerModel = hydra.utils.instantiate(config.model)

    # Build list of algorithms BEFORE optimizer creation
    # This allows algorithms to add modules to the pipeline via add_new_pipeline_modules()
    algorithms: List[Algorithm] = []
    if config.get("algorithms", None) is not None:
        for ag_name, ag_conf in config.algorithms.items():
            if "_target_" in ag_conf:
                print(f"Instantiating algorithm <{ag_conf._target_}>")
                algorithms.append(hydra.utils.instantiate(ag_conf))

    # Call add_new_pipeline_modules() for algorithms that have it
    # This adds modules to pipeline before optimizer creation
    for algorithm in algorithms:
        if hasattr(algorithm, 'add_new_pipeline_modules'):
            algorithm.add_new_pipeline_modules(model)

    # NOW create optimizer - will include any modules added by algorithms
    optimiser_config = OmegaConf.to_container(config.optimizer)
    optimizer = create_optimizer(model, optimiser_config)

    train_dataloader: Union[Iterable[Any], DataSpec, Dict[str, Any]] = hydra.utils.instantiate(
        config.dataset.train_dataset,
        batch_size=config.global_batch_size // dist.get_world_size(),
    )

    eval_set = hydra.utils.instantiate(
        config.dataset.eval_dataset,
        batch_size=config.device_eval_microbatch_size,
    )

    # Build list of callbacks to pass to trainer
    callbacks: List[Callback] = []

    if "callbacks" in config:
        for _, call_conf in config.callbacks.items():
            if call_conf and "_target_" in call_conf:
                print(f"Instantiating callbacks <{call_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(call_conf))

    scheduler = hydra.utils.instantiate(config.scheduler) if "scheduler" in config else None

    # Implementing a 0-second sleep as a temporary workaround for an NCCL race condition issue,
    # which can cause unexpected behavior in distributed training scenarios. This delay helps in
    # avoiding the race condition until a permanent fix is applied. For detailed context and
    # ongoing discussions about this problem, refer to the PyTorch GitHub issue:
    # https://github.com/pytorch/pytorch/issues/119196
    # works by default, may need to increase if we see issues, tbc
    # nccl_sleep = config.get("nccl_sleep", 0)
    # print(
    #     f" > NCCL: initiating a {nccl_sleep}-second sleep to mitigate an NCCL race condition issue. See PyTorch GitHub issue at https://github.com/pytorch/pytorch/issues/119196 for more details."
    # )
    # time.sleep(nccl_sleep)
    # print(" > NCCL: wake-up now ")

    # save config file in the checkpoint folder
    with open(Path(config.trainer.save_folder) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

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
        pipeline = trainer.state.model
        if isinstance(pipeline, DistributedDataParallel):
            pipeline = pipeline.module

        model = pipeline
        for attr in module_name.split("."):
            model = getattr(model, attr)

        if hasattr(model, "experts"):
            for expert in model.experts:
                compile_model(expert, **compile_kwargs)
            return

        compiled_model = torch.compile(model, **compile_kwargs)
        model = compiled_model._orig_mod
        model.forward = compiled_model.dynamo_ctx(model.forward)

    # TODO: flexible torch compile config
    if "compile_denoiser" in config and config.compile_denoiser is True:
        print("> Compiling the denoiser")
        compile_model("denoiser")

    if "compile_vae" in config and config.compile_vae is True:
        print("> Compiling the vae encoder")
        compile_model("vae.encoder", dynamic=True)


    def eval_and_then_train() -> None:
        if config.get("eval_first", True):
            if hasattr(config.trainer, "eval_subset_num_batches"):
                trainer.eval(subset_num_batches=config.trainer.eval_subset_num_batches)
            else:
                trainer.eval()

        trainer.fit()

        if (
            global_destroy_dist
        ):  # cleanup process group at end of training. (not sure if this is needed, but to be safe)
            torch_dist.destroy_process_group()

    return eval_and_then_train()


@hydra.main(version_base=None, config_path="yamls", config_name="PRX-JIT-1024")
def main(config: DictConfig) -> None:
    """Hydra wrapper for train."""
    return train(config)


if __name__ == "__main__":
    main()