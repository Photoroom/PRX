import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from composer.devices import DeviceGPU
from omegaconf import DictConfig, OmegaConf

from models.text_tower import TextTower
from models.vae_tower import VaeTower
from models.prx import PRX

from schedulers.scheduler import EulerDiscreteScheduler, SchedulerConfig

from .pipeline import Pipeline

logger = logging.getLogger(__name__)


class Schedulers(NamedTuple):
    train: EulerDiscreteScheduler
    inference: EulerDiscreteScheduler


def wrap_fsdp_module(module: torch.nn.Module, value: bool) -> None:
    if hasattr(module, "fsdp_wrap"):
        module.fsdp_wrap(value)
    else:
        module._fsdp_wrap = value



def resolve_torch_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_map = {
        "torch.float": torch.float,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "float": torch.float,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    if dtype in dtype_map:
        return dtype_map[dtype]

    raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(dtype_map.keys())}")


def build_schedulers(scheduler_config: Dict[str, Any]) -> Schedulers:
    """Build training and inference noise schedulers.

    Args:
        scheduler_config: Scheduler config dict (from scheduler/*.yaml)

    Returns:
        Schedulers namedtuple with train and inference schedulers
    """
    config = SchedulerConfig(
        num_train_timesteps=scheduler_config.get('num_train_timesteps', 1000),
        prediction_type=scheduler_config.get('prediction_type', 'flow_matching'),
    )

    shift = float(scheduler_config.get('timestep_shift', 1.0))
    return Schedulers(
        train=EulerDiscreteScheduler(config=config, shift=shift),
        inference=EulerDiscreteScheduler(config=config, shift=shift),
    )


def _get_device() -> torch.device:
    if torch.distributed.is_initialized():
        return torch.device(DeviceGPU()._device) if torch.cuda.is_available() else torch.device("cpu")
    return torch.cuda.current_device()


def build_pipeline(
    # Component configs (from Hydra config groups)
    denoiser_config: Dict[str, Any],
    text_tower_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],

    # Pipeline settings
    input_size: int = 512,
    p_drop_caption: float = 0.1,

    # Metrics and validation
    train_metrics: Optional[List[Any]] = None,
    val_metrics: Optional[List[Any]] = None,
    val_guidance_scales: Optional[List[float]] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List[Tuple[float, float]]] = None,
    negative_prompt: str = "",

    # Optional overrides
    latent_channels: Optional[int] = None,
    denoiser_dtype: Optional[Union[str, torch.dtype]] = None,

    **kwargs: Any,
) -> Pipeline:
    """Build diffusion pipeline from Hydra-composed configs.
    config groups instead of string references to Python objects.

    Args:
        denoiser_config: PRX model config dict (from model/*.yaml)
        text_tower_config: Text encoder config dict (from text_tower/*.yaml)
        vae_config: VAE config dict (from vae/*.yaml)
        scheduler_config: Scheduler config dict (from scheduler/*.yaml)
        input_size: Input image size
        p_drop_caption: Caption drop probability for CFG
        train_metrics: Training metrics list
        val_metrics: Validation metrics list
        val_guidance_scales: Guidance scales for validation
        val_seed: Random seed for validation
        loss_bins: Loss binning for logging
        negative_prompt: Default negative prompt
        latent_channels: Override denoiser in_channels if specified
        denoiser_dtype: Override denoiser dtype
        **kwargs: Additional args passed to Pipeline

    Returns:
        Pipeline pipeline instance
    """
    device = _get_device()

    logger.info("Building diffusion pipeline V2 - device %s", device)

    text_tower_cfg = OmegaConf.to_container(text_tower_config) if isinstance(text_tower_config, DictConfig) else dict(text_tower_config)
    if 'torch_dtype' in text_tower_cfg:
        text_tower_cfg['torch_dtype'] = resolve_torch_dtype(text_tower_cfg['torch_dtype'])


    with torch.device(device):
        # Build text tower
        logger.info("Building TextTower: %s", text_tower_cfg.get('model_name'))
        text_tower = TextTower(**text_tower_cfg)
        text_tower.requires_grad_(False)
        wrap_fsdp_module(text_tower, False)

        # Build VAE
        vae_cfg = vae_config.copy()
        if 'torch_dtype' in vae_cfg:
            vae_cfg['torch_dtype'] = resolve_torch_dtype(vae_cfg['torch_dtype'])

        logger.info("Building VaeTower: %s", vae_cfg.get('model_name'))
        vae = VaeTower(**vae_cfg)
        vae.requires_grad_(False)
        wrap_fsdp_module(vae, False)

    # Build schedulers
    schedulers = build_schedulers(scheduler_config)
    wrap_fsdp_module(schedulers.train, False)
    wrap_fsdp_module(schedulers.inference, False)

    # Build denoiser - CRITICAL: adjust context_in_dim to match text_tower
    denoiser_cfg = denoiser_config.copy()
    if denoiser_cfg.get('context_in_dim') != text_tower.hidden_size:
        logger.warning(
            "Adjusting context_in_dim from %s to %s to match text_tower.hidden_size",
            denoiser_cfg.get('context_in_dim'),
            text_tower.hidden_size,
        )
        denoiser_cfg['context_in_dim'] = text_tower.hidden_size

    # Override in_channels if latent_channels is specified
    if latent_channels is not None:
        denoiser_cfg['in_channels'] = latent_channels
        logger.info("Overriding denoiser in_channels to %s", latent_channels)
        

    denoiser = PRX(denoiser_cfg)
    logger.info("Using PRX model")
        

    # Apply dtype to denoiser
    if denoiser_dtype is not None:
        denoiser_dtype = resolve_torch_dtype(denoiser_dtype)
        denoiser.to(denoiser_dtype)
        logger.info("Denoiser dtype: %s", denoiser_dtype)

    wrap_fsdp_module(denoiser, True)

    logger.info("Total denoiser params: %.3fB", sum(p.numel() for p in denoiser.parameters()) / 1e9)

    # Build pipeline
    pipeline = Pipeline(
        denoiser=denoiser,
        vae=vae,
        text_tower=text_tower,
        noise_scheduler=schedulers.train,
        inference_noise_scheduler=schedulers.inference,
        p_drop_caption=p_drop_caption,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_seed=val_seed,
        val_guidance_scales=val_guidance_scales,
        loss_bins=loss_bins,
        negative_prompt=negative_prompt,
        **kwargs,
    )

    if torch.cuda.is_available():
        pipeline = pipeline.to(device=device)

    return pipeline
