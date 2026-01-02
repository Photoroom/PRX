from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from composer.devices import DeviceGPU

from models.text_tower import TextTower
from models.vae_tower import VaeTower
from models.prx import PRX

from schedulers.scheduler import EulerDiscreteScheduler, SchedulerConfig

from .pipeline import LatentDiffusion



def wrap_fsdp_module(module: torch.nn.Module, value: bool) -> None:
    if hasattr(module, "fsdp_wrap"):
        module.fsdp_wrap(value)
    else:
        module._fsdp_wrap = value



def str_to_torch_dtype(dtype_str: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None

    if isinstance(dtype_str, torch.dtype):
        return dtype_str

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

    if dtype_str in dtype_map:
        return dtype_map[dtype_str]

    raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")


def build_schedulers(
    prediction_type: str = "flow_matching",
    timestep_shift: Optional[int] = None,
    num_train_timesteps: int = 1000,
) -> Tuple[EulerDiscreteScheduler, EulerDiscreteScheduler]:
    """Build training and inference noise schedulers.

    Args:
        prediction_type: Type of prediction ("flow_matching")
        timestep_shift: Optional timestep shift parameter (defaults to 1.0 if None)
        num_train_timesteps: Number of training timesteps (default: 1000)

    Returns:
        Tuple of (noise_scheduler, inference_noise_scheduler)
    """
    config = SchedulerConfig(
        num_train_timesteps=num_train_timesteps,
        prediction_type=prediction_type,
    )

    shift = float(timestep_shift) if timestep_shift is not None else 1.0
    noise_scheduler = EulerDiscreteScheduler(
        config=config,
        shift=shift,
    )
    inference_noise_scheduler = EulerDiscreteScheduler(
        config=config,
        shift=shift,
    )

    return noise_scheduler, inference_noise_scheduler


def _get_device() -> torch.device:
    if torch.distributed.is_initialized():
        return torch.device(DeviceGPU()._device) if torch.cuda.is_available() else torch.device("cpu")
    return torch.cuda.current_device()


def build_pipeline(
    # Component configs (from Hydra config groups)
    denoiser_config: Dict[str, Any],
    text_tower_config: Dict[str, Any],
    vae_config: Dict[str, Any],

    # Pipeline settings
    input_size: int = 512,
    p_drop_caption: float = 0.1,
    prediction_type: str = "flow_matching",
    timestep_shift: Optional[float] = None,
    num_train_timesteps: int = 1000,

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
) -> LatentDiffusion:
    """Build diffusion pipeline from Hydra-composed configs.
    config groups instead of string references to Python objects.

    Args:
        denoiser_config: PRX model config dict (from model/*.yaml)
        text_tower_config: Text encoder config dict (from text_tower/*.yaml)
        vae_config: VAE config dict (from vae/*.yaml)
        input_size: Input image size
        p_drop_caption: Caption drop probability for CFG
        prediction_type: Noise prediction type
        timestep_shift: Timestep shift parameter
        num_train_timesteps: Number of training timesteps
        train_metrics: Training metrics list
        val_metrics: Validation metrics list
        val_guidance_scales: Guidance scales for validation
        val_seed: Random seed for validation
        loss_bins: Loss binning for logging
        negative_prompt: Default negative prompt
        latent_channels: Override denoiser in_channels if specified
        denoiser_dtype: Override denoiser dtype
        **kwargs: Additional args passed to LatentDiffusion

    Returns:
        LatentDiffusion pipeline instance
    """
    device = _get_device()

    print(f" > Building diffusion pipeline V2 - device {device}")

    text_tower_cfg = text_tower_config.copy()
    if 'torch_dtype' in text_tower_cfg:
        text_tower_cfg['torch_dtype'] = str_to_torch_dtype(text_tower_cfg['torch_dtype'])

    text_tower_cfg.pop('preset_name', None)

    with torch.device(device):
        # Build text tower
        print(f" > Building TextTower: {text_tower_cfg.get('model_name')}")
        text_tower = TextTower(**text_tower_cfg)
        text_tower.requires_grad_(False)
        wrap_fsdp_module(text_tower, False)

        # Build VAE
        vae_cfg = vae_config.copy()
        if 'torch_dtype' in vae_cfg:
            vae_cfg['torch_dtype'] = str_to_torch_dtype(vae_cfg['torch_dtype'])

        print(f" > Building VaeTower: {vae_cfg.get('model_name')}")
        vae = VaeTower(**vae_cfg)
        vae.requires_grad_(False)
        wrap_fsdp_module(vae, False)

    # Build schedulers
    noise_scheduler, inference_noise_scheduler = build_schedulers(
        prediction_type=prediction_type,
        timestep_shift=timestep_shift,
        num_train_timesteps=num_train_timesteps,
    )
    wrap_fsdp_module(noise_scheduler, False)
    wrap_fsdp_module(inference_noise_scheduler, False)

    # Build denoiser - CRITICAL: adjust context_in_dim to match text_tower
    denoiser_cfg = denoiser_config.copy()
    if denoiser_cfg.get('context_in_dim') != text_tower.hidden_size:
        print(
            f" > WARNING: Adjusting context_in_dim from {denoiser_cfg.get('context_in_dim')} "
            f"to {text_tower.hidden_size} to match text_tower.hidden_size"
        )
        denoiser_cfg['context_in_dim'] = text_tower.hidden_size

    # Override in_channels if latent_channels is specified
    if latent_channels is not None:
        denoiser_cfg['in_channels'] = latent_channels
        print(f" > Overriding denoiser in_channels to {latent_channels}")
        

    denoiser = PRX(denoiser_cfg)
    print(" > Using PRX model")
        

    # Apply dtype to denoiser
    if denoiser_dtype is not None:
        denoiser_dtype = str_to_torch_dtype(denoiser_dtype)
        denoiser.to(denoiser_dtype)
        print(f" > Denoiser dtype: {denoiser_dtype}")

    wrap_fsdp_module(denoiser, True)

    print(f" > Total denoiser params: {sum(p.numel() for p in denoiser.parameters()) / 1e9:.3f}B")

    # Build pipeline
    pipeline = LatentDiffusion(
        denoiser=denoiser,
        vae=vae,
        text_tower=text_tower,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
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
