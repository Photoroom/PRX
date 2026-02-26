from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.functional import fold, unfold

from .prx_layers import (
    EmbedND,  # spellchecker:disable-line
    LastLayer,
    get_image_ids,
    timestep_embedding,
    PRXBlock,
    MLPEmbedder,
)


@dataclass
class PRXParams:
    in_channels: int
    patch_size: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    axes_dim: list[int]
    theta: int

    # Time embedding parameters
    time_factor: float = 1000.0
    time_max_period: int = 10_000

    conditioning_block_ids: list[int] | None = None

    bottleneck_size: int | None = None


def img2seq(img: Tensor, patch_size: int) -> Tensor:
    """
    Flatten an image into a sequence of patches
    b c (h ph) (w pw) -> b (h w) (c ph pw)
    """
    return unfold(img, kernel_size=patch_size, stride=patch_size).transpose(1, 2)


def seq2img(seq: Tensor, patch_size: int, shape: Tensor) -> Tensor:
    """
    Revert img2seq
    b (h w) (c ph pw) -> b c (h ph) (w pw)
    """
    if isinstance(shape, tuple):
        shape = shape[-2:]
    elif isinstance(shape, torch.Tensor):
        shape = (int(shape[0]), int(shape[1]))
    else:
        raise NotImplementedError(f"shape type {type(shape)} not supported")
    return fold(seq.transpose(1, 2), shape, kernel_size=patch_size, stride=patch_size)


class PRX(nn.Module):
    """
    PRX diffusion transformer model.

    A transformer-based architecture for image generation that operates on patchified
    latent representations. The model processes image latents through a series of
    transformer blocks with common attention mechanism where image tokens attend both to image and text tokens.

    Args:
        params: Model configuration, either as PRXParams dataclass, dict, or DictConfig.
            See PRXParams for available configuration options.
        **kwargs: Alternative way to pass parameters when loading from storage.

    Attributes:
        params: The PRXParams configuration object.
        in_channels: Number of input latent channels.
        patch_size: Size of patches for patchifying the latent.
        out_channels: Output channels (in_channels * patch_size^2).
        hidden_size: Transformer hidden dimension.
        num_heads: Number of attention heads.
    """

    transformer_block_class = PRXBlock

    def _init_params(self, params: PRXParams) -> None:
        """
        Store model configuration and forward key parameters to instance attributes.

        This helper keeps __init__ focused on module construction by extracting
        the parameter assignment logic.
        """
        self.params = params
        self.in_channels = params.in_channels
        self.patch_size = params.patch_size
        self.out_channels = self.in_channels * self.patch_size**2
        self.time_factor = params.time_factor
        self.time_max_period = params.time_max_period
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

    def __init__(self, params: PRXParams | Dict[str, Any] | None = None, **kwargs: Any):
        super().__init__()

        if params is None:
            # Case when loaded from bucket: model_class(**parameters)
            params = kwargs

        if isinstance(params, (dict, DictConfig)):
            # Handle both regular dicts and Hydra DictConfig
            # Remove metadata keys
            params_dict = {k: v for k, v in params.items() if not k.startswith("_")}
            # Create PRXParams from the cleaned dictionary
            params = PRXParams(**params_dict)
        elif not isinstance(params, PRXParams):
            raise TypeError(f"params must be PRXParams, dict, or DictConfig, got {type(params)}")

        self._init_params(params)

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")

        pe_dim = params.hidden_size // params.num_heads

        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.pe_embedder = EmbedND(  # spellchecker:disable-line
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        patch_dim = self.in_channels * self.patch_size**2
        if params.bottleneck_size is not None:
            # Two-layer bottleneck: patch_dim -> bottleneck -> hidden
            self.img_in = nn.Sequential(
                nn.Linear(patch_dim, params.bottleneck_size, bias=True),
                nn.Linear(params.bottleneck_size, self.hidden_size, bias=True)
            )
        else:
            # Standard single linear layer
            self.img_in = nn.Linear(patch_dim, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        conditioning_block_ids: list[int] = params.conditioning_block_ids or list(
            range(params.depth)
        )  # Use only conditioning blocks if conditioning_block_ids is not defined

        def block_class(idx: int) -> PRXBlock:
            return self.transformer_block_class if idx in conditioning_block_ids else PRXBlock

        self.blocks = nn.ModuleList(
            [
                block_class(i)(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for i in range(params.depth)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def process_inputs(self, image_latent: Tensor, txt: Tensor, **_: Any) -> tuple[Tensor, Tensor, Tensor]:
        "Timestep independent stuff"
        txt = self.txt_in(txt)
        img = img2seq(image_latent, self.patch_size)
        bs, _, h, w = image_latent.shape
        img_ids = get_image_ids(bs, h, w, patch_size=self.patch_size, device=image_latent.device)
        pe = self.pe_embedder(img_ids)  # [bs, 1, seq_length, 64, 2, 2]
        return img, txt, pe

    def compute_timestep_embedding(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        return self.time_in(
            timestep_embedding(t=timestep, dim=256, max_period=self.time_max_period, time_factor=self.time_factor).to(
                dtype
            )
        )

    def forward_transformers(
        self,
        image_latent: Tensor,
        prompt_embeds: Tensor,
        timestep: Optional[Tensor] = None,
        time_embedding: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **block_kwargs: Any,
    ) -> Tensor:
        img = self.img_in(image_latent)

        if time_embedding is not None:
            # In that case, the provided timestep is already embedded.
            vec = time_embedding
        else:
            if timestep is None:
                raise ValueError("Please provide either a timestep or a timestep_embedding")
            vec = self.compute_timestep_embedding(timestep, dtype=img.dtype)
        for block in self.blocks:
            img = block(img=img, txt=prompt_embeds, vec=vec, attention_mask=attention_mask, **block_kwargs)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(
        self,
        image_latent: Tensor,
        timestep: Tensor,
        prompt_embeds: Tensor,  # TODO: rename text embedding everywhere
        prompt_attention_mask: None | Tensor = None,
    ) -> Tensor:
        img_seq, txt, pe = self.process_inputs(image_latent, prompt_embeds)
        img_seq = self.forward_transformers(
            img_seq,
            txt,
            timestep,
            pe=pe,
            attention_mask=prompt_attention_mask,
        )
        return seq2img(img_seq, self.patch_size, image_latent.shape)


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    DEVICE = torch.device("cpu")
    DTYPE = torch.bfloat16
    TORCH_COMPILE = False

    BS = 2
    LATENT_C = 16
    FEATURE_H, FEATURE_W = 1024 // 8, 1024 // 8
    PROMPT_L = 120
    # Create the denoiser - load config from YAML using Hydra composition
    config_dir = str(Path(__file__).parent.parent / "training" / "yamls" / "model")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="prx_small")

    denoiser = PRX(config)
    total_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Total number of parameters : {total_params / 1e9: .3f}B")
    denoiser = denoiser.to(DEVICE, DTYPE)

    if TORCH_COMPILE:
        denoiser = torch.compile(denoiser)

    out = denoiser(
        image_latent=torch.randn(BS, LATENT_C, FEATURE_H, FEATURE_W, device=DEVICE, dtype=DTYPE),
        timestep=torch.zeros(BS, device=DEVICE, dtype=DTYPE),
        prompt_embeds=torch.zeros(BS, PROMPT_L, 2304, device=DEVICE, dtype=DTYPE),  # T5 text encoding
        prompt_attention_mask=torch.ones(BS, PROMPT_L, device=DEVICE, dtype=DTYPE),
    )
    loss = out.sum()
    loss.backward()

    FEATURE_H, FEATURE_W = 1248 // 8, 832 // 8
    out = denoiser(
        image_latent=torch.randn(BS, LATENT_C, FEATURE_H, FEATURE_W, device=DEVICE, dtype=DTYPE),
        timestep=torch.zeros(BS, device=DEVICE, dtype=DTYPE),
        prompt_embeds=torch.zeros(BS, PROMPT_L, 2304, device=DEVICE, dtype=DTYPE),  # T5 text encoding
        prompt_attention_mask=torch.ones(BS, PROMPT_L, device=DEVICE, dtype=DTYPE),
    )
    loss = out.sum()
    loss.backward()
