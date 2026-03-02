import math
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel



def get_image_ids(bs: int, h: int, w: int, patch_size: int, device: torch.device) -> Tensor:
    img_ids = torch.zeros(h // patch_size, w // patch_size, 2, device=device)
    img_ids[..., 0] = torch.arange(h // patch_size, device=device)[:, None]
    img_ids[..., 1] = torch.arange(w // patch_size, device=device)[None, :]
    return img_ids.reshape((h // patch_size) * (w // patch_size), 2).unsqueeze(0).repeat(bs, 1, 1)


def apply_rope(xq: Tensor, freqs_cis: Tensor) -> Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    # CuDNN SDPA only when it can actually run
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attn_mask
                )
        except RuntimeError:
            pass
    return torch.nn.functional.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attn_mask
    )

def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# Adapted from https://github.com/black-forest-labs/flux
class EmbedND(nn.Module): # spellchecker:disable-line
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.rope_rearrange = Rearrange("b n d (i j) -> b n d i j", i=2, j=2)

    def rope(self, pos: Tensor, dim: int, theta: int) -> Tensor:
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)  # (B,N,1) * (1,D) -> B, N, D
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        out = self.rope_rearrange(out)
        return out.float()

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [self.rope(ids[:, :, i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms * self.scale).to(dtype=x_dtype)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)

        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:])


class PRXBlock(nn.Module):
    """
    PRX transformer block with text-conditioned attention.

    Each block consists of:
        1. Modulated attention where image tokens attend to both image and text tokens,
           with RoPE positional encoding applied to image keys/queries
        2. Gated MLP with SwiGLU-style activation

    The block uses adaptive layer normalization (adaLN) where timestep embeddings
    modulate the normalization via learned shift, scale, and gate parameters.

    Args:
        hidden_size: Dimension of the hidden representations.
        num_heads: Number of attention heads.
        mlp_ratio: Multiplier for the MLP hidden dimension relative to hidden_size.
        qk_scale: Optional scale factor for attention. Defaults to 1/sqrt(head_dim).

    Forward args:
        img: Image token sequence of shape (B, L_img, hidden_size).
        txt: Text conditioning tokens of shape (B, L_txt, hidden_size).
        vec: Timestep embedding vector of shape (B, hidden_size) for modulation.
        pe: Rotary positional embeddings for image tokens.
        attention_mask: Optional mask for text tokens of shape (B, L_txt).

    Returns:
        Updated image token sequence of shape (B, L_img, hidden_size).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()

        self._fsdp_wrap = True
        self._activation_checkpointing = True

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size

        # img qkv
        self.img_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.qk_norm = QKNorm(self.head_dim)

        # txt kv
        self.txt_kv_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.k_norm = RMSNorm(self.head_dim)

        # mlp
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")

        self.modulation = Modulation(hidden_size)

    def attn_forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        modulation: ModulationOut,
        attn_mask: None | Tensor = None,
    ) -> Tensor:
        # image tokens proj and norm
        img_mod = (1 + modulation.scale) * self.img_pre_norm(img) + modulation.shift

        img_qkv = self.img_qkv_proj(img_mod)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)

        # txt tokens proj and norm
        txt_kv = self.txt_kv_proj(txt)
        txt_k, txt_v = rearrange(txt_kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        txt_k = self.k_norm(txt_k)

        # compute attention
        img_q, img_k = apply_rope(img_q, pe), apply_rope(img_k, pe)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # optional spatial conditioning tokens appended to keys/values
        cond_len = 0

        # build multiplicative 0/1 mask for provided attention_mask over [cond?, text, image] keys
        if attn_mask is not None:
            bs, _, l_img, _ = img_q.shape
            l_txt = txt_k.shape[2]

            assert attn_mask.dim() == 2, f"Unsupported attention_mask shape: {attn_mask.shape}"
            assert (
                attn_mask.shape[-1] == l_txt
            ), f"attention_mask last dim {attn_mask.shape[-1]} must equal text length {l_txt}"

            device = img_q.device

            ones_img = torch.ones((bs, l_img), dtype=torch.bool, device=device)
            cond_mask = torch.ones((bs, cond_len), dtype=torch.bool, device=device)

            mask_parts = [
                cond_mask,
                attn_mask.to(torch.bool),
                ones_img,
            ]
            joint_mask = torch.cat(mask_parts, dim=-1)  # (B, L_all)

            # repeat across heads and query positions
            attn_mask = joint_mask[:, None, None, :].expand(-1, self.num_heads, l_img, -1)  # (B,H,L_img,L_all)

        # convert 0/1 mask to additive bias: 0 -> large negative, 1 -> 0
        if attn_mask is not None:
            dtype = img_q.dtype
            neg_large = torch.finfo(dtype).min
            attn_mask = (1.0 - attn_mask.to(dtype)) * neg_large

        attn = _sdpa(img_q, k, v, attn_mask=attn_mask)
        attn = rearrange(attn, "B H L D -> B L (H D)")
        attn = self.attn_out(attn)

        return attn

    def ffn_forward(self, x: Tensor, modulation: ModulationOut) -> Tensor:
        x = (1 + modulation.scale) * self.post_attention_layernorm(x) + modulation.shift
        return self.down_proj(self.mlp_act(self.gate_proj(x)) * self.up_proj(x))

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: Tensor | None = None,
        **_: dict[str, Any],
    ) -> Tensor:  # override: ignore
        mod_attn, mod_mlp = self.modulation(vec)

        img = img + mod_attn.gate * self.attn_forward(
            img,
            txt,
            pe,
            mod_attn,
            attn_mask=attention_mask,
        )
        img = img + mod_mlp.gate * self.ffn_forward(img, mod_mlp)
        return img


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
