"""Integration tests for TextTower (Qwen3-VL-2B)."""

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from prx.models.text_tower import TextTower

CONFIG_REL_PATH: str = "configs/yamls/text_tower/qwen3vl_2b_256_bf16.yaml"

SAMPLE_TEXTS: list[str] = [
    "A photo of a cat sitting on a windowsill",
    "An abstract painting with vibrant colors",
    "",  # empty string edge case
]


@pytest.fixture(scope="module")
def cfg(repo_root: Path) -> dict[str, Any]:
    with open(repo_root / CONFIG_REL_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def torch_dtype(cfg: dict[str, Any]) -> torch.dtype:
    return getattr(torch, cfg["torch_dtype"].replace("torch.", ""))


# ── Tokenizer-only tests (no GPU required) ──────────────────────────


@pytest.fixture(scope="module")
def tower_tok(cfg: dict[str, Any], torch_dtype: torch.dtype) -> TextTower:
    return TextTower(
        model_name=cfg["model_name"],
        only_tokenizer=True,
        use_attn_mask=cfg["use_attn_mask"],
        prompt_max_tokens=cfg["prompt_max_tokens"],
        use_last_hidden_state=cfg["use_last_hidden_state"],
        torch_dtype=torch_dtype,
        skip_text_cleaning=cfg["skip_text_cleaning"],
    )


@pytest.mark.integration
class TestTextTowerTokenizer:

    def test_tokenize_shapes(
        self, tower_tok: TextTower, cfg: dict[str, Any]
    ) -> None:
        result = tower_tok.text_to_token(SAMPLE_TEXTS)
        max_tokens: int = cfg["prompt_max_tokens"]
        assert result.tokens.shape == (len(SAMPLE_TEXTS), max_tokens)
        assert result.attention_mask.shape == result.tokens.shape

    def test_single_string_input(self, tower_tok: TextTower) -> None:
        result = tower_tok.text_to_token("hello world")
        assert result.tokens.shape[0] == 1

    def test_hidden_size_is_zero(self, tower_tok: TextTower) -> None:
        assert tower_tok.hidden_size == 0


# ── Full model tests (GPU required) ─────────────────────────────────


@pytest.fixture(scope="module")
def tower_full(
    cfg: dict[str, Any], torch_dtype: torch.dtype, device: torch.device
) -> TextTower:
    tower = TextTower(
        model_name=cfg["model_name"],
        only_tokenizer=False,
        use_attn_mask=cfg["use_attn_mask"],
        prompt_max_tokens=cfg["prompt_max_tokens"],
        use_last_hidden_state=cfg["use_last_hidden_state"],
        torch_dtype=torch_dtype,
        skip_text_cleaning=cfg["skip_text_cleaning"],
    )
    return tower.to(device)


@pytest.mark.integration
@pytest.mark.gpu
class TestTextTowerFullModel:

    def test_hidden_size_positive(self, tower_full: TextTower) -> None:
        assert tower_full.hidden_size > 0

    def test_forward_shapes(
        self, tower_full: TextTower, cfg: dict[str, Any]
    ) -> None:
        out: dict[str, torch.Tensor] = tower_full(SAMPLE_TEXTS)
        max_tokens: int = cfg["prompt_max_tokens"]
        assert out["text_embed"].shape == (
            len(SAMPLE_TEXTS),
            max_tokens,
            tower_full.hidden_size,
        )

    def test_forward_attention_mask(self, tower_full: TextTower) -> None:
        out: dict[str, torch.Tensor] = tower_full(SAMPLE_TEXTS)
        assert "attention_mask" in out
        assert out["attention_mask"].shape[0] == len(SAMPLE_TEXTS)

    def test_forward_no_nan_inf(self, tower_full: TextTower) -> None:
        out: dict[str, torch.Tensor] = tower_full(SAMPLE_TEXTS)
        embed: torch.Tensor = out["text_embed"]
        assert not torch.isnan(embed).any(), "NaN detected in text embeddings"
        assert not torch.isinf(embed).any(), "Inf detected in text embeddings"
