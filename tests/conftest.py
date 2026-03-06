"""Root conftest: register custom markers and provide shared mock infrastructure."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn.functional as F
from composer.core import Time, Timestamp, TimeUnit
from torch import Tensor, nn

from prx.dataset.constants import BatchKeys
from prx.pipeline.pipeline import EMAModel


def pytest_configure(config: object) -> None:
    config.addinivalue_line("markers", "unit: fast unit tests (no GPU, no model downloads)")  # type: ignore[attr-defined]
    config.addinivalue_line("markers", "integration: integration tests (load real models, slow)")  # type: ignore[attr-defined]
    config.addinivalue_line("markers", "gpu: tests that require a CUDA GPU")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Mock denoiser components
# ---------------------------------------------------------------------------


class MockBlock(nn.Module):
    """Minimal transformer block that preserves shape (img: [B, N, C] -> [B, N, C]).

    Accepts the same kwargs as PRXBlock (img, txt, vec, pe, attention_mask)
    so TREAD/SPRINT hooks can register on it.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        img: Tensor,
        txt: Optional[Tensor] = None,
        vec: Optional[Tensor] = None,
        pe: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.linear(img)


class MockDenoiser(nn.Module):
    """Mock denoiser with ``blocks: nn.ModuleList`` and ``hidden_size``.

    This is the minimal interface that all algorithms require:
    - EMA: any nn.Module (for deepcopy / weight averaging)
    - REPA: ``blocks[layer_index]`` forward hook, ``hidden_size``
    - TREAD/SPRINT: ``blocks`` as nn.ModuleList, forward hooks
    """

    def __init__(self, hidden_size: int = 256, num_blocks: int = 12) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList([MockBlock(hidden_size) for _ in range(num_blocks)])
        self.final_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_latent: Tensor,
        timestep: Tensor,
        cross_attn_conditioning: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Simplified forward: patchify -> blocks -> project."""
        B, C, H, W = image_latent.shape
        # Flatten spatial dims to sequence: [B, C, H, W] -> [B, H*W, C_hidden]
        img = image_latent.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C_in]
        # Project to hidden size
        img = img[..., : self.hidden_size]
        if img.shape[-1] < self.hidden_size:
            img = F.pad(img, (0, self.hidden_size - img.shape[-1]))

        pe = kwargs.get("pe", None)
        for block in self.blocks:
            img = block(img=img, pe=pe)

        return self.final_proj(img)


# ---------------------------------------------------------------------------
# Mock pipeline
# ---------------------------------------------------------------------------


class MockLogger:
    """Lightweight logger that captures metrics and hyperparameters."""

    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {}
        self.hyperparameters: Dict[str, Any] = {}

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        self.hyperparameters.update(hyperparameters)


class MockPipeline(nn.Module):
    """Minimal pipeline.

    Provides: ``denoiser``, ``ema_denoiser``, ``forward(batch)``, ``loss(outputs, batch)``,
    and a ``logger`` attribute for algorithms that log metrics.
    """

    def __init__(self, hidden_size: int = 256, num_blocks: int = 12) -> None:
        super().__init__()
        self.denoiser = MockDenoiser(hidden_size=hidden_size, num_blocks=num_blocks)
        self.ema_denoiser = EMAModel()
        self.logger = MockLogger()
        self._hidden_size = hidden_size

    def forward(self, batch: Dict[BatchKeys, Any]) -> Dict[str, Tensor]:
        img_latent = batch[BatchKeys.IMAGE_LATENT]
        B = img_latent.shape[0]

        # Run through denoiser
        txt = batch.get(BatchKeys.PROMPT_EMBEDDING)
        timesteps = torch.rand(B, device=img_latent.device)
        prediction = self.denoiser(
            image_latent=img_latent,
            timestep=timesteps,
            cross_attn_conditioning=txt,
        )
        target = torch.randn_like(prediction)

        return {"prediction": prediction, "target": target, "timesteps": timesteps}

    def loss(self, outputs: Dict[str, Tensor], batch: Dict[BatchKeys, Any]) -> Tensor:
        loss = F.mse_loss(outputs["prediction"], outputs["target"])
        self.logger.log_metrics({"loss/train/mse": loss.detach().cpu()})
        return loss


# ---------------------------------------------------------------------------
# Mock Composer State
# ---------------------------------------------------------------------------


class MockState:
    """Lightweight stand-in for ``composer.core.State``.

    Provides the attributes algorithms access: ``model``, ``timestamp``,
    and helpers to advance the timestamp.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.timestamp = Timestamp()

    def get_elapsed_duration(self) -> Optional[Time]:
        return None

    def advance_batch(self) -> None:
        """Simulate one batch step advancing the timestamp."""
        self.timestamp = self.timestamp.to_next_batch(
            samples=4,
            tokens=0,
            duration=datetime.timedelta(seconds=1),
        )

    def advance_epoch(self) -> None:
        """Simulate advancing to next epoch."""
        self.timestamp = self.timestamp.to_next_epoch()


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def hidden_size() -> int:
    return 256


@pytest.fixture
def num_blocks() -> int:
    return 12


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def seq_length() -> int:
    """Number of spatial tokens after patchification (8x8)."""
    return 64


@pytest.fixture
def mock_denoiser(hidden_size: int, num_blocks: int) -> MockDenoiser:
    return MockDenoiser(hidden_size=hidden_size, num_blocks=num_blocks)


@pytest.fixture
def mock_pipeline(hidden_size: int, num_blocks: int) -> MockPipeline:
    return MockPipeline(hidden_size=hidden_size, num_blocks=num_blocks)


@pytest.fixture
def mock_batch(batch_size: int, hidden_size: int) -> Dict[BatchKeys, Any]:
    """Batch dict with IMAGE, IMAGE_LATENT, and PROMPT_EMBEDDING."""
    H = W = 8  # spatial dims of latent -> 64 tokens
    C = 16  # latent channels
    return {
        BatchKeys.IMAGE: torch.randn(batch_size, 3, H * 8, W * 8),
        BatchKeys.IMAGE_LATENT: torch.randn(batch_size, C, H, W),
        BatchKeys.PROMPT_EMBEDDING: torch.randn(batch_size, 77, hidden_size),
    }


@pytest.fixture
def mock_state(mock_pipeline: MockPipeline) -> MockState:
    return MockState(model=mock_pipeline)


@pytest.fixture
def mock_logger() -> MockLogger:
    return MockLogger()
