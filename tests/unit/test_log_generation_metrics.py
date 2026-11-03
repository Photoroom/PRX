"""Unit tests for the LogQualityMetrics callback (no model downloads)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torchmetrics.image.fid as tm_fid
from torch import nn

from prx.callbacks.log_generation_metrics import LogQualityMetrics
from prx.dataset.constants import BatchKeys


class _RecordingInception(nn.Module):
    """Stand-in for torchmetrics' InceptionV3 that records what FID feeds it."""

    seen: list[torch.Tensor] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        _RecordingInception.seen.append(imgs.clone())
        return imgs.float().flatten(1).mean(dim=1, keepdim=True).expand(-1, 2048).contiguous()


class _FakeModel:
    denoiser_device = torch.device("cpu")
    ema_denoiser = None
    denoiser = None

    def __init__(self, gen_images: torch.Tensor) -> None:
        self._gen_images = gen_images

    def get_image_size_from_batch(self, batch: dict) -> tuple[int, int]:
        return tuple(batch[BatchKeys.IMAGE].shape[-2:])

    def generate(self, **kwargs: object) -> torch.Tensor:
        return self._gen_images


@pytest.mark.unit
def test_fid_receives_unmodified_uint8_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tm_fid, "NoTrainInceptionV3", _RecordingInception)
    _RecordingInception.seen = []

    real = torch.rand(2, 3, 8, 8)
    gen = torch.rand(2, 3, 8, 8)
    state = SimpleNamespace(
        model=_FakeModel(gen),
        batch={BatchKeys.IMAGE: real},
        timestamp=SimpleNamespace(
            batch=SimpleNamespace(value=1), epoch=SimpleNamespace(value=0), sample=SimpleNamespace(value=2)
        ),
    )

    callback = LogQualityMetrics(
        frequency="1ba",
        guidance_scales=(1.0,),
        max_samples=100,
        compute_fid=True,
        compute_cmmd=False,
        compute_dino_mmd=False,
    )
    callback.eval_start(state, logger=None)
    callback.eval_batch_end(state, logger=None)

    expected_real = torch.clamp(real * 255.0, 0, 255).to(torch.uint8)
    expected_gen = torch.clamp(gen * 255.0, 0, 255).to(torch.uint8)
    assert len(_RecordingInception.seen) == 2
    assert torch.equal(_RecordingInception.seen[0], expected_real)
    assert torch.equal(_RecordingInception.seen[1], expected_gen)
