"""Tests for FMPipeline — the pure-torch flow-matching pipeline.

Verifies that FMPipeline works independently of Composer.
"""

from unittest.mock import MagicMock

import pytest
import torch

from prx.dataset.constants import BatchKeys
from prx.pipeline.fm_pipeline import (
    EMAModel,
    FMPipeline,
    ForwardOutput,
    ImageSize,
    ModelInputs,
    PredictionType,
)
from prx.schedulers.scheduler import EulerDiscreteScheduler, SchedulerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.randn(2, 2))

    def forward(self, *, image_latent, **kwargs):
        return torch.randn_like(image_latent)


def _make_mock_text_tower(hidden_size=32, seq_len=8):
    tower = MagicMock()
    tower.hidden_size = hidden_size
    tower.use_attn_mask = True
    tower.tokenizer_max_length = seq_len

    def encode(texts):
        bs = len(texts) if isinstance(texts, list) else 1
        return {
            "text_embed": torch.randn(bs, seq_len, hidden_size),
            "attention_mask": torch.ones(bs, seq_len, dtype=torch.bool),
        }

    tower.__call__ = encode
    return tower


def _make_mock_vae(scale_factor=2, channels=4):
    vae = MagicMock()
    vae.vae_scale_factor = scale_factor
    vae.vae_channels = channels
    vae.device = torch.device("cpu")
    vae.scale_latent = MagicMock(side_effect=lambda x: x * 0.5)
    vae.encode = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], channels, x.shape[2] // scale_factor, x.shape[3] // scale_factor))
    vae.decode = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 3, x.shape[2] * scale_factor, x.shape[3] * scale_factor))
    return vae


def _make_scheduler():
    return EulerDiscreteScheduler(
        SchedulerConfig(num_train_timesteps=1000, prediction_type="flow_matching")
    )


@pytest.fixture
def fm_pipeline():
    """Pure-torch FMPipeline with mocked components — no Composer."""
    p = FMPipeline(
        denoiser=_MockDenoiser(),
        vae=_make_mock_vae(),
        text_tower=_make_mock_text_tower(),
        noise_scheduler=_make_scheduler(),
        inference_noise_scheduler=_make_scheduler(),
        p_drop_caption=0.0,
    )
    p.train()
    # Pre-init null embedding to avoid text tower call during conditioning dropout
    seq_len = p.text_tower.tokenizer_max_length
    hidden = p.text_tower.hidden_size
    p.null_prompt_embedding = torch.zeros(1, seq_len, hidden)
    p.null_prompt_mask = torch.ones(1, seq_len, dtype=torch.bool)
    return p


def _make_batch(pipeline, batch_size=2):
    seq_len = pipeline.text_tower.tokenizer_max_length
    hidden = pipeline.text_tower.hidden_size
    ch = pipeline.vae_channels
    return {
        BatchKeys.IMAGE_LATENT: torch.randn(batch_size, ch, 8, 8),
        BatchKeys.PROMPT_EMBEDDING: torch.randn(batch_size, seq_len, hidden),
        BatchKeys.PROMPT_EMBEDDING_MASK: torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Tests: FMPipeline works without Composer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFMPipelineStandalone:
    """FMPipeline should work as a pure nn.Module without Composer."""

    def test_is_nn_module(self, fm_pipeline):
        assert isinstance(fm_pipeline, torch.nn.Module)

    def test_not_composer_model(self, fm_pipeline):
        from composer.models import ComposerModel
        assert not isinstance(fm_pipeline, ComposerModel)

    def test_forward_returns_expected_keys(self, fm_pipeline):
        batch = _make_batch(fm_pipeline)
        outputs = fm_pipeline.forward(batch)
        assert "prediction" in outputs
        assert "target" in outputs
        assert "timesteps" in outputs
        assert "noised_latents" in outputs

    def test_forward_shapes(self, fm_pipeline):
        bs = 3
        batch = _make_batch(fm_pipeline, batch_size=bs)
        outputs = fm_pipeline.forward(batch)
        assert outputs["prediction"].shape[0] == bs
        assert outputs["prediction"].shape == outputs["target"].shape
        assert outputs["timesteps"].shape == (bs,)

    def test_has_no_loss_method(self, fm_pipeline):
        """FMPipeline should NOT have a loss method — that's Composer's job."""
        assert not hasattr(fm_pipeline, "loss")

    def test_has_no_eval_forward(self, fm_pipeline):
        """eval_forward is Composer-specific."""
        assert not hasattr(fm_pipeline, "eval_forward")

    def test_has_no_get_metrics(self, fm_pipeline):
        """Metrics are Composer-specific."""
        assert not hasattr(fm_pipeline, "get_metrics")


@pytest.mark.unit
class TestFMPipelineInference:
    """Test generate() on FMPipeline directly."""

    def test_generate_returns_tensor(self, fm_pipeline):
        fm_pipeline.eval()
        batch = _make_batch(fm_pipeline, batch_size=1)
        result = fm_pipeline.generate(
            batch=batch,
            image_size=(16, 16),
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=42,
            decode_latents=True,
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1

    def test_generate_no_decode(self, fm_pipeline):
        fm_pipeline.eval()
        batch = _make_batch(fm_pipeline, batch_size=1)
        result = fm_pipeline.generate(
            batch=batch,
            image_size=(16, 16),
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=42,
            decode_latents=False,
        )
        assert isinstance(result, torch.Tensor)
        # Latent shape: (bs, channels, h//sf, w//sf) = (1, 4, 8, 8)
        assert result.shape == (1, 4, 8, 8)


# ---------------------------------------------------------------------------
# Tests: Backward compatibility
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBackwardCompatibility:
    """Verify all old import paths still work."""

    def test_pipeline_alias(self):
        from prx.pipeline.composer_pipeline import Pipeline, ComposerFMPipeline
        assert Pipeline is ComposerFMPipeline

    def test_pipeline_from_init(self):
        from prx.pipeline import Pipeline, ComposerFMPipeline, FMPipeline
        assert Pipeline is ComposerFMPipeline
        assert issubclass(ComposerFMPipeline, FMPipeline)

    def test_types_from_pipeline_py(self):
        """Types should be importable from the old location."""
        from prx.pipeline.composer_pipeline import (
            EMAModel,
            ForwardOutput,
            ImageSize,
            ModelInputs,
            PredictionType,
        )

    def test_types_from_fm_pipeline_py(self):
        """Types should also be importable from the new location."""
        from prx.pipeline.fm_pipeline import (
            EMAModel,
            ForwardOutput,
            ImageSize,
            ModelInputs,
            PredictionType,
        )

    def test_composer_fm_pipeline_inherits_correctly(self):
        from prx.pipeline import ComposerFMPipeline, FMPipeline
        from composer.models import ComposerModel
        assert issubclass(ComposerFMPipeline, FMPipeline)
        assert issubclass(ComposerFMPipeline, ComposerModel)
        assert issubclass(ComposerFMPipeline, torch.nn.Module)

    def test_mro_order(self):
        """MRO should be: ComposerFMPipeline -> FMPipeline -> ComposerModel -> nn.Module."""
        from prx.pipeline import ComposerFMPipeline, FMPipeline
        from composer.models import ComposerModel
        mro = ComposerFMPipeline.__mro__
        fm_idx = mro.index(FMPipeline)
        composer_idx = mro.index(ComposerModel)
        module_idx = mro.index(torch.nn.Module)
        assert fm_idx < composer_idx < module_idx
