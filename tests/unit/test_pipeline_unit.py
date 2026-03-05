"""Unit tests for Pipeline — safety-net tests before refactoring.

These tests mock heavy components (VAE, text tower, denoiser) and run on CPU.
They capture current behavior to ensure the refactoring doesn't break anything.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torchmetrics import MeanSquaredError

from prx.dataset.constants import BatchKeys
from prx.pipeline.composer_pipeline import (
    EMAModel,
    ForwardOutput,
    ImageSize,
    ModelInputs,
    Pipeline,
    PredictionType,
)
from prx.schedulers.scheduler import EulerDiscreteScheduler, SchedulerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_text_tower(hidden_size: int = 32, seq_len: int = 8, use_attn_mask: bool = True):
    """Mock text tower returning fixed-shape embeddings."""
    tower = MagicMock()
    tower.hidden_size = hidden_size
    tower.use_attn_mask = use_attn_mask
    tower.tokenizer_max_length = seq_len

    def encode(texts):
        bs = len(texts) if isinstance(texts, list) else 1
        result = {"text_embed": torch.randn(bs, seq_len, hidden_size)}
        if use_attn_mask:
            result["attention_mask"] = torch.ones(bs, seq_len, dtype=torch.bool)
        return result

    tower.side_effect = encode  # tower(texts) calls encode
    tower.__call__ = encode
    return tower


def _make_mock_vae(scale_factor: int = 2, channels: int = 4):
    """Mock VAE with identity-like behavior."""
    vae = MagicMock()
    vae.vae_scale_factor = scale_factor
    vae.vae_channels = channels
    vae.device = torch.device("cpu")
    vae.scale_latent = MagicMock(side_effect=lambda x: x * 0.5)
    vae.encode = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], channels, x.shape[2] // scale_factor, x.shape[3] // scale_factor))
    vae.decode = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 3, x.shape[2] * scale_factor, x.shape[3] * scale_factor))
    return vae


class _MockDenoiser(torch.nn.Module):
    """Mock denoiser returning random output of same shape as input."""

    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.randn(2, 2))

    def forward(self, *, image_latent, **kwargs):
        return torch.randn_like(image_latent)


def _make_scheduler():
    return EulerDiscreteScheduler(
        SchedulerConfig(num_train_timesteps=1000, prediction_type="flow_matching")
    )


@pytest.fixture
def pipeline():
    """CPU pipeline with mocked components."""
    text_tower = _make_mock_text_tower()
    vae = _make_mock_vae()

    p = Pipeline(
        denoiser=_MockDenoiser(),
        vae=vae,
        text_tower=text_tower,
        noise_scheduler=_make_scheduler(),
        inference_noise_scheduler=_make_scheduler(),
        p_drop_caption=0.0,
    )
    p.logger = MagicMock()
    p.train()
    return p


def _make_batch(pipeline, batch_size: int = 2):
    """Batch with precomputed embeddings."""
    seq_len = pipeline.text_tower.tokenizer_max_length
    hidden = pipeline.text_tower.hidden_size
    ch = pipeline.vae_channels
    sf = pipeline.vae_scale_factor
    return {
        BatchKeys.IMAGE_LATENT: torch.randn(batch_size, ch, 8, 8),
        BatchKeys.PROMPT_EMBEDDING: torch.randn(batch_size, seq_len, hidden),
        BatchKeys.PROMPT_EMBEDDING_MASK: torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# EMAModel tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEMAModel:
    def test_initial_state_is_inactive(self):
        ema = EMAModel()
        assert ema.is_active is False

    def test_activate(self):
        ema = EMAModel()
        ema.is_active = True
        assert ema.is_active is True

    def test_init_model_copies_and_freezes(self):
        source = torch.nn.Linear(4, 4)
        source.weight.data.fill_(1.0)
        ema = EMAModel()
        ema.init_model(source)
        # Frozen
        assert not ema.model.training
        for p in ema.model.parameters():
            assert not p.requires_grad
        # Weights copied
        assert torch.allclose(ema.model.weight, source.weight)

    def test_forward_delegates(self):
        source = torch.nn.Linear(4, 4)
        ema = EMAModel()
        ema.init_model(source)
        x = torch.randn(2, 4)
        out = ema(x)
        assert out.shape == (2, 4)

    def test_copy_weights_from_source(self):
        source = torch.nn.Linear(4, 4)
        ema = EMAModel()
        ema.init_model(source)
        # Change source
        source.weight.data.fill_(42.0)
        ema.copy_weights_from_source(source)
        assert torch.allclose(ema.model.weight, source.weight)

    def test_is_active_persistent_buffer(self):
        ema = EMAModel()
        ema.is_active = True
        sd = ema.state_dict()
        assert "_is_active" in sd

        ema2 = EMAModel()
        ema2.load_state_dict(sd)
        assert ema2.is_active is True


# ---------------------------------------------------------------------------
# Pipeline data types
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDataTypes:
    def test_model_inputs_values(self):
        assert ModelInputs.IMAGE == "image"
        assert ModelInputs.PROMPT_EMBEDS == "prompt_embeds"
        assert ModelInputs.PROMPT_ATTENTION_MASK == "prompt_attention_mask"

    def test_prediction_type_values(self):
        assert PredictionType.FLOW_MATCHING == "flow_matching"
        assert PredictionType.X_PREDICTION_FLOW_MATCHING == "x_prediction_flow_matching"

    def test_image_size_namedtuple(self):
        size = ImageSize(height=64, width=128)
        assert size.height == 64
        assert size.width == 128


# ---------------------------------------------------------------------------
# Pipeline init
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPipelineInit:
    def test_p_drop_caption_validation(self):
        with pytest.raises(ValueError, match="p_drop_caption"):
            Pipeline(
                denoiser=_MockDenoiser(),
                vae=_make_mock_vae(),
                text_tower=_make_mock_text_tower(),
                noise_scheduler=_make_scheduler(),
                inference_noise_scheduler=_make_scheduler(),
                p_drop_caption=-0.1,
            )

    def test_vae_scale_factor_stored(self, pipeline):
        assert pipeline.vae_scale_factor == 2
        assert pipeline.vae_channels == 4


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchProcessing:
    def test_get_image_latents_from_precomputed(self, pipeline):
        batch = {BatchKeys.IMAGE_LATENT: torch.randn(2, 4, 8, 8)}
        result = pipeline.get_image_latents(batch)
        assert ModelInputs.IMAGE_LATENT in result
        assert result[ModelInputs.IMAGE_LATENT].shape == (2, 4, 8, 8)

    def test_get_image_latents_from_image(self, pipeline):
        batch = {BatchKeys.IMAGE: torch.randn(2, 3, 16, 16)}
        result = pipeline.get_image_latents(batch)
        assert ModelInputs.IMAGE_LATENT in result

    def test_get_image_latents_raises_without_image(self, pipeline):
        with pytest.raises(ValueError, match="image_latent.*image"):
            pipeline.get_image_latents({})

    def test_get_text_embedding_from_precomputed(self, pipeline):
        batch = {
            BatchKeys.PROMPT_EMBEDDING: torch.randn(2, 8, 32),
            BatchKeys.PROMPT_EMBEDDING_MASK: torch.ones(2, 8, dtype=torch.bool),
        }
        result = pipeline.get_text_embedding(batch)
        assert ModelInputs.PROMPT_EMBEDS in result
        assert ModelInputs.PROMPT_ATTENTION_MASK in result

    def test_get_text_embedding_from_prompt(self, pipeline):
        batch = {BatchKeys.PROMPT: ["hello", "world"]}
        result = pipeline.get_text_embedding(batch)
        assert ModelInputs.PROMPT_EMBEDS in result

    def test_get_text_embedding_raises_without_prompt(self, pipeline):
        with pytest.raises(ValueError, match="prompt_embedding.*prompt"):
            pipeline.get_text_embedding({})

    def test_prepare_batch(self, pipeline):
        batch = _make_batch(pipeline)
        result = pipeline.prepare_batch(batch)
        assert ModelInputs.IMAGE_LATENT in result
        assert ModelInputs.PROMPT_EMBEDS in result

    def test_get_batch_size_from_batch(self, pipeline):
        batch = _make_batch(pipeline, batch_size=3)
        assert pipeline.get_batch_size_from_batch(batch) == 3

    def test_get_batch_size_raises_empty(self, pipeline):
        with pytest.raises(ValueError, match="Cannot infer batch size"):
            pipeline.get_batch_size_from_batch({})


# ---------------------------------------------------------------------------
# Pure math / noise methods
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPureMath:
    def test_get_latent_noise_shape(self, pipeline):
        latents = torch.randn(2, 4, 8, 8)
        noise = pipeline.get_latent_noise(latents)
        assert noise.shape == latents.shape

    def test_get_latent_noise_scaled(self):
        """Noise scale should affect the noise magnitude."""
        text_tower = _make_mock_text_tower()
        p = Pipeline(
            denoiser=_MockDenoiser(),
            vae=_make_mock_vae(),
            text_tower=text_tower,
            noise_scheduler=_make_scheduler(),
            inference_noise_scheduler=_make_scheduler(),
            noise_scale=2.0,
        )
        latents = torch.randn(100, 4, 8, 8)
        noise = p.get_latent_noise(latents)
        # noise_scale=2.0 should give ~2x std
        assert noise.std() > 1.5  # relaxed bound

    def test_get_target_flow_matching(self, pipeline):
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([0.5, 0.8])
        target = pipeline.get_target(latents, noise, timesteps)
        expected = noise - latents
        assert torch.allclose(target, expected)

    def test_get_target_x_prediction(self):
        scheduler = EulerDiscreteScheduler(
            SchedulerConfig(num_train_timesteps=1000, prediction_type="x_prediction_flow_matching")
        )
        p = Pipeline(
            denoiser=_MockDenoiser(),
            vae=_make_mock_vae(),
            text_tower=_make_mock_text_tower(),
            noise_scheduler=scheduler,
            inference_noise_scheduler=_make_scheduler(),
        )
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([0.5, 0.8])
        target = p.get_target(latents, noise, timesteps)
        assert torch.allclose(target, latents)

    def test_convert_x_to_v(self, pipeline):
        prediction = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        noised = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([0.5, 0.8])
        v_pred, v_target = pipeline.convert_x_to_v(prediction, target, noised, timesteps)
        assert v_pred.shape == prediction.shape
        assert v_target.shape == target.shape

    def test_convert_x_to_v_clamps_timestep(self, pipeline):
        """Timesteps near 0 should be clamped to 0.05."""
        prediction = torch.ones(1, 1, 1, 1)
        target = torch.ones(1, 1, 1, 1)
        noised = torch.ones(1, 1, 1, 1) * 2
        timesteps = torch.tensor([0.01])  # Below clamp threshold
        v_pred, _ = pipeline.convert_x_to_v(prediction, target, noised, timesteps)
        # With clamp to 0.05: (2 - 1) / 0.05 = 20
        assert torch.allclose(v_pred, torch.tensor([[[[20.0]]]]))

    def test_compute_cfg_guidance(self, pipeline):
        uncond = torch.zeros(2, 4, 8, 8)
        cond = torch.ones(2, 4, 8, 8)
        model_output = torch.cat([uncond, cond])
        result = pipeline._compute_cfg_guidance(model_output, guidance_scale=7.0)
        # uncond + 7.0 * (cond - uncond) = 0 + 7*1 = 7
        assert torch.allclose(result, torch.full((2, 4, 8, 8), 7.0))


# ---------------------------------------------------------------------------
# Image size utilities
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestImageSizeUtils:
    def test_get_image_size_from_batch_with_image(self, pipeline):
        batch = {BatchKeys.IMAGE: torch.randn(2, 3, 32, 64)}
        size = pipeline.get_image_size_from_batch(batch)
        assert size == ImageSize(height=32, width=64)

    def test_get_image_size_from_batch_with_latent(self, pipeline):
        batch = {BatchKeys.IMAGE_LATENT: torch.randn(2, 4, 8, 16)}
        size = pipeline.get_image_size_from_batch(batch)
        # vae_scale_factor=2, so 8*2=16, 16*2=32
        assert size == ImageSize(height=16, width=32)

    def test_get_image_latent_size_from_batch_with_latent(self, pipeline):
        batch = {BatchKeys.IMAGE_LATENT: torch.randn(2, 4, 8, 16)}
        size = pipeline.get_image_latent_size_from_batch(batch)
        assert size == ImageSize(height=8, width=16)

    def test_get_image_latent_size_from_batch_with_image(self, pipeline):
        batch = {BatchKeys.IMAGE: torch.randn(2, 3, 32, 64)}
        size = pipeline.get_image_latent_size_from_batch(batch)
        assert size == ImageSize(height=16, width=32)


# ---------------------------------------------------------------------------
# Conditioning / dropout
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConditioning:
    def test_drop_text_conditioning_with_zero_prob(self, pipeline):
        """p_drop=0.0 should not change embeddings."""
        pipeline.p_drop_caption = 0.0
        # Pre-set null embedding to avoid text tower call
        pipeline.null_prompt_embedding = torch.zeros(1, 8, 32)
        pipeline.null_prompt_mask = torch.ones(1, 8, dtype=torch.bool)

        embeds = torch.randn(4, 8, 32)
        original = embeds.clone()
        model_inputs = {
            ModelInputs.PROMPT_EMBEDS: embeds,
            ModelInputs.PROMPT_ATTENTION_MASK: torch.ones(4, 8, dtype=torch.bool),
        }
        pipeline.drop_text_conditioning(model_inputs)
        assert torch.allclose(model_inputs[ModelInputs.PROMPT_EMBEDS], original)

    def test_drop_text_conditioning_with_full_prob(self, pipeline):
        """p_drop=1.0 should replace all embeddings with null."""
        pipeline.p_drop_caption = 1.0
        null_embed = torch.zeros(1, 8, 32)
        pipeline.null_prompt_embedding = null_embed
        pipeline.null_prompt_mask = torch.ones(1, 8, dtype=torch.bool)

        embeds = torch.randn(4, 8, 32)
        model_inputs = {
            ModelInputs.PROMPT_EMBEDS: embeds,
            ModelInputs.PROMPT_ATTENTION_MASK: torch.ones(4, 8, dtype=torch.bool),
        }
        pipeline.drop_text_conditioning(model_inputs)
        # All should be replaced with null
        for i in range(4):
            assert torch.allclose(model_inputs[ModelInputs.PROMPT_EMBEDS][i], null_embed.squeeze(0))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoss:
    def test_loss_scalar(self, pipeline):
        outputs = {
            "prediction": torch.randn(2, 4, 8, 8),
            "target": torch.randn(2, 4, 8, 8),
        }
        loss = pipeline.loss(outputs, {})
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_logs_metric(self, pipeline):
        outputs = {
            "prediction": torch.randn(2, 4, 8, 8),
            "target": torch.randn(2, 4, 8, 8),
        }
        pipeline.loss(outputs, {})
        pipeline.logger.log_metrics.assert_called_once()
        call_args = pipeline.logger.log_metrics.call_args[0][0]
        assert "loss/train/mse" in call_args


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetrics:
    def test_get_metrics_train(self, pipeline):
        metrics = pipeline.get_metrics(is_train=True)
        assert "MeanSquaredError" in metrics

    def test_get_metrics_val(self, pipeline):
        metrics = pipeline.get_metrics(is_train=False)
        assert "MeanSquaredError" in metrics

    def test_update_metric_all_timesteps(self, pipeline):
        metric = MeanSquaredError()
        outputs = {
            "prediction": torch.randn(2, 4, 8, 8),
            "target": torch.randn(2, 4, 8, 8),
            "timesteps": torch.tensor([0.3, 0.7]),
        }
        pipeline.update_metric({}, outputs, metric)
        assert metric.compute() > 0

    def test_update_metric_binned(self, pipeline):
        metric = MeanSquaredError()
        metric.loss_bin = (0.0, 0.5)
        outputs = {
            "prediction": torch.randn(4, 4, 8, 8),
            "target": torch.randn(4, 4, 8, 8),
            "timesteps": torch.tensor([0.1, 0.3, 0.6, 0.9]),
        }
        pipeline.update_metric({}, outputs, metric)
        # Only timesteps 0.1 and 0.3 are in bin [0.0, 0.5)


# ---------------------------------------------------------------------------
# Initialize latents
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInitializeLatents:
    def test_with_init_latents(self, pipeline):
        init = torch.randn(2, 4, 4, 4)
        result = pipeline._initialize_latents(2, (16, 16), init, None, torch.device("cpu"))
        assert torch.equal(result, init)

    def test_with_single_seed(self, pipeline):
        result = pipeline._initialize_latents(2, (16, 16), None, 42, torch.device("cpu"))
        assert result.shape == (2, 4, 16 // 2, 16 // 2)  # vae_scale_factor=2

    def test_with_per_sample_seeds(self, pipeline):
        result = pipeline._initialize_latents(2, (16, 16), None, [42, 43], torch.device("cpu"))
        assert result.shape == (2, 4, 16 // 2, 16 // 2)

    def test_reproducibility(self, pipeline):
        r1 = pipeline._initialize_latents(2, (16, 16), None, 42, torch.device("cpu"))
        r2 = pipeline._initialize_latents(2, (16, 16), None, 42, torch.device("cpu"))
        assert torch.equal(r1, r2)
