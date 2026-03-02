"""Integration tests for the full training pipeline.

Two test classes:
- TestPipelineLightweight: tiny PRX + identity VAE + pre-computed embeddings (no text encoder)
- TestPipelineEndToEnd: tiny PRX + FLUX VAE + real T5Gemma-2B text encoder (GPU)
"""

from unittest.mock import MagicMock

import pytest
import torch

from prx.dataset.constants import BatchKeys
from prx.models.prx import PRX, PRXParams
from prx.models.text_tower import TextTower
from prx.models.vae_tower import VaeTower
from prx.pipeline.pipeline import Pipeline
from prx.schedulers.scheduler import EulerDiscreteScheduler, SchedulerConfig


def _tiny_prx(context_dim: int, in_channels: int = 3) -> PRX:
    """Create a minimal PRX model for testing."""
    return PRX(
        PRXParams(
            in_channels=in_channels,
            patch_size=1,
            context_in_dim=context_dim,
            hidden_size=64,
            mlp_ratio=2.0,
            num_heads=4,
            depth=2,
            axes_dim=[8, 8],
            theta=10_000,
        )
    )


def _make_scheduler() -> EulerDiscreteScheduler:
    return EulerDiscreteScheduler(
        SchedulerConfig(num_train_timesteps=1000, prediction_type="flow_matching")
    )


def _make_identity_vae() -> VaeTower:
    return VaeTower(
        model_name="identity",
        model_class="IdentityVAE",
        default_channels=3,
        torch_dtype=torch.bfloat16,
    )


def _make_flux_vae() -> VaeTower:
    return VaeTower(
        model_name="black-forest-labs/FLUX.1-dev",
        model_class="AutoencoderKL",
        default_channels=16,
        torch_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# Lightweight pipeline: pre-computed embeddings, no real text encoder needed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def lightweight_pipeline(device: torch.device) -> Pipeline:
    """Pipeline with tokenizer-only text tower and pre-computed embeddings."""
    text_tower = TextTower(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        only_tokenizer=True,
        use_attn_mask=True,
        prompt_max_tokens=16,
        use_last_hidden_state=True,
        torch_dtype=torch.bfloat16,
        skip_text_cleaning=True,
    )
    context_dim = 32
    model = Pipeline(
        denoiser=_tiny_prx(context_dim=context_dim, in_channels=3),
        vae=_make_identity_vae(),
        text_tower=text_tower,
        noise_scheduler=_make_scheduler(),
        inference_noise_scheduler=_make_scheduler(),
        p_drop_caption=0.0,
    )
    model = model.to(device)
    model.logger = MagicMock()  # normally set by Composer Trainer
    model.train()
    # Pre-init null embedding so caption dropout doesn't call the (missing) text encoder
    seq_len = text_tower.tokenizer_max_length
    model.null_prompt_embedding = torch.zeros(1, seq_len, context_dim, device=device, dtype=torch.bfloat16)
    model.null_prompt_mask = torch.ones(1, seq_len, device=device, dtype=torch.bool)
    return model


def _make_precomputed_batch(
    pipeline: Pipeline,
    device: torch.device,
    batch_size: int = 2,
) -> dict:
    """Batch with pre-computed embeddings (skips text encoder)."""
    context_dim = pipeline.denoiser.params.context_in_dim
    seq_len = pipeline.text_tower.tokenizer_max_length
    return {
        BatchKeys.IMAGE: torch.rand(batch_size, 3, 16, 16, device=device, dtype=torch.bfloat16),
        BatchKeys.PROMPT_EMBEDDING: torch.randn(
            batch_size, seq_len, context_dim, device=device, dtype=torch.bfloat16
        ),
        BatchKeys.PROMPT_EMBEDDING_MASK: torch.ones(
            batch_size, seq_len, device=device, dtype=torch.bool
        ),
    }


@pytest.mark.integration
@pytest.mark.gpu
class TestPipelineLightweight:
    """Forward + backward with pre-computed embeddings (no text encoder download)."""

    def test_forward_keys(self, lightweight_pipeline: Pipeline, device: torch.device) -> None:
        outputs = lightweight_pipeline.forward(_make_precomputed_batch(lightweight_pipeline, device))
        for key in ("prediction", "target", "timesteps", "noised_latents"):
            assert key in outputs

    def test_forward_shapes(self, lightweight_pipeline: Pipeline, device: torch.device) -> None:
        bs = 2
        outputs = lightweight_pipeline.forward(_make_precomputed_batch(lightweight_pipeline, device, bs))
        assert outputs["prediction"].shape == outputs["target"].shape
        assert outputs["prediction"].shape[0] == bs
        assert outputs["timesteps"].shape == (bs,)

    def test_loss_and_backward(self, lightweight_pipeline: Pipeline, device: torch.device) -> None:
        batch = _make_precomputed_batch(lightweight_pipeline, device)
        outputs = lightweight_pipeline.forward(batch)
        loss = lightweight_pipeline.loss(outputs, batch)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in lightweight_pipeline.denoiser.parameters()
        )
        assert has_grad, "denoiser should have non-zero gradients"

    def test_two_steps_loss_changes(self, lightweight_pipeline: Pipeline, device: torch.device) -> None:
        optimizer = torch.optim.AdamW(lightweight_pipeline.denoiser.parameters(), lr=1e-3)
        torch.manual_seed(42)
        losses: list[float] = []
        for _ in range(2):
            optimizer.zero_grad()
            outputs = lightweight_pipeline.forward(_make_precomputed_batch(lightweight_pipeline, device))
            loss = lightweight_pipeline.loss(outputs, _make_precomputed_batch(lightweight_pipeline, device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[0] != losses[1], "loss should change between steps"


# ---------------------------------------------------------------------------
# Full pipeline: real T5Gemma-2B text encoder + FLUX VAE, text passed as strings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_pipeline(device: torch.device) -> Pipeline:
    """Pipeline with real T5Gemma-2B text encoder and FLUX VAE."""
    text_tower = TextTower(
        model_name="google/t5gemma-2b-2b-ul2",
        only_tokenizer=False,
        use_attn_mask=True,
        prompt_max_tokens=32,
        use_last_hidden_state=True,
        torch_dtype=torch.bfloat16,
        skip_text_cleaning=False,
    )
    vae = _make_flux_vae()
    context_dim = text_tower.hidden_size
    model = Pipeline(
        denoiser=_tiny_prx(context_dim=context_dim, in_channels=vae.latent_channels),
        vae=vae,
        text_tower=text_tower,
        noise_scheduler=_make_scheduler(),
        inference_noise_scheduler=_make_scheduler(),
        p_drop_caption=0.1,
    )
    model = model.to(device)
    model.logger = MagicMock()  # normally set by Composer Trainer
    model.train()
    return model


def _make_text_batch(device: torch.device, batch_size: int = 2) -> dict:
    """Batch with raw text prompts (text encoder runs online).

    Images are 64x64 — the minimum for FLUX VAE (8x spatial compression → 8x8 latent).
    """
    return {
        BatchKeys.IMAGE: torch.rand(batch_size, 3, 64, 64, device=device, dtype=torch.bfloat16),
        BatchKeys.PROMPT: ["a photo of a cat", "an abstract painting"][:batch_size],
    }


@pytest.mark.integration
@pytest.mark.gpu
class TestPipelineEndToEnd:
    """End-to-end pipeline with real text encoder, FLUX VAE, and caption dropout."""

    def test_forward_with_text_prompts(self, full_pipeline: Pipeline, device: torch.device) -> None:
        batch = _make_text_batch(device)
        outputs = full_pipeline.forward(batch)

        assert outputs["prediction"].shape == outputs["target"].shape
        assert outputs["prediction"].shape[0] == 2
        assert not torch.isnan(outputs["prediction"]).any()

    def test_loss_and_backward_with_text(self, full_pipeline: Pipeline, device: torch.device) -> None:
        batch = _make_text_batch(device)
        outputs = full_pipeline.forward(batch)
        loss = full_pipeline.loss(outputs, batch)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in full_pipeline.denoiser.parameters()
        )
        assert has_grad
