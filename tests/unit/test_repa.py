"""Tests for the REPA (Representation Alignment) algorithm."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
from composer.core import Event
from torch import Tensor, nn

from prx.algorithm.repa import MLP, REPA, REPALoss
from prx.dataset.constants import BatchKeys
from tests.conftest import MockDenoiser, MockLogger, MockPipeline, MockState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock encoder to avoid torch.hub downloads
# ---------------------------------------------------------------------------


class MockDinoEncoder(nn.Module):
    """Lightweight stand-in for DinoWrapper that returns random features."""

    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size_pixels = 16

    def forward(
        self, img: Tensor, denoiser_downsampling_ratio: float
    ) -> Tensor:
        B = img.shape[0]
        # Compute num patches as the real encoder would
        resize_factor = self.patch_size_pixels / denoiser_downsampling_ratio
        h = int(img.shape[-2] * resize_factor) // self.patch_size_pixels
        w = int(img.shape[-1] * resize_factor) // self.patch_size_pixels
        num_patches = h * w
        # Real DinoWrapper returns just the patch tokens tensor [B, num_patches, hidden_dim]
        return torch.randn(B, num_patches, self.hidden_dim, device=img.device, dtype=img.dtype)


@pytest.fixture
def mock_encoder() -> MockDinoEncoder:
    return MockDinoEncoder(hidden_dim=1024)


@pytest.fixture
def patch_repa_encoder(monkeypatch: pytest.MonkeyPatch, mock_encoder: MockDinoEncoder) -> MockDinoEncoder:
    """Patch REPALoss.build_encoder to return our mock encoder."""

    def _mock_build_encoder(self: Any, model: str) -> nn.Module:
        return mock_encoder

    monkeypatch.setattr(REPALoss, "build_encoder", _mock_build_encoder)
    return mock_encoder


# ---------------------------------------------------------------------------
# Unit tests: MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_mlp_shape(self) -> None:
        mlp = MLP(in_dim=256, hidden_dim=1024)
        x = torch.randn(2, 64, 256)
        out = mlp(x)
        assert out.shape == (2, 64, 1024)

    def test_mlp_layers(self) -> None:
        mlp = MLP(in_dim=128, hidden_dim=512)
        assert mlp.in_layer.in_features == 128
        assert mlp.in_layer.out_features == 512
        assert mlp.hidden_layer.in_features == 512
        assert mlp.out_layer.out_features == 512

    def test_mlp_gradient_flow(self) -> None:
        mlp = MLP(in_dim=32, hidden_dim=64)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Unit tests: REPALoss
# ---------------------------------------------------------------------------


class TestREPALoss:
    def test_init_creates_mlp(self, patch_repa_encoder: MockDinoEncoder) -> None:
        loss_mod = REPALoss(
            denoiser_hidden_dim=256,
            lambda_weight=0.5,
            layer_index=7,
            encoder="dinov3_vitl16",
            compile_encoder=False,
        )
        assert isinstance(loss_mod.mlp, MLP)
        assert loss_mod.mlp.in_layer.in_features == 256
        assert loss_mod.mlp.out_layer.out_features == 1024  # encoder hidden dim

    def test_forward_with_target_feature(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """When target_feature is provided directly, skip encoder."""
        loss_mod = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.5, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        )
        loss_mod = loss_mod.to(torch.bfloat16)
        # Simulate activations captured by hook
        loss_mod.activations = torch.randn(2, 64, 256, dtype=torch.bfloat16)
        target = torch.randn(2, 64, 1024, dtype=torch.bfloat16)

        loss = loss_mod(target_feature=target)
        assert loss.ndim == 0

    def test_forward_with_image(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """When image is provided, use encoder to compute target features."""
        loss_mod = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.5, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        )
        loss_mod = loss_mod.to(torch.bfloat16)
        # Simulate activations: [B, N_patches, denoiser_hidden]
        # For 64x64 image with 16px patches, that's 4*4=16 patches
        loss_mod.activations = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        image = torch.randn(2, 3, 64, 64, dtype=torch.bfloat16)

        loss = loss_mod(image=image)
        assert loss.ndim == 0

    def test_forward_no_activations_raises(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """Forward should fail when activations have not been captured."""
        loss_mod = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.5, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        )
        loss_mod = loss_mod.to(torch.bfloat16)
        # Do NOT set activations
        with pytest.raises(Exception):
            loss_mod(target_feature=torch.randn(2, 64, 1024, dtype=torch.bfloat16))

    def test_forward_no_input_raises(self, patch_repa_encoder: MockDinoEncoder) -> None:
        loss_mod = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.5, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        )
        loss_mod.activations = torch.randn(2, 64, 256)
        with pytest.raises(ValueError, match="Either target_feature or image"):
            loss_mod()

    def test_lambda_weight_scales_loss(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """Higher lambda should produce larger magnitude loss."""
        activations = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        target = torch.randn(2, 16, 1024, dtype=torch.bfloat16)

        loss_small = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.1, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        ).to(torch.bfloat16)
        loss_small.activations = activations.clone()

        loss_large = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=1.0, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        ).to(torch.bfloat16)
        loss_large.activations = activations.clone()

        # Use same MLP weights for fair comparison
        loss_large.mlp.load_state_dict(loss_small.mlp.state_dict())

        l_small = loss_small(target_feature=target).abs()
        l_large = loss_large(target_feature=target).abs()
        assert l_large > l_small

    def test_tread_visible_idx_selects_tokens(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """When tread_visible_idx is provided, target features should be gathered."""
        loss_mod = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.5, layer_index=7,
            encoder="dinov3_vitl16", compile_encoder=False,
        ).to(torch.bfloat16)

        B, N_visible = 2, 8
        N_full = 16
        loss_mod.activations = torch.randn(B, N_visible, 256, dtype=torch.bfloat16)
        target = torch.randn(B, N_full, 1024, dtype=torch.bfloat16)
        visible_idx = torch.arange(N_visible).unsqueeze(0).expand(B, -1)

        loss = loss_mod(
            target_feature=target,
            tread_original_num_tokens=N_full,
            tread_visible_idx=visible_idx,
        )
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Unit tests: hook registration
# ---------------------------------------------------------------------------


class TestREPAHookRegistration:
    def test_prepare_denoiser_adds_hook(self, patch_repa_encoder: MockDinoEncoder) -> None:
        loss_mod = REPALoss(
            denoiser_hidden_dim=256, lambda_weight=0.5, layer_index=3,
            encoder="dinov3_vitl16", compile_encoder=False,
        )
        denoiser = MockDenoiser(hidden_size=256, num_blocks=12)

        loss_mod.prepare_denoiser(denoiser)

        # Run a forward pass through the hooked block
        x = torch.randn(2, 10, 256)
        output = denoiser.blocks[3](img=x)

        # Activations should have been captured
        assert loss_mod.activations is not None
        assert loss_mod.activations.shape == output.shape


# ---------------------------------------------------------------------------
# Unit tests: REPA algorithm
# ---------------------------------------------------------------------------


class TestREPAAlgorithm:
    def test_match_only_init(self) -> None:
        repa = REPA(lambda_weight=0.5, layer_index=7)
        state = MockState(model=MockPipeline())
        assert repa.match(Event.INIT, state) is True
        assert repa.match(Event.BATCH_END, state) is False
        assert repa.match(Event.FIT_START, state) is False

    def test_add_new_pipeline_modules(self, patch_repa_encoder: MockDinoEncoder) -> None:
        repa = REPA(lambda_weight=0.5, layer_index=7, compile_encoder=False)
        pipeline = MockPipeline(hidden_size=256, num_blocks=12)

        repa.add_new_pipeline_modules(pipeline)

        assert hasattr(pipeline, "repa_loss")
        assert isinstance(pipeline.repa_loss, REPALoss)
        assert pipeline.repa_loss.layer_index == 7

    def test_add_modules_idempotent(self, patch_repa_encoder: MockDinoEncoder) -> None:
        repa = REPA(lambda_weight=0.5, layer_index=7, compile_encoder=False)
        pipeline = MockPipeline(hidden_size=256, num_blocks=12)

        repa.add_new_pipeline_modules(pipeline)
        first_id = id(pipeline.repa_loss)

        repa.add_new_pipeline_modules(pipeline)
        assert id(pipeline.repa_loss) == first_id

    def test_apply_init_wraps_loss(self, patch_repa_encoder: MockDinoEncoder) -> None:
        repa = REPA(lambda_weight=0.5, layer_index=7, compile_encoder=False)
        pipeline = MockPipeline(hidden_size=256, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        repa.add_new_pipeline_modules(pipeline)
        original_loss = pipeline.loss

        repa.apply(Event.INIT, state, logger)

        # Loss method should be replaced
        assert state.model.loss is not original_loss

    def test_apply_init_logs_hyperparameters(self, patch_repa_encoder: MockDinoEncoder) -> None:
        repa = REPA(lambda_weight=0.3, layer_index=5, compile_encoder=False)
        pipeline = MockPipeline(hidden_size=256, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        repa.add_new_pipeline_modules(pipeline)
        repa.apply(Event.INIT, state, logger)

        assert logger.hyperparameters["repa/lambda_weight"] == 0.3
        assert logger.hyperparameters["repa/layer_index"] == 5

    def test_apply_without_modules_raises(self) -> None:
        repa = REPA(lambda_weight=0.5, layer_index=7)
        pipeline = MockPipeline(hidden_size=256, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        # Skip add_new_pipeline_modules -> INIT should fail
        with pytest.raises(RuntimeError, match="REPA module not found"):
            repa.apply(Event.INIT, state, logger)


# ---------------------------------------------------------------------------
# Integration: REPA with forward + loss
# ---------------------------------------------------------------------------


class TestREPAIntegration:
    def test_full_forward_with_repa_loss(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """Forward pass should capture activations, loss should include REPA term."""
        repa = REPA(lambda_weight=0.5, layer_index=3, compile_encoder=False)
        # Use bfloat16 to match REPA's MLP dtype (production denoisers run in bf16)
        pipeline = MockPipeline(hidden_size=256, num_blocks=12).to(torch.bfloat16)
        state = MockState(model=pipeline)
        logger = MockLogger()

        repa.add_new_pipeline_modules(pipeline)
        repa.apply(Event.INIT, state, logger)

        batch = {
            BatchKeys.IMAGE: torch.randn(2, 3, 64, 64),
            BatchKeys.IMAGE_LATENT: torch.randn(2, 16, 8, 8, dtype=torch.bfloat16),
            BatchKeys.PROMPT_EMBEDDING: torch.randn(2, 77, 256, dtype=torch.bfloat16),
        }

        outputs = pipeline.forward(batch)

        # After forward, activations should be captured
        assert pipeline.repa_loss.activations is not None

        loss = state.model.loss(outputs, batch)
        assert loss.ndim == 0
        assert loss.requires_grad

        # REPA loss should be logged
        assert "loss/train/repa" in pipeline.logger.metrics

    def test_backward_flows_to_mlp(self, patch_repa_encoder: MockDinoEncoder) -> None:
        """Gradients from REPA loss should flow to the MLP parameters."""
        repa = REPA(lambda_weight=0.5, layer_index=3, compile_encoder=False)
        # Use bfloat16 to match REPA's MLP dtype
        pipeline = MockPipeline(hidden_size=256, num_blocks=12).to(torch.bfloat16)
        state = MockState(model=pipeline)
        logger = MockLogger()

        repa.add_new_pipeline_modules(pipeline)
        repa.apply(Event.INIT, state, logger)

        batch = {
            BatchKeys.IMAGE: torch.randn(2, 3, 64, 64),
            BatchKeys.IMAGE_LATENT: torch.randn(2, 16, 8, 8, dtype=torch.bfloat16),
            BatchKeys.PROMPT_EMBEDDING: torch.randn(2, 77, 256, dtype=torch.bfloat16),
        }

        outputs = pipeline.forward(batch)
        loss = state.model.loss(outputs, batch)
        loss.backward()

        # MLP should have gradients
        for name, p in pipeline.repa_loss.mlp.named_parameters():
            assert p.grad is not None, f"MLP parameter {name} has no gradient"
