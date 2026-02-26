"""Tests for the SPRINT algorithm (extends TREAD with dense-shallow fusion)."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
from composer.core import Event
from torch import nn

from prx.algorithm.sprint import SPRINT
from prx.dataset.constants import BatchKeys
from tests.conftest import MockLogger, MockPipeline, MockState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Unit tests: initialization
# ---------------------------------------------------------------------------


class TestSprintInit:
    def test_valid_init(self) -> None:
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.75, seed=42)
        assert s.fuse_name == "sprint_fuse"
        assert s.mask_name == "sprint_mask_token"
        assert s.learnable_mask is True

    def test_custom_names(self) -> None:
        s = SPRINT(
            route_start=2, route_end=8, routing_probability=0.5,
            fuse_name="my_fuse", mask_name="my_mask",
        )
        assert s.fuse_name == "my_fuse"
        assert s.mask_name == "my_mask"


# ---------------------------------------------------------------------------
# Unit tests: add_new_pipeline_modules
# ---------------------------------------------------------------------------


class TestSprintModuleRegistration:
    def test_adds_fusion_layer(self) -> None:
        """Fusion layer should be Linear(2*C, C)."""
        hidden_size = 64
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5)
        pipeline = MockPipeline(hidden_size=hidden_size, num_blocks=12)

        s.add_new_pipeline_modules(pipeline)

        assert hasattr(pipeline.denoiser, "sprint_fuse")
        fuse = pipeline.denoiser.sprint_fuse
        assert isinstance(fuse, nn.Linear)
        assert fuse.in_features == 2 * hidden_size
        assert fuse.out_features == hidden_size

    def test_adds_learnable_mask_token(self) -> None:
        """Learnable mask token should be an nn.Parameter of shape [1, 1, C]."""
        hidden_size = 64
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5, learnable_mask=True)
        pipeline = MockPipeline(hidden_size=hidden_size, num_blocks=12)

        s.add_new_pipeline_modules(pipeline)

        assert hasattr(pipeline.denoiser, "sprint_mask_token")
        mask = pipeline.denoiser.sprint_mask_token
        assert isinstance(mask, nn.Parameter)
        assert mask.shape == (1, 1, hidden_size)

    def test_adds_buffer_mask_token(self) -> None:
        """Non-learnable mask token should be a buffer, not a parameter."""
        hidden_size = 64
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5, learnable_mask=False)
        pipeline = MockPipeline(hidden_size=hidden_size, num_blocks=12)

        s.add_new_pipeline_modules(pipeline)

        mask = pipeline.denoiser.sprint_mask_token
        # Should not be in parameters (it's a buffer)
        param_names = [n for n, _ in pipeline.denoiser.named_parameters()]
        assert "sprint_mask_token" not in param_names
        # But should be in state dict (persistent buffer)
        assert "sprint_mask_token" in pipeline.denoiser.state_dict()

    def test_modules_included_in_optimizer(self) -> None:
        """Fusion layer params should be picked up by optimizer after registration."""
        hidden_size = 64
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5, learnable_mask=True)
        pipeline = MockPipeline(hidden_size=hidden_size, num_blocks=12)

        s.add_new_pipeline_modules(pipeline)

        optimizer = torch.optim.Adam(pipeline.parameters())
        param_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        fuse_count = sum(p.numel() for p in pipeline.denoiser.sprint_fuse.parameters())
        mask_count = pipeline.denoiser.sprint_mask_token.numel()

        # Optimizer should include fusion layer and mask token
        assert param_count >= fuse_count + mask_count

    def test_idempotent_registration(self) -> None:
        """Calling add_new_pipeline_modules twice should not add duplicate modules."""
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5)
        pipeline = MockPipeline(hidden_size=64, num_blocks=12)

        s.add_new_pipeline_modules(pipeline)
        fuse_id = id(pipeline.denoiser.sprint_fuse)

        s.add_new_pipeline_modules(pipeline)
        assert id(pipeline.denoiser.sprint_fuse) == fuse_id


# ---------------------------------------------------------------------------
# Unit tests: routing with fusion
# ---------------------------------------------------------------------------


class TestSprintFusion:
    def _setup_sprint(
        self, hidden_size: int = 32, num_blocks: int = 12, routing_prob: float = 0.5
    ) -> tuple[SPRINT, MockPipeline, MockState, MockLogger]:
        s = SPRINT(route_start=2, route_end=8, routing_probability=routing_prob, seed=42)
        pipeline = MockPipeline(hidden_size=hidden_size, num_blocks=num_blocks)
        s.add_new_pipeline_modules(pipeline)
        state = MockState(model=pipeline)
        logger = MockLogger()
        return s, pipeline, state, logger

    def test_dense_tokens_stashed(self) -> None:
        """Pre-hook should stash the full (dense) token sequence before routing."""
        s, pipeline, state, logger = self._setup_sprint()

        s.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        s.apply(Event.BATCH_START, state, logger)

        # Capture the input to route_start block to compare with stash
        captured_inputs: list[torch.Tensor] = []

        def capture_input(
            module: nn.Module, args: tuple[Any, ...], kwargs: Dict[str, Any]
        ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
            captured_inputs.append(kwargs["img"].clone())
            return args, kwargs

        handle = pipeline.denoiser.blocks[s.route_start].register_forward_pre_hook(
            capture_input, with_kwargs=True, prepend=True,
        )

        B, C, H, W = 2, 16, 8, 8
        pipeline.denoiser(
            image_latent=torch.randn(B, C, H, W),
            timestep=torch.rand(B),
        )
        handle.remove()

        assert "dense_tokens" in s._stash
        assert s._stash["dense_tokens"].shape[0] == B
        assert s._stash["dense_tokens"].shape[1] == H * W  # full sequence

        # Dense tokens should be the ORIGINAL full sequence seen at route_start
        torch.testing.assert_close(s._stash["dense_tokens"], captured_inputs[0])

        s.apply(Event.FIT_END, state, logger)

    def test_fusion_output_shape(self) -> None:
        """Output from SPRINT forward should have same shape as input sequence."""
        s, pipeline, state, logger = self._setup_sprint(hidden_size=32)

        s.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        s.apply(Event.BATCH_START, state, logger)

        B, C, H, W = 2, 16, 8, 8
        N = H * W
        output = pipeline.denoiser(
            image_latent=torch.randn(B, C, H, W),
            timestep=torch.rand(B),
        )

        # Output sequence length should match input
        assert output.shape[0] == B
        assert output.shape[1] == N

        s.apply(Event.FIT_END, state, logger)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSprintIntegration:
    def test_full_forward_backward(self) -> None:
        """Full forward + backward with SPRINT should work without errors."""
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        s.add_new_pipeline_modules(pipeline)
        state = MockState(model=pipeline)
        logger = MockLogger()

        s.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        s.apply(Event.BATCH_START, state, logger)

        batch = {
            BatchKeys.IMAGE_LATENT: torch.randn(4, 16, 8, 8),
            BatchKeys.PROMPT_EMBEDDING: torch.randn(4, 77, 32),
        }

        outputs = pipeline.forward(batch)
        loss = pipeline.loss(outputs, batch)
        assert loss.requires_grad
        loss.backward()

        # Verify gradients flow to fusion layer
        for p in pipeline.denoiser.sprint_fuse.parameters():
            assert p.grad is not None, "Fusion layer should receive gradients"

        s.apply(Event.FIT_END, state, logger)

    def test_eval_mode_no_routing_but_fusion(self) -> None:
        """During eval, routing_p=0 but fusion still applies."""
        s = SPRINT(
            route_start=2, route_end=8, routing_probability=0.75,
            seed=42, train_only=False,
        )
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        s.add_new_pipeline_modules(pipeline)
        state = MockState(model=pipeline)
        logger = MockLogger()

        s.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        s.apply(Event.BATCH_START, state, logger)

        # Put denoiser in eval mode (SPRINT checks _module.training in pre_route_start)
        pipeline.denoiser.eval()

        B, C, H, W = 2, 16, 8, 8
        N = H * W

        output = pipeline.denoiser(
            image_latent=torch.randn(B, C, H, W),
            timestep=torch.rand(B),
        )

        # Shape should still be preserved (fusion runs even with routing_p=0)
        assert output.shape[0] == B
        assert output.shape[1] == N

        s.apply(Event.FIT_END, state, logger)

    def test_multiple_forward_passes(self) -> None:
        """Multiple consecutive forward passes should work (state resets between passes)."""
        s = SPRINT(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        s.add_new_pipeline_modules(pipeline)
        state = MockState(model=pipeline)
        logger = MockLogger()

        s.apply(Event.FIT_START, state, logger)

        for step in range(3):
            state.advance_batch()
            s.apply(Event.BATCH_START, state, logger)

            B, C, H, W = 2, 16, 8, 8
            output = pipeline.denoiser(
                image_latent=torch.randn(B, C, H, W),
                timestep=torch.rand(B),
            )
            assert output.shape == (B, H * W, 32)

        s.apply(Event.FIT_END, state, logger)
