"""Tests for the TREAD (Token Routing) algorithm."""

from __future__ import annotations

import pytest
import torch
from composer.core import Event

from prx.algorithm.tread import Tread
from prx.dataset.constants import BatchKeys
from tests.conftest import MockLogger, MockPipeline, MockState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Unit tests: initialization and validation
# ---------------------------------------------------------------------------


class TestTreadInit:
    def test_valid_init(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        assert t.route_start == 2
        assert t.route_end == 8
        assert t.routing_probability == 0.5
        assert t.train_only is True

    def test_invalid_range(self) -> None:
        with pytest.raises(AssertionError):
            Tread(route_start=5, route_end=3, routing_probability=0.5)

    def test_invalid_probability(self) -> None:
        with pytest.raises(AssertionError):
            Tread(route_start=2, route_end=8, routing_probability=1.5)


# ---------------------------------------------------------------------------
# Unit tests: event matching
# ---------------------------------------------------------------------------


class TestTreadMatch:
    def test_matches_fit_start(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5)
        state = MockState(model=MockPipeline())
        assert t.match(Event.FIT_START, state) is True

    def test_matches_fit_end(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5)
        state = MockState(model=MockPipeline())
        assert t.match(Event.FIT_END, state) is True

    def test_matches_batch_start(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5)
        state = MockState(model=MockPipeline())
        assert t.match(Event.BATCH_START, state) is True

    def test_matches_eval_when_train_only(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, train_only=True)
        state = MockState(model=MockPipeline())
        assert t.match(Event.EVAL_START, state) is True
        assert t.match(Event.EVAL_END, state) is True

    def test_no_match_eval_when_not_train_only(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, train_only=False)
        state = MockState(model=MockPipeline())
        assert t.match(Event.EVAL_START, state) is False
        assert t.match(Event.EVAL_END, state) is False

    def test_no_match_irrelevant_events(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5)
        state = MockState(model=MockPipeline())
        assert t.match(Event.INIT, state) is False
        assert t.match(Event.BATCH_END, state) is False


# ---------------------------------------------------------------------------
# Unit tests: deterministic seed computation
# ---------------------------------------------------------------------------


class TestTreadSeeding:
    def test_batch_seed_deterministic(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        seed1 = t._compute_batch_seed(step_idx=10, rank=0)
        seed2 = t._compute_batch_seed(step_idx=10, rank=0)
        assert seed1 == seed2

    def test_different_steps_different_seeds(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        seed1 = t._compute_batch_seed(step_idx=10, rank=0)
        seed2 = t._compute_batch_seed(step_idx=11, rank=0)
        assert seed1 != seed2

    def test_different_ranks_different_seeds(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        seed1 = t._compute_batch_seed(step_idx=10, rank=0)
        seed2 = t._compute_batch_seed(step_idx=10, rank=1)
        assert seed1 != seed2


# ---------------------------------------------------------------------------
# Unit tests: token sampling
# ---------------------------------------------------------------------------


class TestTreadSampling:
    def test_sample_indices_shapes(self) -> None:
        B, N = 4, 100
        routed_idx, visible_idx = Tread._sample_indices(
            batch_size=B, num_tokens=N, routing_probability=0.5,
            device=torch.device("cpu"), generator=torch.Generator().manual_seed(42),
        )
        num_routed = int(round(N * 0.5))
        assert routed_idx.shape == (B, num_routed)
        assert visible_idx.shape == (B, N - num_routed)

    def test_indices_cover_all_tokens(self) -> None:
        """Routed + visible indices should cover all tokens exactly once."""
        B, N = 2, 50
        routed_idx, visible_idx = Tread._sample_indices(
            batch_size=B, num_tokens=N, routing_probability=0.3,
            device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
        )
        for b in range(B):
            all_idx = torch.cat([routed_idx[b], visible_idx[b]]).sort().values
            expected = torch.arange(N)
            torch.testing.assert_close(all_idx, expected)

    def test_zero_routing_probability(self) -> None:
        """routing_probability=0 should route nothing."""
        B, N = 2, 50
        routed_idx, visible_idx = Tread._sample_indices(
            batch_size=B, num_tokens=N, routing_probability=0.0,
            device=torch.device("cpu"), generator=None,
        )
        assert routed_idx.shape == (B, 0)
        assert visible_idx.shape == (B, N)

    def test_high_routing_probability(self) -> None:
        """routing_probability=0.99 should route most tokens but keep at least 1 visible."""
        B, N = 2, 50
        routed_idx, visible_idx = Tread._sample_indices(
            batch_size=B, num_tokens=N, routing_probability=0.99,
            device=torch.device("cpu"), generator=torch.Generator().manual_seed(0),
        )
        # At least 1 token must remain visible
        assert visible_idx.shape[1] >= 1
        # All tokens still covered
        for b in range(B):
            all_idx = torch.cat([routed_idx[b], visible_idx[b]]).sort().values
            torch.testing.assert_close(all_idx, torch.arange(N))


# ---------------------------------------------------------------------------
# Unit tests: gather / scatter helpers
# ---------------------------------------------------------------------------


class TestTreadGatherScatter:
    def test_gather_batch_tokens(self) -> None:
        B, N, C = 2, 10, 8
        x = torch.randn(B, N, C)
        idx = torch.tensor([[0, 3, 7], [1, 4, 9]])  # [B, 3]
        gathered = Tread._gather_batch_tokens(x, idx)
        assert gathered.shape == (B, 3, C)
        # Verify correctness
        for b in range(B):
            for j in range(3):
                torch.testing.assert_close(gathered[b, j], x[b, idx[b, j]])

    def test_scatter_batch_tokens(self) -> None:
        B, N, C = 2, 10, 8
        out = torch.zeros(B, N, C)
        idx = torch.tensor([[0, 5], [3, 8]])  # [B, 2]
        vals = torch.ones(B, 2, C)
        result = Tread._scatter_batch_tokens(out, idx, vals)
        # Values should be at the specified indices
        for b in range(B):
            for j in range(2):
                torch.testing.assert_close(result[b, idx[b, j]], vals[b, j])

    def test_gather_scatter_roundtrip(self) -> None:
        """Gathering then scattering back should reconstruct the original tokens."""
        B, N, C = 2, 20, 16
        x = torch.randn(B, N, C)
        idx = torch.arange(N).unsqueeze(0).expand(B, -1)

        gathered = Tread._gather_batch_tokens(x, idx)
        out = torch.zeros_like(x)
        result = Tread._scatter_batch_tokens(out, idx, gathered)
        torch.testing.assert_close(result, x)


class TestTreadPositionalEncoding:
    def test_gather_pe_3d_shape(self) -> None:
        """PE shape [B, 1, N, D] should be gathered correctly along token dim."""
        B, N, D = 2, 20, 64
        pe = torch.randn(B, 1, N, D)
        idx = torch.tensor([[0, 5, 10], [2, 7, 15]])  # [B, 3]
        gathered = Tread._gather_positional_encoding(pe, idx)
        assert gathered.shape == (B, 1, 3, D)

    def test_gather_pe_higher_dim(self) -> None:
        """PE shape [B, 1, N, D1, D2] should be gathered correctly."""
        B, N, D1, D2 = 2, 20, 32, 2
        pe = torch.randn(B, 1, N, D1, D2)
        idx = torch.tensor([[0, 5], [2, 7]])
        gathered = Tread._gather_positional_encoding(pe, idx)
        assert gathered.shape == (B, 1, 2, D1, D2)

    def test_gather_pe_values_correct(self) -> None:
        B, N, D = 2, 10, 4
        pe = torch.randn(B, 1, N, D)
        idx = torch.tensor([[3], [7]])
        gathered = Tread._gather_positional_encoding(pe, idx)
        for b in range(B):
            torch.testing.assert_close(gathered[b, 0, 0], pe[b, 0, idx[b, 0]])


# ---------------------------------------------------------------------------
# Unit tests: hook registration and teardown
# ---------------------------------------------------------------------------


class TestTreadHooks:
    def test_fit_start_registers_hooks(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        t.apply(Event.FIT_START, state, logger)

        assert t._hooks_registered is True
        # Pre-hook on route_start + pre-hooks on middle blocks + post-hook on route_end
        expected_handles = 1 + (8 - 2) + 1  # route_start + mid blocks (3-8) + route_end post
        assert len(t._handles) == expected_handles

    def test_fit_end_removes_hooks(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        t.apply(Event.FIT_START, state, logger)
        assert t._hooks_registered is True

        t.apply(Event.FIT_END, state, logger)
        assert t._hooks_registered is False
        assert len(t._handles) == 0

    def test_eval_start_disables_routing(self) -> None:
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, train_only=True)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        t.apply(Event.FIT_START, state, logger)
        assert t._enabled is True

        t.apply(Event.EVAL_START, state, logger)
        assert t._enabled is False

        t.apply(Event.EVAL_END, state, logger)
        assert t._enabled is True

    def test_invalid_route_range_raises(self) -> None:
        t = Tread(route_start=0, route_end=15, routing_probability=0.5)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        with pytest.raises(ValueError, match="out of range"):
            t.apply(Event.FIT_START, state, logger)


# ---------------------------------------------------------------------------
# Integration: TREAD with forward pass through denoiser
# ---------------------------------------------------------------------------


class TestTreadIntegration:
    def test_forward_pass_preserves_shape(self) -> None:
        """After routing + scatter, output shape should match input shape."""
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        # Setup
        t.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        t.apply(Event.BATCH_START, state, logger)

        # Forward pass through denoiser
        B, C, H, W = 2, 16, 8, 8
        img_latent = torch.randn(B, C, H, W)
        timesteps = torch.rand(B)

        output = pipeline.denoiser(image_latent=img_latent, timestep=timesteps)
        # Output shape should match what the denoiser would produce without routing
        assert output.shape[0] == B
        assert output.shape[1] == H * W  # sequence length preserved

        # Cleanup
        t.apply(Event.FIT_END, state, logger)

    def test_routing_reduces_tokens_in_middle_blocks(self) -> None:
        """Verify that middle blocks see fewer tokens when routing is active."""
        hidden_size = 32
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=hidden_size, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        t.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        t.apply(Event.BATCH_START, state, logger)

        # Track input shapes to middle blocks via a hook
        middle_input_shapes: list[torch.Size] = []

        def capture_shape(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
            middle_input_shapes.append(kwargs.get("img", args[0] if args else None).shape)

        handle = pipeline.denoiser.blocks[5].register_forward_pre_hook(
            capture_shape, with_kwargs=True
        )

        B, C, H, W = 2, 16, 8, 8
        N = H * W
        pipeline.denoiser(
            image_latent=torch.randn(B, C, H, W),
            timestep=torch.rand(B),
        )

        handle.remove()
        t.apply(Event.FIT_END, state, logger)

        # Middle block should see fewer tokens than full N
        assert len(middle_input_shapes) == 1
        assert middle_input_shapes[0][1] < N

    def test_disabled_routing_passes_all_tokens(self) -> None:
        """When routing is disabled (eval mode), all tokens should pass through."""
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42, train_only=True)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        t.apply(Event.FIT_START, state, logger)
        t.apply(Event.EVAL_START, state, logger)  # Disable routing

        middle_input_shapes: list[torch.Size] = []

        def capture_shape(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
            middle_input_shapes.append(kwargs.get("img", args[0] if args else None).shape)

        handle = pipeline.denoiser.blocks[5].register_forward_pre_hook(
            capture_shape, with_kwargs=True
        )

        B, C, H, W = 2, 16, 8, 8
        N = H * W
        pipeline.denoiser(
            image_latent=torch.randn(B, C, H, W),
            timestep=torch.rand(B),
        )

        handle.remove()
        t.apply(Event.EVAL_END, state, logger)
        t.apply(Event.FIT_END, state, logger)

        # All tokens should pass through when disabled
        assert middle_input_shapes[0][1] == N

    def test_full_training_step_with_loss(self) -> None:
        """Full forward + loss with TREAD active should work end to end."""
        t = Tread(route_start=2, route_end=8, routing_probability=0.5, seed=42)
        pipeline = MockPipeline(hidden_size=32, num_blocks=12)
        state = MockState(model=pipeline)
        logger = MockLogger()

        t.apply(Event.FIT_START, state, logger)
        state.advance_batch()
        t.apply(Event.BATCH_START, state, logger)

        batch = {
            BatchKeys.IMAGE_LATENT: torch.randn(4, 16, 8, 8),
            BatchKeys.PROMPT_EMBEDDING: torch.randn(4, 77, 32),
        }

        outputs = pipeline.forward(batch)
        loss = pipeline.loss(outputs, batch)

        assert loss.requires_grad
        loss.backward()

        t.apply(Event.FIT_END, state, logger)
