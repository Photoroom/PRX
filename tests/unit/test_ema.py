"""Tests for the EMA (Exponential Moving Average) algorithm."""

from __future__ import annotations

import copy

import pytest
import torch
from composer.core import Event

from prx.algorithm.ema import EMA, compute_ema
from tests.conftest import MockLogger, MockPipeline, MockState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Unit tests for compute_ema
# ---------------------------------------------------------------------------


class TestComputeEma:
    def test_weight_update_formula(self) -> None:
        """EMA update: W_ema = smoothing * W_ema + (1 - smoothing) * W_model."""
        model = torch.nn.Linear(4, 4, bias=False)
        pipeline = MockPipeline(hidden_size=4, num_blocks=2)
        pipeline.ema_denoiser.init_model(model)

        # Snapshot EMA weights before update
        ema_before = copy.deepcopy(pipeline.ema_denoiser.model.state_dict())

        smoothing = 0.9
        # Perturb model weights so they differ from EMA
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        compute_ema(model, pipeline.ema_denoiser, smoothing=smoothing)

        for name, ema_param in pipeline.ema_denoiser.model.named_parameters():
            expected = smoothing * ema_before[name] + (1 - smoothing) * model.state_dict()[name]
            torch.testing.assert_close(ema_param.data, expected)

    def test_smoothing_one_keeps_ema(self) -> None:
        """With smoothing=1.0 the EMA weights should not change."""
        model = torch.nn.Linear(4, 4, bias=False)
        pipeline = MockPipeline(hidden_size=4, num_blocks=2)
        pipeline.ema_denoiser.init_model(model)
        ema_before = copy.deepcopy(pipeline.ema_denoiser.model.state_dict())

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        compute_ema(model, pipeline.ema_denoiser, smoothing=1.0)

        for name, ema_param in pipeline.ema_denoiser.model.named_parameters():
            torch.testing.assert_close(ema_param.data, ema_before[name])

    def test_smoothing_zero_copies_model(self) -> None:
        """With smoothing=0.0 the EMA weights should exactly match the model."""
        model = torch.nn.Linear(4, 4, bias=False)
        pipeline = MockPipeline(hidden_size=4, num_blocks=2)
        pipeline.ema_denoiser.init_model(model)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 5)

        compute_ema(model, pipeline.ema_denoiser, smoothing=0.0)

        for name, ema_param in pipeline.ema_denoiser.model.named_parameters():
            torch.testing.assert_close(ema_param.data, model.state_dict()[name])


# ---------------------------------------------------------------------------
# Unit tests for EMA algorithm
# ---------------------------------------------------------------------------


class TestEMAAlgorithm:
    def test_init_defaults(self) -> None:
        ema = EMA(smoothing=0.999, update_interval="1ba")
        assert ema.smoothing == 0.999
        assert ema.ema_weights_active is False

    def test_match_init_event(self) -> None:
        """INIT event should always match."""
        ema = EMA(smoothing=0.99, update_interval="1ba")
        state = MockState(model=MockPipeline())
        assert ema.match(Event.INIT, state) is True

    def test_match_batch_end_before_start(self) -> None:
        """BATCH_END should not match before ema_start is reached and EMA initialized."""
        ema = EMA(smoothing=0.99, update_interval="1ba", ema_start="5ba")
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        # EMA model not yet active
        assert ema.match(Event.BATCH_END, state) is False

    def test_match_batch_end_after_start(self) -> None:
        """BATCH_END should match once ema_start is reached."""
        ema = EMA(smoothing=0.99, update_interval="1ba", ema_start="0ba")
        pipeline = MockPipeline()
        state = MockState(model=pipeline)

        # Manually activate EMA (simulating what INIT does)
        pipeline.ema_denoiser.init_model(pipeline.denoiser)
        pipeline.ema_denoiser.is_active = True

        # Advance to batch 1
        state.advance_batch()
        assert ema.match(Event.BATCH_END, state) is True

    def test_match_respects_update_interval(self) -> None:
        """BATCH_END should only match every N batches."""
        ema = EMA(smoothing=0.99, update_interval="2ba", ema_start="0ba")
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        pipeline.ema_denoiser.init_model(pipeline.denoiser)
        pipeline.ema_denoiser.is_active = True

        # Batch 1: should NOT match (1 % 2 != 0)
        state.advance_batch()
        assert ema.match(Event.BATCH_END, state) is False

        # Batch 2: should match (2 % 2 == 0)
        state.advance_batch()
        assert ema.match(Event.BATCH_END, state) is True

    def test_apply_init_creates_ema_model(self) -> None:
        """INIT event should create a deep copy of the denoiser in ema_denoiser."""
        ema = EMA(smoothing=0.99, update_interval="1ba")
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        assert not hasattr(pipeline.ema_denoiser, "model")

        ema.apply(Event.INIT, state, logger)

        assert hasattr(pipeline.ema_denoiser, "model")
        # EMA model should have same architecture
        ema_params = dict(pipeline.ema_denoiser.model.named_parameters())
        denoiser_params = dict(pipeline.denoiser.named_parameters())
        assert set(ema_params.keys()) == set(denoiser_params.keys())

    def test_apply_batch_end_updates_weights(self) -> None:
        """BATCH_END should update EMA weights towards model weights."""
        ema = EMA(smoothing=0.9, update_interval="1ba")
        pipeline = MockPipeline(hidden_size=32, num_blocks=2)
        state = MockState(model=pipeline)
        logger = MockLogger()

        # Initialize EMA
        ema.apply(Event.INIT, state, logger)
        ema_before = copy.deepcopy(pipeline.ema_denoiser.model.state_dict())

        # Perturb denoiser weights
        with torch.no_grad():
            for p in pipeline.denoiser.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        # Apply batch end update
        ema.apply(Event.BATCH_END, state, logger)

        # Weights should have moved towards the model
        for name in ema_before:
            if name in dict(pipeline.ema_denoiser.model.named_parameters()):
                assert not torch.allclose(
                    pipeline.ema_denoiser.model.state_dict()[name],
                    ema_before[name],
                ), f"EMA weight {name} was not updated"

    def test_invalid_update_interval_unit(self) -> None:
        """Only 'ba' and 'ep' units are supported for update_interval."""
        with pytest.raises(ValueError):
            EMA(smoothing=0.99, update_interval="1dur")

    def test_state_dict_serialization(self) -> None:
        """State dict should contain serialized attributes."""
        ema = EMA(smoothing=0.99, update_interval="1ba")
        ema.ema_started = True
        sd = ema.state_dict()
        assert "ema_started" in sd
        assert sd["ema_started"] is True


# ---------------------------------------------------------------------------
# Unit tests for EMAModel wrapper
# ---------------------------------------------------------------------------


class TestEMAModel:
    def test_init_model_creates_deepcopy(self) -> None:
        pipeline = MockPipeline(hidden_size=16, num_blocks=2)
        pipeline.ema_denoiser.init_model(pipeline.denoiser)

        # Modify original - EMA should not change
        with torch.no_grad():
            for p in pipeline.denoiser.parameters():
                p.fill_(999.0)

        for p in pipeline.ema_denoiser.model.parameters():
            assert not torch.all(p == 999.0), "EMA should be a deep copy"

    def test_is_active_toggle(self) -> None:
        pipeline = MockPipeline()
        assert pipeline.ema_denoiser.is_active is False
        pipeline.ema_denoiser.is_active = True
        assert pipeline.ema_denoiser.is_active is True

    def test_copy_weights_from_source(self) -> None:
        pipeline = MockPipeline(hidden_size=16, num_blocks=2)
        pipeline.ema_denoiser.init_model(pipeline.denoiser)

        # Modify denoiser
        with torch.no_grad():
            for p in pipeline.denoiser.parameters():
                p.fill_(42.0)

        pipeline.ema_denoiser.copy_weights_from_source(pipeline.denoiser)

        for name, p in pipeline.ema_denoiser.model.named_parameters():
            model_p = dict(pipeline.denoiser.named_parameters())[name]
            torch.testing.assert_close(p, model_p)


# ---------------------------------------------------------------------------
# Integration: EMA with manual event dispatch
# ---------------------------------------------------------------------------


class TestEMAIntegration:
    def test_full_ema_lifecycle(self) -> None:
        """Simulate a training loop: INIT -> N x (forward + BATCH_END) -> verify EMA diverges."""
        ema = EMA(smoothing=0.99, update_interval="1ba", ema_start="0ba")
        pipeline = MockPipeline(hidden_size=32, num_blocks=2)
        state = MockState(model=pipeline)
        logger = MockLogger()

        # INIT
        ema.apply(Event.INIT, state, logger)
        # Manually activate (match would do this on first BATCH_END when ema_start reached)
        pipeline.ema_denoiser.is_active = True

        # Snapshot initial EMA weights
        init_ema_weights = {
            name: p.clone() for name, p in pipeline.ema_denoiser.model.named_parameters()
        }

        # Simulate 5 training steps
        for _ in range(5):
            # Simulate optimizer step: perturb denoiser
            with torch.no_grad():
                for p in pipeline.denoiser.parameters():
                    p.add_(torch.randn_like(p) * 0.1)

            state.advance_batch()
            if ema.match(Event.BATCH_END, state):
                ema.apply(Event.BATCH_END, state, logger)

        # EMA weights should have diverged from initial
        for name, p in pipeline.ema_denoiser.model.named_parameters():
            assert not torch.allclose(p, init_ema_weights[name], atol=1e-6), (
                f"EMA weight {name} did not change after 5 steps"
            )

        # EMA weights should differ from current denoiser weights
        for name, p in pipeline.ema_denoiser.model.named_parameters():
            denoiser_p = dict(pipeline.denoiser.named_parameters())[name]
            assert not torch.allclose(p, denoiser_p, atol=1e-6), (
                f"EMA weight {name} matches denoiser exactly - EMA is not averaging"
            )
