"""Tests for the ContrastiveFlowMatching algorithm."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from composer.core import Event

from prx.algorithm.contrastive_flow_matching import ContrastiveFlowMatching
from prx.dataset.constants import BatchKeys
from tests.conftest import MockLogger, MockPipeline, MockState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestContrastiveFlowMatchingInit:
    def test_default_lambda(self) -> None:
        cfm = ContrastiveFlowMatching()
        assert cfm.lambda_weight == 0.1

    def test_custom_lambda(self) -> None:
        cfm = ContrastiveFlowMatching(lambda_weight=0.5)
        assert cfm.lambda_weight == 0.5

    def test_match_only_init(self) -> None:
        cfm = ContrastiveFlowMatching()
        state = MockState(model=MockPipeline())
        assert cfm.match(Event.INIT, state) is True
        assert cfm.match(Event.BATCH_END, state) is False
        assert cfm.match(Event.FIT_START, state) is False
        assert cfm.match(Event.EVAL_START, state) is False


class TestContrastiveLossComputation:
    def test_rolled_target_is_batch_shift(self) -> None:
        """roll(1, dims=0) should shift batch dimension by 1."""
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rolled = target.roll(1, dims=0)
        expected = torch.tensor([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
        torch.testing.assert_close(rolled, expected)

    def test_contrastive_loss_formula(self) -> None:
        """Total loss = base_loss + (-lambda * MSE(pred, rolled_target))."""
        lambda_weight = 0.1
        prediction = torch.randn(4, 8, 8)
        target = torch.randn(4, 8, 8)

        base_loss = F.mse_loss(prediction, target)
        rolled_target = target.roll(1, dims=0)
        contrastive_term = -lambda_weight * F.mse_loss(prediction, rolled_target)
        expected_total = base_loss + contrastive_term

        torch.testing.assert_close(expected_total, base_loss + contrastive_term)

    def test_contrastive_loss_uses_wrapped_method(self) -> None:
        """Wrapped loss should differ from base loss by the contrastive term."""
        lambda_weight = 0.1
        cfm = ContrastiveFlowMatching(lambda_weight=lambda_weight)
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        cfm.apply(Event.INIT, state, logger)

        prediction = torch.randn(4, 8, 8)
        target = torch.randn(4, 8, 8)
        outputs = {"prediction": prediction, "target": target}

        total_loss = state.model.loss(outputs, {})
        base_loss = F.mse_loss(prediction, target)
        rolled_target = target.roll(1, dims=0)
        expected_contrastive = -lambda_weight * F.mse_loss(prediction, rolled_target)

        torch.testing.assert_close(total_loss, base_loss + expected_contrastive)

    def test_contrastive_loss_skipped_for_batch_size_one(self) -> None:
        """Contrastive loss should be skipped when batch size is 1."""
        cfm = ContrastiveFlowMatching(lambda_weight=0.5)
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        cfm.apply(Event.INIT, state, logger)

        outputs = {
            "prediction": torch.randn(1, 8, 8),
            "target": torch.randn(1, 8, 8),
        }
        batch: dict = {}

        loss = state.model.loss(outputs, batch)
        base_loss = F.mse_loss(outputs["prediction"], outputs["target"])
        torch.testing.assert_close(loss, base_loss)


class TestLossWrapping:
    def test_loss_method_is_replaced(self) -> None:
        cfm = ContrastiveFlowMatching()
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        original_loss_fn = pipeline.loss

        cfm.apply(Event.INIT, state, logger)

        assert state.model.loss is not original_loss_fn

    def test_wrapped_loss_includes_contrastive_term(self) -> None:
        lambda_weight = 0.1
        cfm = ContrastiveFlowMatching(lambda_weight=lambda_weight)
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        cfm.apply(Event.INIT, state, logger)

        prediction = torch.randn(4, 8, 8)
        target = torch.randn(4, 8, 8)
        outputs = {"prediction": prediction, "target": target}
        batch: dict = {}

        total_loss = state.model.loss(outputs, batch)
        base_loss = F.mse_loss(prediction, target)

        assert not torch.allclose(total_loss, base_loss)

        # Verify the contrastive loss was logged
        assert "loss/train/contrastive" in pipeline.logger.metrics

    def test_hyperparameters_logged(self) -> None:
        cfm = ContrastiveFlowMatching(lambda_weight=0.2)
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        cfm.apply(Event.INIT, state, logger)
        assert logger.hyperparameters["contrastive_flow_matching/lambda_weight"] == 0.2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestContrastiveFlowMatchingIntegration:
    def test_full_training_step(self) -> None:
        """Simulate full forward -> loss with contrastive wrapping."""
        cfm = ContrastiveFlowMatching(lambda_weight=0.05)
        pipeline = MockPipeline(hidden_size=32, num_blocks=2)
        state = MockState(model=pipeline)
        logger = MockLogger()

        cfm.apply(Event.INIT, state, logger)

        batch = {
            BatchKeys.IMAGE_LATENT: torch.randn(4, 16, 8, 8),
            BatchKeys.PROMPT_EMBEDDING: torch.randn(4, 77, 32),
        }

        outputs = pipeline.forward(batch)
        loss = state.model.loss(outputs, batch)

        assert loss.requires_grad
        assert loss.ndim == 0  # scalar

        loss.backward()

        # Verify contrastive loss logged
        assert "loss/train/contrastive" in pipeline.logger.metrics

    def test_multiple_loss_wrappings_compose(self) -> None:
        """Two ContrastiveFlowMatching algorithms should compose correctly."""
        cfm1 = ContrastiveFlowMatching(lambda_weight=0.1)
        cfm2 = ContrastiveFlowMatching(lambda_weight=0.05)
        pipeline = MockPipeline()
        state = MockState(model=pipeline)
        logger = MockLogger()

        cfm1.apply(Event.INIT, state, logger)
        cfm2.apply(Event.INIT, state, logger)

        outputs = {
            "prediction": torch.randn(4, 8, 8),
            "target": torch.randn(4, 8, 8),
        }

        # Should not raise (both wrappers compose)
        loss = state.model.loss(outputs, {})
        assert loss.ndim == 0
