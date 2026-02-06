"""Contrastive Flow Matching Algorithm (https://arxiv.org/abs/2506.05350).

This algorithm adds a contrastive loss term that encourages the denoiser to produce
predictions that are different from the predictions for other samples in the batch.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

from dataset.constants import BatchKeys

log = logging.getLogger(__name__)


class ContrastiveFlowMatching(Algorithm):
    """
    Contrastive Flow Matching Algorithm.

    Adds a contrastive loss term that encourages diversity in predictions across
    the batch by penalizing similarity to rolled targets.

    The algorithm:
    1. During INIT event: wraps model.loss() method to inject contrastive loss
    2. Contrastive loss computed as: -lambda_weight * MSE(prediction, rolled_target)
    3. Rolled target is batch-shifted target (target[1:], target[0])

    Args:
        lambda_weight: Weight for contrastive loss term (default: 0.1)

    Example:
        In your YAML config:

        algorithms:
          contrastive_flow_matching:
            _target_: algorithm.contrastive_flow_matching.ContrastiveFlowMatching
            lambda_weight: 0.1
    """

    def __init__(self, lambda_weight: float = 0.1):
        super().__init__()
        self.lambda_weight = lambda_weight

    def match(self, event: Event, state: State) -> bool:
        """Match only INIT event for setup."""
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """
        Apply algorithm logic during INIT event:
        Wrap model.loss() method to inject contrastive loss computation.
        """
        if event == Event.INIT:
            log.info(f"ContrastiveFlowMatching: Initializing (lambda_weight={self.lambda_weight})")

            # Wrap the model's loss method to inject contrastive loss
            self._wrap_loss_method(state)

            # Log hyperparameters
            logger.log_hyperparameters({
                "contrastive_flow_matching/lambda_weight": self.lambda_weight,
            })

    def _wrap_loss_method(self, state: State) -> None:
        """
        Wrap the model's loss() method to inject contrastive loss computation.

        The original loss method computes MSE loss.
        Our wrapper adds contrastive loss on top.
        """
        original_loss_fn = state.model.loss
        lambda_weight = self.lambda_weight

        def augmented_loss(outputs: Dict[str, torch.Tensor], batch: Dict[BatchKeys, Any]) -> torch.Tensor:
            # Compute base loss (MSE)
            base_loss = original_loss_fn(outputs, batch)

            # Compute contrastive loss
            # Loss introduced in Contrastive Flow Matching https://arxiv.org/abs/2506.05350
            if outputs["target"].shape[0] > 1:
                rolled_target = outputs["target"].roll(1, dims=0)
                contrastive_loss = -lambda_weight * F.mse_loss(
                    outputs["prediction"], rolled_target, reduction="mean"
                )
                # Log contrastive loss separately
                state.model.logger.log_metrics({"loss/train/contrastive": contrastive_loss.detach().cpu()})

                # Return combined loss
                return base_loss + contrastive_loss
            else:
                # Batch size is 1, skip contrastive loss
                return base_loss

        # Replace the loss method
        state.model.loss = augmented_loss
        log.info("ContrastiveFlowMatching: Wrapped model.loss() method to inject contrastive loss computation")