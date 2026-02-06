# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Exponential Moving Average (EMA) classes and functions."""

# Copied from https://github.com/mosaicml/composer/blob/dev/composer/algorithms/ema/ema.py with minor modifications

from __future__ import annotations

import os
import contextlib
import itertools
import logging
from typing import Any, ContextManager, Dict, Optional

import torch
from torch.distributed.fsdp import FSDPModule

import composer.utils.misc as misc
from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger

from diffusion.models.ema import EMAModel

log = logging.getLogger(__name__)

__all__ = ["EMA", "compute_ema"]


def compute_ema(model: torch.nn.Module, ema_model: EMAModel, smoothing: float = 0.99) -> None:
    r"""Updates the weights of ``ema_model`` to be closer to the weights of ``model``
    according to an exponential weighted average. Weights are updated according to

    .. math::
        W_{ema_model}^{(t+1)} = smoothing\times W_{ema_model}^{(t)}+(1-smoothing)\times W_{model}^{(t)}

    The update to ``ema_model`` happens in place.

    The half life of the weights for terms in the average is given by

    .. math::
        t_{1/2} = -\frac{\log(2)}{\log(smoothing)}

    Therefore, to set smoothing to obtain a target half life, set smoothing according to

    .. math::
        smoothing = \exp\left[-\frac{\log(2)}{t_{1/2}}\right]

    Args:
        model (torch.nn.Module): the model containing the latest weights to use to update the moving average weights.
        ema_model (torch.nn.Module, EMAParameters): the model containing the moving average weights to be updated.
        smoothing (float, optional): the coefficient representing the degree to which older observations are kept.
            Must be in the interval :math:`(0, 1)`. Default: ``0.99``.

    Example:
        .. testcode::

                import composer.functional as cf
                from torchvision import models
                model = models.resnet50()
                ema_model = models.resnet50()
                cf.compute_ema(model, ema_model, smoothing=0.9)
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()

    model_context_manager = get_model_context_manager(model)

    with model_context_manager:
        with torch.no_grad():
            ema_params = ema_model.model.state_dict()
            for name, param in itertools.chain(model.named_parameters(), model.named_buffers()):
                if name in ema_params:
                    ema_params[name].copy_(ema_params[name] * smoothing + param.data * (1.0 - smoothing))


def get_model_context_manager(model: torch.nn.Module) -> ContextManager[None]:
    """Summons full params for FSDP, which is required to update sharded params."""
    fsdp1_enabled = misc.is_model_fsdp(model) and os.environ.get("FSDP_VERSION", "1") == "1"
    model_context_manager = contextlib.nullcontext()
    if fsdp1_enabled:
        model_context_manager = model.denoiser.summon_full_params(model.denoiser)
    return model_context_manager


class EMA(Algorithm):
    r"""Maintains a set of weights that follow the exponential moving average of the training model weights.

    Weights are updated according to

    .. math::
        W_{ema_model}^{(t+1)} = smoothing\times W_{ema_model}^{(t)}+(1-smoothing)\times W_{model}^{(t)}

    Where the smoothing is determined from ``half_life`` according to

    .. math::
        smoothing = \exp\left[-\frac{\log(2)}{t_{1/2}}\right]

    Model evaluation is done with the moving average weights, which can result in better generalization. Because of the
    ema weights, EMA can double the model's memory consumption. Note that this does not mean that the total memory
    required doubles, since stored activations and the optimizer state are not duplicated. EMA also uses a small
    amount of extra compute to update the moving average weights.

    Args:
        half_life (str, optional): The time string specifying the half life for terms in the average. A longer half
            life means old information is remembered longer, a shorter half life means old information is discarded
            sooner. A half life of ``0`` means no averaging is done, an infinite half life means no update is done.
            Currently only units of epoch ('ep') and batch ('ba'). Time must be an integer value in the units
            specified. Cannot be used if ``smoothing`` is also specified. Default: ``"1000ba"``.
        smoothing (float, optional): The coefficient representing the degree to which older observations are kept.
            Must be in the interval :math:`(0, 1)`. Cannot be used if ``half_life`` also specified. This value will
            not be adjusted if ``update_interval`` is changed. Default: ``None``.
        ema_start (str, optional): The time string denoting the amount of training completed before EMA begins.
            Currently only units of duration ('dur'), batch ('ba') and epoch ('ep') are supported.
            Default: ``'0.0dur'``.
        update_interval (str, optional): The time string specifying the period at which updates are done. For example,
            an ``update_interval='1ep'`` means updates are done every epoch, while ``update_interval='10ba'`` means
            updates are done once every ten batches. Units must match the units used to specify ``half_life`` if not
            using ``smoothing``. If not specified, ``update_interval`` will default to ``1`` in the units of
            ``half_life``, or ``"1ba"`` if ``smoothing`` is specified. Time must be an integer value in the units
            specified. Default: ``None``.

    Example:

        .. testcode::

            from composer.algorithms import EMA
            algorithm = EMA(half_life='1000ba', update_interval='1ba')
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(
        self,
        smoothing: float = 0.999,
        ema_start: str = "0.0dur",
        update_interval: Optional[str] = None,
    ):
        self.ema_weights_active = False
        self.ema_started = False
        self.serialized_attributes = ["ema_started"]
        self.smoothing = smoothing

        # Convert start time to a time object
        self.ema_start = Time.from_timestring(ema_start)

        self.update_interval = Time.from_timestring(update_interval)

        # Verify that the time strings have supported units.
        if self.update_interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f"Invalid time unit for parameter update_interval: " f"{self.update_interval.unit}")

        # Construct the appropriate matching events
        self.checkpoint_events = [Event.BATCH_CHECKPOINT, Event.EPOCH_CHECKPOINT]
        if self.update_interval.unit == TimeUnit.BATCH:
            self.update_event = Event.BATCH_END
        elif self.update_interval.unit == TimeUnit.EPOCH:
            self.update_event = Event.EPOCH_END

    def _should_start(self, state: State) -> bool:
        if self.ema_start.unit == TimeUnit.DURATION:
            current_time = state.get_elapsed_duration()
            if current_time is not None:
                should_start: bool = self.ema_start <= current_time
            else:
                should_start = False
        else:
            current_time = state.timestamp.get(self.ema_start.unit).value
            should_start = self.ema_start.value <= current_time

        return should_start

    def get_ema_model(self, state: State) -> EMAModel:
        """Returns the EMA model if it exists, else None."""
        # TODO unwrap DDP
        return state.model.ema_denoiser

    def match(self, event: Event, state: State) -> bool:
        # Always run on init
        if event == Event.INIT:
            return True

        ema_model = self.get_ema_model(state)
        # Check if ema should start running, and if so reinitialize models
        if event == self.update_event and ema_model.is_active is False and self._should_start(state):
            ema_model.copy_weights_from_source(state.model.denoiser)
            ema_model.is_active = True

        # Conditionally run on the update event if ema has started
        if event == self.update_event and ema_model.is_active:
            return bool(state.timestamp.get(self.update_interval.unit).value % self.update_interval.value == 0)

        return False

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if not isinstance(self.update_interval, Time):
            raise ValueError("self.update_interval must be of type Time.")
        if not isinstance(self.smoothing, float):
            raise ValueError("self.smoothing must be of type float.")

        ema_model = self.get_ema_model(state)

        if event == Event.INIT:
            # Create the models so that the checkpoints can be loaded
            ema_model.init_model(state.model.denoiser)

        if event in [Event.BATCH_END, Event.EPOCH_END]:
            # Update the ema model
            # check if ema_model has a model attribute could happen if the model is not initialized and the event INIT is not called
            if not hasattr(ema_model, "model"):
                ema_model.init_model(state.model.denoiser)
            compute_ema(state.model.denoiser, ema_model, smoothing=self.smoothing)

        if event == Event.EVAL_START:
            # Verify that the ema params are on the correct device.
            # Needed to ensure doing eval before training can resume correctly.
            state.model.ema_denoiser.move_params_to_device(destination_model=state.model)
            # Swap out the training model for the ema model in state
            self._ensure_ema_weights_active(state)

        if event == Event.EVAL_END:
            # Swap out the ema model for the training model in state
            self._ensure_training_weights_active(state)

        if event in self.checkpoint_events:
            # Make sure to have the training weights in the model before saving a checkpoint
            self._ensure_training_weights_active(state)

    def state_dict(self) -> Dict[str, Any]:
        """We just store the EMA status, not the EMA weights here."""
        state_dict: Dict[str, Any] = super().state_dict()
        for attribute_name in self.serialized_attributes:
            state_dict[attribute_name] = getattr(self, attribute_name)
        return state_dict
