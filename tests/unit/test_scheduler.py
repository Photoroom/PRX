"""Unit tests for prx/schedulers/scheduler.py."""

import pytest
import torch

from prx.schedulers.scheduler import EulerDiscreteScheduler, SchedulerConfig


# ---------------------------------------------------------------------------
# SchedulerConfig defaults
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchedulerConfig:
    def test_defaults(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.num_train_timesteps == 1000
        assert cfg.prediction_type == "flow_matching"

    def test_custom_values(self) -> None:
        cfg = SchedulerConfig(num_train_timesteps=500, prediction_type="epsilon")
        assert cfg.num_train_timesteps == 500
        assert cfg.prediction_type == "epsilon"


# ---------------------------------------------------------------------------
# EulerDiscreteScheduler
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEulerDiscreteScheduler:
    def test_init_defaults(self) -> None:
        sched = EulerDiscreteScheduler()
        assert sched.config.prediction_type == "flow_matching"
        assert sched.config.num_train_timesteps == 1000
        assert sched.shift == 1.0
        assert sched.timesteps is None
        assert sched.sigmas is None

    def test_init_with_config(self) -> None:
        cfg = SchedulerConfig(num_train_timesteps=200, prediction_type="flow_matching")
        sched = EulerDiscreteScheduler(config=cfg, shift=3.0)
        assert sched.num_train_timesteps == 200
        assert sched.shift == 3.0

    # -- set_timesteps / sigma schedule --

    def test_set_timesteps_shapes(self) -> None:
        sched = EulerDiscreteScheduler()
        num_steps = 20
        sched.set_timesteps(num_steps)

        assert sched.timesteps is not None
        assert sched.sigmas is not None
        assert sched.timesteps.shape == (num_steps,)
        assert sched.sigmas.shape == (num_steps + 1,)

    def test_set_timesteps_range(self) -> None:
        """Timesteps should go from ~1.0 down towards 0."""
        sched = EulerDiscreteScheduler()
        sched.set_timesteps(10)

        assert sched.timesteps is not None
        # First timestep is largest (closest to 1.0)
        assert sched.timesteps[0] > sched.timesteps[-1]
        # Sigmas end at 0.0
        assert sched.sigmas[-1].item() == pytest.approx(0.0)
        # Sigmas start at 1.0 (no shift)
        assert sched.sigmas[0].item() == pytest.approx(1.0)

    def test_set_timesteps_with_shift(self) -> None:
        sched = EulerDiscreteScheduler(shift=3.0)
        sched.set_timesteps(10)

        assert sched.sigmas is not None
        # With shift > 1, all sigmas should still be in [0, 1] and monotonically decreasing
        for i in range(len(sched.sigmas) - 1):
            assert sched.sigmas[i] >= sched.sigmas[i + 1]
        assert sched.sigmas[-1].item() == pytest.approx(0.0)

    # -- add_noise --

    def test_add_noise_shape(self) -> None:
        sched = EulerDiscreteScheduler()
        bs, c, h, w = 2, 4, 8, 8
        x0 = torch.randn(bs, c, h, w)
        noise = torch.randn_like(x0)
        t = torch.tensor([0.5, 0.8])

        noisy = sched.add_noise(x0, noise, t)
        assert noisy.shape == (bs, c, h, w)

    def test_add_noise_at_zero(self) -> None:
        """At t=0, noisy sample should equal x0."""
        sched = EulerDiscreteScheduler()
        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        t = torch.zeros(2)

        noisy = sched.add_noise(x0, noise, t)
        assert torch.allclose(noisy, x0)

    def test_add_noise_at_one(self) -> None:
        """At t=1, noisy sample should equal pure noise."""
        sched = EulerDiscreteScheduler()
        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        t = torch.ones(2)

        noisy = sched.add_noise(x0, noise, t)
        assert torch.allclose(noisy, noise)

    # -- step --

    def test_step_output_shape(self) -> None:
        sched = EulerDiscreteScheduler()
        sched.set_timesteps(10)

        assert sched.timesteps is not None
        bs, seq, dim = 2, 16, 32
        model_output = torch.randn(bs, seq, dim)
        sample = torch.randn(bs, seq, dim)
        timestep = sched.timesteps[0]

        prev = sched.step(model_output, timestep, sample)
        assert prev.shape == (bs, seq, dim)

    def test_step_requires_set_timesteps(self) -> None:
        sched = EulerDiscreteScheduler()
        with pytest.raises(ValueError, match="Timesteps not set"):
            sched.step(torch.randn(1, 4), torch.tensor(0.5), torch.randn(1, 4))

    def test_step_zero_velocity_preserves_sample(self) -> None:
        """If velocity is zero, sample should not change."""
        sched = EulerDiscreteScheduler()
        sched.set_timesteps(5)
        assert sched.timesteps is not None

        sample = torch.randn(1, 8, 16)
        zero_vel = torch.zeros_like(sample)
        prev = sched.step(zero_vel, sched.timesteps[0], sample)
        assert torch.allclose(prev, sample)

    # -- sample_timesteps --

    def test_sample_timesteps_shape_and_range(self) -> None:
        sched = EulerDiscreteScheduler()
        ts = sched.sample_timesteps(64, device=torch.device("cpu"))
        assert ts.shape == (64,)
        assert ts.min() >= 0.0
        assert ts.max() <= 1.0

    def test_sample_timesteps_custom_range(self) -> None:
        sched = EulerDiscreteScheduler()
        ts = sched.sample_timesteps(32, device=torch.device("cpu"), timesteps_range=(0.2, 0.8))
        assert ts.shape == (32,)
        assert ts.min() >= 0.2 - 1e-6
        assert ts.max() <= 0.8 + 1e-6

    # -- init_noise_sigma --

    def test_init_noise_sigma(self) -> None:
        sched = EulerDiscreteScheduler()
        assert sched.init_noise_sigma == 1.0

    # -- shift_timesteps --

    def test_shift_timesteps_identity_when_shift_is_one(self) -> None:
        sched = EulerDiscreteScheduler(shift=1.0)
        t = torch.linspace(0, 1, 11)
        shifted = sched.shift_timesteps(t)
        assert torch.allclose(shifted, t)

    def test_shift_timesteps_endpoints(self) -> None:
        """Shift should preserve 0 and 1 endpoints."""
        sched = EulerDiscreteScheduler(shift=3.0)
        t = torch.tensor([0.0, 1.0])
        shifted = sched.shift_timesteps(t)
        assert shifted[0].item() == pytest.approx(0.0, abs=1e-6)
        assert shifted[1].item() == pytest.approx(1.0, abs=1e-6)
