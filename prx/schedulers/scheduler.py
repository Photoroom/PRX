from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class SchedulerConfig:
    """Base configuration for schedulers."""
    num_train_timesteps: int = 1000
    prediction_type: str = "flow_matching"  # "epsilon", "sample", "v_prediction", "flow_matching"


class BaseScheduler(ABC, nn.Module):
    """Abstract base class for noise schedulers."""

    def __init__(self, config: SchedulerConfig):
        super().__init__()
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps

    @abstractmethod
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to the original samples at given timesteps.

        Args:
            original_samples: Clean samples (x0)
            noise: Random noise
            timesteps: Timestep indices

        Returns:
            Noised samples (xt)
        """
        pass

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Perform one denoising step.

        Args:
            model_output: Model prediction at timestep t
            timestep: Current timestep
            sample: Current noisy sample (xt)
            generator: Random number generator

        Returns:
            Previous sample (xt-1)
        """
        pass

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the discrete timesteps for inference.

        Args:
            num_inference_steps: Number of diffusion steps
        """
        pass

   
    def shift_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.shift != 1.0:
            timesteps = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)
        return timesteps
    
    def sample_timesteps(
        self, size: int, device: torch.device, timesteps_range: tuple[float, float] | None = None
    ) -> torch.Tensor:
        """Sample random timesteps for training.

        Args:
            size: Number of timesteps to sample
            device: Device to create tensor on
            timesteps_range: Optional (min, max) range in [0, 1]

        Returns:
            Sampled timesteps
        """
        # log normal sampling
        def log_normal_sample(size: int, timestep_min: float = 0.0, timestep_max: float = 1.0) -> torch.Tensor:
            nt = torch.randn((size * 10,), device=device)  # normal
            timesteps = torch.sigmoid(nt)  # log normal
            timesteps = self.shift_timesteps(timesteps)  # shift
            timesteps = timesteps[(timesteps < timestep_max) & (timesteps >= timestep_min)][:size]  # reject
            return timesteps

        timestep_min, timestep_max = timesteps_range or [1e-3, 1.0]
        assert timestep_min >= 0
        assert timestep_max <= 1

        timesteps = torch.tensor([], device=device)  # reset timesteps
        while timesteps.size(0) < size:  # make sure we have enough valid timesteps
            timesteps = torch.cat([timesteps, log_normal_sample(size, timestep_min, timestep_max)])
            timesteps = timesteps[:size]

        # uniform sampling
        uniform_timesteps = timestep_min + torch.rand_like(timesteps, device=device) * (timestep_max - timestep_min)
        timesteps = torch.where(
            torch.rand(timesteps.size(0), device=device) < 0.1, uniform_timesteps, timesteps
        )
        return timesteps
    
    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Scale the model input (for compatibility with some schedulers).

        Args:
            sample: Current sample
            timestep: Current timestep

        Returns:
            Scaled sample (default: no scaling)
        """
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma for generation (default: 1.0)."""
        return 1.0


class EulerDiscreteScheduler(BaseScheduler):
    """Euler Discrete Scheduler for Flow Matching.

    This scheduler implements the Euler method for solving ODEs in flow matching models.
    Flow matching learns a conditional probability path from noise to data.

    The forward process interpolates linearly:
        x_t = (1 - t) * x_0 + t * noise

    The model predicts the velocity (vector field):
        v_t = noise - x_0

    During inference, we integrate the learned vector field using Euler's method.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        """Initialize Euler Discrete Scheduler.

        Args:
            config: Scheduler configuration
            num_train_timesteps: Number of training timesteps
            shift: Time shift parameter (default: 1.0, no shift)
        """
        if config is None:
            config = SchedulerConfig(
                num_train_timesteps=num_train_timesteps,
                prediction_type="flow_matching",
            )
        super().__init__(config)

        self.shift = shift
        self.timesteps = None
        self.sigmas = None

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise using flow matching interpolation.

        x_t = (1 - t) * x_0 + t * noise

        Args:
            original_samples: Clean samples (x0)
            noise: Random noise
            timesteps: Continuous timesteps in [0, 1]

        Returns:
            Noised samples (xt)
        """
        # Use continuous timesteps directly (already in [0, 1])
        t = timesteps.float()
        t = t.view(-1, 1, 1, 1)  # Reshape for broadcasting

        # Flow matching interpolation: x_t = (1-t)*x_0 + t*noise
        noisy_samples = (1 - t) * original_samples + t * noise

        return noisy_samples

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set discrete timesteps for inference using Euler method.

        Args:
            num_inference_steps: Number of diffusion steps
        """
        # Create linearly spaced timesteps from 1.0 to 0.0
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)

        # Apply time shift if needed (for some models)
        timesteps = self.shift_timesteps(timesteps)

        self.timesteps = timesteps[:-1]  # Remove the last 0.0
        self.sigmas = timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Perform one Euler step.

        For flow matching, the model predicts velocity v_t.
        If prediction_type is "x_prediction_flow_matching", converts x-prediction to v-prediction first.
        Euler update: x_{t-dt} = x_t - dt * v_t

        Args:
            model_output: Model prediction (velocity for flow matching, or x0 if prediction_type is "x_prediction_flow_matching")
            timestep: Current timestep (continuous time in [0, 1])
            sample: Current sample (xt)
            generator: Random number generator (unused for deterministic Euler)

        Returns:
            Previous sample (xt-dt)
        """
        # Find the timestep index
        if self.timesteps is None:
            raise ValueError("Timesteps not set. Call set_timesteps() first.")

        # Convert x-prediction to v-prediction if needed
        # From "Back to Basics: Let Denoising Generative Models Denoise"
        # https://arxiv.org/abs/2511.13720
        if self.config.prediction_type == "x_prediction_flow_matching":
            t = torch.clamp(timestep, min=0.05)
            model_output = (sample - model_output) / t

        # Get dt (time step size)
        timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(timestep_idx) == 0:
            # Handle floating point comparison issues
            timestep_idx = torch.argmin(torch.abs(self.timesteps - timestep))
        else:
            timestep_idx = timestep_idx[0]

        if timestep_idx < len(self.sigmas) - 1:
            dt = self.sigmas[timestep_idx] - self.sigmas[timestep_idx + 1]
        else:
            dt = self.sigmas[timestep_idx]

        # Euler step: x_{t-dt} = x_t - dt * v_t
        prev_sample = sample - dt * model_output

        return prev_sample

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """No scaling needed for Euler method."""
        return sample

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise sigma for flow matching is 1.0."""
        return 1.0

