"""Tests for prx.rewards module.

Tests RewardModel wrapper, registry, and build_reward_model factory.
All imscore models are mocked since they require large downloads.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock imscore model
# ---------------------------------------------------------------------------

def _make_mock_imscore_model(embed_dim: int = 64):
    """Create a mock imscore model with a .score() method."""
    model = MagicMock(spec=torch.nn.Module)
    # Give it real parameters so RewardModel can freeze them
    real_param = torch.nn.Parameter(torch.randn(4, 4))
    model.parameters = MagicMock(return_value=iter([real_param]))
    model.named_parameters = MagicMock(return_value=iter([("weight", real_param)]))
    # Make it iterable for nn.Module registration
    model.modules = MagicMock(return_value=iter([model]))
    model.children = MagicMock(return_value=iter([]))

    def score_fn(images, prompts):
        B = images.shape[0]
        return torch.randn(B)

    model.score = MagicMock(side_effect=score_fn)
    return model


@pytest.fixture
def mock_imscore_model():
    return _make_mock_imscore_model()


# ---------------------------------------------------------------------------
# 1. TestRewardModel
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRewardModel:
    """Tests for the RewardModel wrapper."""

    def test_is_nn_module(self):
        from prx.rewards.base import RewardModel
        model = RewardModel(_make_mock_imscore_model())
        assert isinstance(model, torch.nn.Module)

    def test_forward_output_shape(self, mock_imscore_model):
        from prx.rewards.base import RewardModel
        model = RewardModel(mock_imscore_model)
        B = 3
        images = torch.rand(B, 3, 256, 256)
        prompts = ["prompt"] * B
        out = model(images, prompts)
        assert out.shape == (B,)

    def test_forward_output_finite(self, mock_imscore_model):
        from prx.rewards.base import RewardModel
        model = RewardModel(mock_imscore_model)
        images = torch.rand(4, 3, 128, 128)
        prompts = ["test"] * 4
        out = model(images, prompts)
        assert torch.isfinite(out).all()

    def test_forward_single_image(self, mock_imscore_model):
        from prx.rewards.base import RewardModel
        model = RewardModel(mock_imscore_model)
        images = torch.rand(1, 3, 256, 256)
        prompts = ["single"]
        out = model(images, prompts)
        assert out.shape == (1,)

    def test_no_grad_by_default(self, mock_imscore_model):
        """Default mode should not produce gradients."""
        from prx.rewards.base import RewardModel
        model = RewardModel(mock_imscore_model, differentiable=False)

        # Make score return a tensor that tracks grad
        def score_with_grad(images, prompts):
            return (images.sum(dim=(1, 2, 3)) * 0.01)

        mock_imscore_model.score = score_with_grad

        images = torch.rand(2, 3, 32, 32, requires_grad=True)
        out = model(images, ["a", "b"])
        assert not out.requires_grad

    def test_differentiable_mode(self, mock_imscore_model):
        """Differentiable mode should preserve gradients."""
        from prx.rewards.base import RewardModel
        model = RewardModel(mock_imscore_model, differentiable=True)

        def score_with_grad(images, prompts):
            return images.sum(dim=(1, 2, 3)) * 0.01

        mock_imscore_model.score = score_with_grad

        images = torch.rand(2, 3, 32, 32, requires_grad=True)
        out = model(images, ["a", "b"])
        assert out.requires_grad
        out.sum().backward()
        assert images.grad is not None

    def test_delegates_to_imscore_score(self, mock_imscore_model):
        """Forward should call the underlying imscore model's score()."""
        from prx.rewards.base import RewardModel
        model = RewardModel(mock_imscore_model)
        images = torch.rand(2, 3, 64, 64)
        prompts = ["a", "b"]
        model(images, prompts)
        mock_imscore_model.score.assert_called_once()


# ---------------------------------------------------------------------------
# 2. TestRewardRegistry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRewardRegistry:
    """Tests for the reward model registry."""

    def test_registry_has_expected_models(self):
        from prx.rewards import REWARD_MODELS
        expected = {"hpsv2", "pickscore", "image_reward", "mps", "clip_score", "aesthetic"}
        assert set(REWARD_MODELS.keys()) == expected

    def test_registry_entries_are_tuples(self):
        from prx.rewards import REWARD_MODELS
        for name, entry in REWARD_MODELS.items():
            assert isinstance(entry, tuple), f"{name} should map to a tuple"
            assert len(entry) == 3, f"{name} tuple should have 3 elements"
            module_path, class_name, pretrained_id = entry
            assert isinstance(module_path, str)
            assert isinstance(class_name, str)
            assert isinstance(pretrained_id, str)

    def test_unknown_model_raises(self):
        from prx.rewards import build_reward_model
        with pytest.raises(KeyError):
            build_reward_model("nonexistent")

    def test_build_reward_model_calls_from_pretrained(self):
        """build_reward_model should import the class and call from_pretrained."""
        from prx.rewards import build_reward_model
        from prx.rewards.base import RewardModel

        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = _make_mock_imscore_model()

        mock_module = MagicMock()
        mock_module.HPSv2 = mock_cls

        with patch("importlib.import_module", return_value=mock_module):
            result = build_reward_model("hpsv2")

        assert isinstance(result, RewardModel)
        mock_cls.from_pretrained.assert_called_once_with("RE-N-Y/hpsv21")
