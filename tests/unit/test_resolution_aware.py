"""Tests for ResolutionAware algorithm.

Tests ResolutionEmbedder (nn.Module) and ResolutionAware (Algorithm)
for injecting resolution conditioning into the denoiser.
"""

import copy
from unittest.mock import MagicMock

import pytest
import torch
from composer.core import Event, State

from prx.algorithm.resolution_aware import ResolutionAware, ResolutionEmbedder
from prx.models.prx import PRX, PRXParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_denoiser():
    torch.manual_seed(0)
    model = PRX(PRXParams(
        hidden_size=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        in_channels=4,
        patch_size=1,
        context_in_dim=32,
        axes_dim=[8, 8],
        theta=10000,
    ))
    # Override zero-init of final_layer so outputs are non-trivial
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.01)
    torch.nn.init.normal_(model.final_layer.adaLN_modulation[1].weight, std=0.01)
    return model


@pytest.fixture
def vec_embedder():
    return ResolutionEmbedder(hidden_size=64, mode="vec")


@pytest.fixture
def token_embedder():
    return ResolutionEmbedder(hidden_size=64, mode="token")


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolutionEmbedderShapes:
    """Verify output shapes for vec and token modes."""

    def test_vec_mode_output_shape(self, vec_embedder):
        B = 4
        height = torch.full((B,), 32.0)
        width = torch.full((B,), 32.0)
        out = vec_embedder(height, width)
        assert out.shape == (B, 64)

    def test_token_mode_output_shape(self, token_embedder):
        B = 4
        height = torch.full((B,), 32.0)
        width = torch.full((B,), 32.0)
        out = token_embedder(height, width)
        assert out.shape == (B, 2, 64)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_vec_mode_different_batch_sizes(self, vec_embedder, batch_size):
        height = torch.full((batch_size,), 16.0)
        width = torch.full((batch_size,), 16.0)
        out = vec_embedder(height, width)
        assert out.shape == (batch_size, 64)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_token_mode_different_batch_sizes(self, token_embedder, batch_size):
        height = torch.full((batch_size,), 16.0)
        width = torch.full((batch_size,), 16.0)
        out = token_embedder(height, width)
        assert out.shape == (batch_size, 2, 64)


# ---------------------------------------------------------------------------
# 2. Logic tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolutionEmbedderLogic:
    """Verify correctness properties of the embedder."""

    def test_vec_output_not_nan(self, vec_embedder):
        out = vec_embedder(torch.tensor([32.0]), torch.tensor([32.0]))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_token_output_not_nan(self, token_embedder):
        out = token_embedder(torch.tensor([32.0]), torch.tensor([32.0]))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_inputs_produce_different_outputs_vec(self, vec_embedder):
        out_a = vec_embedder(torch.tensor([16.0]), torch.tensor([16.0]))
        out_b = vec_embedder(torch.tensor([64.0]), torch.tensor([32.0]))
        assert not torch.allclose(out_a, out_b), "Different (H, W) should produce different embeddings"

    def test_different_inputs_produce_different_outputs_token(self, token_embedder):
        out_a = token_embedder(torch.tensor([16.0]), torch.tensor([16.0]))
        out_b = token_embedder(torch.tensor([64.0]), torch.tensor([32.0]))
        assert not torch.allclose(out_a, out_b), "Different (H, W) should produce different embeddings"

    def test_gradients_flow_vec(self, vec_embedder):
        height = torch.tensor([32.0])
        width = torch.tensor([24.0])
        out = vec_embedder(height, width)
        out.sum().backward()
        for p in vec_embedder.parameters():
            assert p.grad is not None, "All parameters should receive gradients"
            assert p.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_gradients_flow_token(self, token_embedder):
        height = torch.tensor([32.0])
        width = torch.tensor([24.0])
        out = token_embedder(height, width)
        out.sum().backward()
        for p in token_embedder.parameters():
            assert p.grad is not None, "All parameters should receive gradients"
            assert p.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_vec_output_matches_hidden_size(self):
        for hidden_size in [32, 64, 128]:
            emb = ResolutionEmbedder(hidden_size=hidden_size, mode="vec")
            out = emb(torch.tensor([16.0]), torch.tensor([16.0]))
            assert out.shape[-1] == hidden_size

    def test_token_mode_produces_distinct_tokens_for_asymmetric_input(self, token_embedder):
        """For asymmetric H != W, the h_token and w_token should differ."""
        out = token_embedder(torch.tensor([64.0]), torch.tensor([32.0]))
        h_token = out[:, 0, :]  # [1, hidden_size]
        w_token = out[:, 1, :]  # [1, hidden_size]
        assert not torch.allclose(h_token, w_token), (
            "h_token and w_token should differ for asymmetric (H, W)"
        )


# ---------------------------------------------------------------------------
# 3. Hook tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolutionAwareHooks:
    """Verify that hooks correctly modify denoiser behavior."""

    def _run_denoiser(self, denoiser, B=2, H=8, W=8, seq_len=4):
        """Run a forward pass through the denoiser with random inputs."""
        image_latent = torch.randn(B, 4, H, W)
        timestep = torch.rand(B)
        prompt_embeds = torch.randn(B, seq_len, 32)
        prompt_mask = torch.ones(B, seq_len)
        return denoiser(
            image_latent=image_latent,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
        )

    def test_vec_mode_changes_output(self, tiny_denoiser):
        """With vec hooks, denoiser output should differ from baseline."""
        torch.manual_seed(42)
        B, H, W = 2, 8, 8
        image_latent = torch.randn(B, 4, H, W)
        timestep = torch.rand(B)
        prompt_embeds = torch.randn(B, 4, 32)
        prompt_mask = torch.ones(B, 4)

        # Baseline: no hooks
        tiny_denoiser.eval()
        with torch.no_grad():
            baseline = tiny_denoiser(
                image_latent=image_latent,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
            ).clone()

        # Apply algorithm hooks
        algo = ResolutionAware(mode="vec")
        model = MagicMock()
        model.denoiser = tiny_denoiser
        algo.add_new_pipeline_modules(model)
        algo._register_hooks_on_denoiser(tiny_denoiser)

        with torch.no_grad():
            hooked = tiny_denoiser(
                image_latent=image_latent,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
            )

        assert not torch.allclose(baseline, hooked, atol=1e-6), (
            "Vec mode hooks should change the denoiser output"
        )

    def test_token_mode_extends_sequence(self, tiny_denoiser):
        """Token mode should prepend 2 tokens to the text sequence inside blocks."""
        algo = ResolutionAware(mode="token")
        model = MagicMock()
        model.denoiser = tiny_denoiser
        algo.add_new_pipeline_modules(model)
        algo._register_hooks_on_denoiser(tiny_denoiser)

        B, H, W, seq_len = 2, 8, 8, 4
        image_latent = torch.randn(B, 4, H, W)
        timestep = torch.rand(B)
        prompt_embeds = torch.randn(B, seq_len, 32)
        prompt_mask = torch.ones(B, seq_len)

        # Capture txt length inside a block via a test hook
        captured_txt_lengths = []

        def capture_hook(module, args, kwargs):
            # PRXBlock.forward receives txt as the second positional arg
            txt = args[1] if len(args) > 1 else kwargs.get("txt")
            if txt is not None:
                captured_txt_lengths.append(txt.shape[1])

        handle = tiny_denoiser.blocks[0].register_forward_pre_hook(capture_hook, with_kwargs=True)

        with torch.no_grad():
            tiny_denoiser(
                image_latent=image_latent,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
            )

        handle.remove()

        # The original txt after txt_in has seq_len tokens.
        # Token mode should prepend 2, so blocks should see seq_len + 2.
        assert len(captured_txt_lengths) > 0, "Block hook should have fired"
        assert captured_txt_lengths[0] == seq_len + 2, (
            f"Expected text sequence length {seq_len + 2}, got {captured_txt_lengths[0]}"
        )


# ---------------------------------------------------------------------------
# 4. Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolutionAwareIntegration:
    """End-to-end integration of ResolutionAware with a tiny denoiser."""

    def test_add_new_pipeline_modules_attaches_embedder(self, tiny_denoiser):
        algo = ResolutionAware(mode="vec")
        model = MagicMock()
        model.denoiser = tiny_denoiser
        algo.add_new_pipeline_modules(model)
        assert hasattr(tiny_denoiser, "resolution_embedder"), (
            "add_new_pipeline_modules should set denoiser.resolution_embedder"
        )
        assert isinstance(tiny_denoiser.resolution_embedder, ResolutionEmbedder)

    def test_embedder_params_in_model_parameters(self, tiny_denoiser):
        """When attached to a real nn.Module, embedder params should be in model.parameters()."""
        # Use the denoiser itself as the model container
        algo = ResolutionAware(mode="token")
        algo.add_new_pipeline_modules(tiny_denoiser)
        param_ids = {id(p) for p in tiny_denoiser.parameters()}
        embedder = tiny_denoiser.resolution_embedder
        for p in embedder.parameters():
            assert id(p) in param_ids, (
                "Embedder parameters should be discoverable via model.parameters()"
            )

    def test_match_returns_true_for_init_and_fit_start(self):
        algo = ResolutionAware(mode="vec")
        state = MagicMock(spec=State)
        assert algo.match(Event.INIT, state) is True
        assert algo.match(Event.FIT_START, state) is True
        assert algo.match(Event.BATCH_START, state) is False
        assert algo.match(Event.BATCH_END, state) is False
        assert algo.match(Event.EPOCH_START, state) is False

    def test_full_forward_vec_mode_no_error(self, tiny_denoiser):
        """Full forward pass with vec hooks registered should not raise."""
        algo = ResolutionAware(mode="vec")
        algo.add_new_pipeline_modules(tiny_denoiser)
        algo._register_hooks_on_denoiser(tiny_denoiser)

        B, H, W, seq_len = 2, 8, 8, 4
        out = tiny_denoiser(
            image_latent=torch.randn(B, 4, H, W),
            timestep=torch.rand(B),
            prompt_embeds=torch.randn(B, seq_len, 32),
            prompt_attention_mask=torch.ones(B, seq_len),
        )
        assert out.shape == (B, 4, H, W)
        assert not torch.isnan(out).any()

    def test_full_forward_token_mode_no_error(self, tiny_denoiser):
        """Full forward pass with token hooks registered should not raise."""
        algo = ResolutionAware(mode="token")
        algo.add_new_pipeline_modules(tiny_denoiser)
        algo._register_hooks_on_denoiser(tiny_denoiser)

        B, H, W, seq_len = 2, 8, 8, 4
        out = tiny_denoiser(
            image_latent=torch.randn(B, 4, H, W),
            timestep=torch.rand(B),
            prompt_embeds=torch.randn(B, seq_len, 32),
            prompt_attention_mask=torch.ones(B, seq_len),
        )
        assert out.shape == (B, 4, H, W)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 5. EMA interaction tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolutionAwareEMA:
    """Verify that EMA denoiser copies get resolution hooks."""

    def _make_inputs(self, B=2, H=8, W=8, seq_len=4):
        return dict(
            image_latent=torch.randn(B, 4, H, W),
            timestep=torch.rand(B),
            prompt_embeds=torch.randn(B, seq_len, 32),
            prompt_attention_mask=torch.ones(B, seq_len),
        )

    def test_deepcopy_includes_embedder(self, tiny_denoiser):
        """EMA's deepcopy of the denoiser should include the resolution_embedder."""
        algo = ResolutionAware(mode="vec")
        algo.add_new_pipeline_modules(tiny_denoiser)

        ema_copy = copy.deepcopy(tiny_denoiser)
        assert hasattr(ema_copy, "resolution_embedder"), (
            "deepcopy should preserve resolution_embedder on denoiser"
        )
        assert isinstance(ema_copy.resolution_embedder, ResolutionEmbedder)
        # Weights should be independent copies
        for p_orig, p_copy in zip(
            tiny_denoiser.resolution_embedder.parameters(),
            ema_copy.resolution_embedder.parameters(),
        ):
            assert p_orig.data_ptr() != p_copy.data_ptr()

    def test_ema_copy_with_hooks_produces_valid_output(self, tiny_denoiser):
        """EMA denoiser with hooks registered should produce non-noise output."""
        torch.manual_seed(42)
        algo = ResolutionAware(mode="vec")
        algo.add_new_pipeline_modules(tiny_denoiser)

        # Simulate EMA lifecycle: deepcopy, then register hooks on both
        ema_denoiser = copy.deepcopy(tiny_denoiser).eval().requires_grad_(False)
        algo._register_hooks_on_denoiser(tiny_denoiser)
        algo._register_hooks_on_denoiser(ema_denoiser)

        inputs = self._make_inputs()
        with torch.no_grad():
            out_train = tiny_denoiser(**inputs)
            out_ema = ema_denoiser(**inputs)

        # Both should be valid (not NaN/Inf)
        assert not torch.isnan(out_ema).any()
        assert not torch.isinf(out_ema).any()
        # With same weights, outputs should match
        assert torch.allclose(out_train, out_ema, atol=1e-5), (
            "EMA denoiser with same weights and hooks should match training denoiser"
        )

    def test_ema_copy_without_hooks_differs(self, tiny_denoiser):
        """EMA denoiser WITHOUT hooks produces different output (the bug scenario)."""
        torch.manual_seed(42)
        algo = ResolutionAware(mode="vec")
        algo.add_new_pipeline_modules(tiny_denoiser)

        ema_denoiser = copy.deepcopy(tiny_denoiser).eval().requires_grad_(False)
        # Only register hooks on training denoiser, NOT on ema
        algo._register_hooks_on_denoiser(tiny_denoiser)

        inputs = self._make_inputs()
        with torch.no_grad():
            out_train = tiny_denoiser(**inputs)
            out_ema = ema_denoiser(**inputs)

        # Without hooks, EMA output should differ (missing resolution conditioning)
        assert not torch.allclose(out_train, out_ema, atol=1e-5), (
            "EMA denoiser without resolution hooks should produce different output"
        )

    @pytest.mark.parametrize("mode", ["vec", "token"])
    def test_hooks_on_both_denoisers(self, tiny_denoiser, mode):
        """Both training and EMA denoisers should work with hooks for both modes."""
        torch.manual_seed(42)
        algo = ResolutionAware(mode=mode)
        algo.add_new_pipeline_modules(tiny_denoiser)

        ema_denoiser = copy.deepcopy(tiny_denoiser).eval().requires_grad_(False)
        algo._register_hooks_on_denoiser(tiny_denoiser)
        algo._register_hooks_on_denoiser(ema_denoiser)

        inputs = self._make_inputs()
        with torch.no_grad():
            out_train = tiny_denoiser(**inputs)
            out_ema = ema_denoiser(**inputs)

        assert out_train.shape == out_ema.shape
        assert not torch.isnan(out_ema).any()
        assert torch.allclose(out_train, out_ema, atol=1e-5)
