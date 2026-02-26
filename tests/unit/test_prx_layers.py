"""Unit tests for prx/models/prx_layers.py and img2seq/seq2img from prx/models/prx.py."""

import pytest
import torch

from prx.models.prx_layers import get_image_ids, timestep_embedding
from prx.models.prx import img2seq, seq2img


# ---------------------------------------------------------------------------
# timestep_embedding
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTimestepEmbedding:
    def test_output_shape_even_dim(self) -> None:
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = timestep_embedding(t, dim=128)
        assert emb.shape == (3, 128)

    def test_output_shape_odd_dim(self) -> None:
        """Odd dim should be padded with a zero column."""
        t = torch.tensor([0.5])
        emb = timestep_embedding(t, dim=129)
        assert emb.shape == (1, 129)
        # Last element should be zero (the padding)
        assert emb[0, -1].item() == 0.0

    def test_dtype_is_float(self) -> None:
        t = torch.tensor([0.1, 0.9])
        emb = timestep_embedding(t, dim=64)
        assert emb.dtype == torch.float32

    def test_different_timesteps_give_different_embeddings(self) -> None:
        t = torch.tensor([0.0, 1.0])
        emb = timestep_embedding(t, dim=64)
        assert not torch.allclose(emb[0], emb[1])

    def test_batch_consistency(self) -> None:
        """Embedding of a single t should match corresponding row in batched call."""
        t_batch = torch.tensor([0.3, 0.7])
        emb_batch = timestep_embedding(t_batch, dim=64)
        emb_single = timestep_embedding(torch.tensor([0.3]), dim=64)
        assert torch.allclose(emb_batch[0], emb_single[0])


# ---------------------------------------------------------------------------
# get_image_ids
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetImageIds:
    def test_shape(self) -> None:
        bs, h, w, patch = 2, 16, 16, 2
        ids = get_image_ids(bs, h, w, patch, device=torch.device("cpu"))
        num_patches = (h // patch) * (w // patch)
        assert ids.shape == (bs, num_patches, 2)

    def test_batch_dimension_is_replicated(self) -> None:
        ids = get_image_ids(3, 8, 8, 2, device=torch.device("cpu"))
        # All batch elements should be identical
        assert torch.equal(ids[0], ids[1])
        assert torch.equal(ids[1], ids[2])

    def test_values_cover_grid(self) -> None:
        """IDs should cover a grid from 0..h//p-1 and 0..w//p-1."""
        h, w, p = 8, 12, 4
        ids = get_image_ids(1, h, w, p, device=torch.device("cpu"))  # (1, 6, 2)
        row_ids = ids[0, :, 0]
        col_ids = ids[0, :, 1]
        assert set(row_ids.tolist()) == {0.0, 1.0}  # h//p = 2
        assert set(col_ids.tolist()) == {0.0, 1.0, 2.0}  # w//p = 3


# ---------------------------------------------------------------------------
# img2seq / seq2img round-trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestImg2SeqRoundTrip:
    def test_img2seq_shape(self) -> None:
        bs, c, h, w = 2, 4, 16, 16
        patch_size = 2
        img = torch.randn(bs, c, h, w)
        seq = img2seq(img, patch_size)

        num_patches = (h // patch_size) * (w // patch_size)
        patch_dim = c * patch_size * patch_size
        assert seq.shape == (bs, num_patches, patch_dim)

    def test_seq2img_shape(self) -> None:
        bs, c, h, w = 2, 4, 16, 16
        patch_size = 2
        img = torch.randn(bs, c, h, w)
        seq = img2seq(img, patch_size)
        reconstructed = seq2img(seq, patch_size, (h, w))
        assert reconstructed.shape == (bs, c, h, w)

    def test_round_trip_exact(self) -> None:
        """img2seq -> seq2img should perfectly reconstruct the image."""
        bs, c, h, w = 1, 3, 8, 12
        patch_size = 4
        img = torch.randn(bs, c, h, w)
        seq = img2seq(img, patch_size)
        reconstructed = seq2img(seq, patch_size, (h, w))
        assert torch.allclose(img, reconstructed)

    def test_round_trip_with_tensor_shape(self) -> None:
        """seq2img accepts a torch.Tensor for the shape argument."""
        bs, c, h, w = 1, 4, 16, 16
        patch_size = 2
        img = torch.randn(bs, c, h, w)
        seq = img2seq(img, patch_size)
        shape_tensor = torch.tensor([h, w])
        reconstructed = seq2img(seq, patch_size, shape_tensor)
        assert torch.allclose(img, reconstructed)

    def test_non_square(self) -> None:
        """Round-trip should work for non-square images."""
        bs, c, h, w = 2, 8, 16, 32
        patch_size = 4
        img = torch.randn(bs, c, h, w)
        seq = img2seq(img, patch_size)
        reconstructed = seq2img(seq, patch_size, (h, w))
        assert torch.allclose(img, reconstructed)
