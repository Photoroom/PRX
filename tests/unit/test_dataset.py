"""Unit tests for dataset utilities: constants, CaptionSelector, transforms."""

from typing import Any

import pytest

from prx.dataset.constants import BatchKeys
from prx.dataset.dataset import (
    EMPTY_SAMPLING_WEIGHT,
    CaptionKeyAndWeight,
    CaptionSelector,
    parse_caption_keys,
)
from prx.dataset.transforms import build_image_size_list


# ---------------------------------------------------------------------------
# BatchKeys enum
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchKeys:
    def test_core_keys_exist(self) -> None:
        assert BatchKeys.IMAGE
        assert BatchKeys.IMAGE_LATENT
        assert BatchKeys.PROMPT
        assert BatchKeys.PROMPT_EMBEDDING
        assert BatchKeys.PROMPT_EMBEDDING_MASK
        assert BatchKeys.NOISE

    def test_values_are_strings(self) -> None:
        for member in BatchKeys:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# parse_caption_keys
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseCaptionKeys:
    def test_single_string(self) -> None:
        result = parse_caption_keys("caption")
        assert result == [CaptionKeyAndWeight("caption", 1.0)]

    def test_list_of_strings(self) -> None:
        result = parse_caption_keys(["caption", "alt_text"])
        assert len(result) == 2
        assert result[0] == CaptionKeyAndWeight("caption", 1.0)
        assert result[1] == CaptionKeyAndWeight("alt_text", 1.0)

    def test_weighted_tuples(self) -> None:
        result = parse_caption_keys([("caption", 0.8), ("alt", 0.2)])
        assert result == [
            CaptionKeyAndWeight("caption", 0.8),
            CaptionKeyAndWeight("alt", 0.2),
        ]

    def test_empty_list(self) -> None:
        result = parse_caption_keys([])
        assert result == []


# ---------------------------------------------------------------------------
# CaptionSelector
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCaptionSelector:
    """Tests for CaptionSelector with has_text_latents=False (text-only mode)."""

    def _make_selector(
        self,
        keys_and_weights: list[CaptionKeyAndWeight] | None = None,
        has_text_latents: bool = False,
        has_mask_text_latents: bool = False,
    ) -> CaptionSelector:
        if keys_and_weights is None:
            keys_and_weights = [
                CaptionKeyAndWeight("caption", 1.0),
                CaptionKeyAndWeight("alt_text", 0.5),
            ]
        return CaptionSelector(
            caption_keys_and_weights=keys_and_weights,
            has_text_latents=has_text_latents,
            has_mask_text_latents=has_mask_text_latents,
            text_tower_name="test_tower",
        )

    def test_get_valid_text_captions_all_present(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {"caption": "a cat", "alt_text": "kitty"}
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 2
        assert valid[0] == CaptionKeyAndWeight("caption", 1.0)
        assert valid[1] == CaptionKeyAndWeight("alt_text", 0.5)

    def test_get_valid_text_captions_missing_key(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {"caption": "a cat"}
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 1
        assert valid[0].key == "caption"

    def test_empty_caption_gets_downweighted(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {"caption": "", "alt_text": "kitty"}
        valid = sel.get_valid_captions(sample)
        cap_entry = next(v for v in valid if v.key == "caption")
        alt_entry = next(v for v in valid if v.key == "alt_text")
        assert cap_entry.weight == EMPTY_SAMPLING_WEIGHT
        assert alt_entry.weight == 0.5

    def test_select_caption_raises_when_no_valid(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {"unrelated_key": "value"}
        with pytest.raises(ValueError, match="No valid captions found"):
            sel.select_caption(sample)

    def test_select_caption_returns_valid_key(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {"caption": "hello", "alt_text": "world"}
        for _ in range(20):
            key = sel.select_caption(sample)
            assert key in ("caption", "alt_text")


@pytest.mark.unit
class TestCaptionSelectorLatentMode:
    """Tests for CaptionSelector with has_text_latents=True."""

    def _make_selector(self, has_mask: bool = False) -> CaptionSelector:
        return CaptionSelector(
            caption_keys_and_weights=[CaptionKeyAndWeight("caption", 1.0)],
            has_text_latents=True,
            has_mask_text_latents=has_mask,
            text_tower_name="t5",
        )

    def test_valid_latent_caption(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {
            "caption": "a dog",
            "latent_caption_t5": [1.0, 2.0, 3.0],
        }
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 1
        assert valid[0].weight == 1.0

    def test_missing_latent_key_is_filtered(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {"caption": "a dog"}  # no latent_caption_t5
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 0

    def test_empty_latent_gets_downweighted(self) -> None:
        sel = self._make_selector()
        sample: dict[str, Any] = {
            "caption": "a dog",
            "latent_caption_t5": [0.0],  # length <= 1, treated as empty
        }
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 1
        assert valid[0].weight == EMPTY_SAMPLING_WEIGHT

    def test_mask_required_but_missing(self) -> None:
        sel = self._make_selector(has_mask=True)
        sample: dict[str, Any] = {
            "caption": "a dog",
            "latent_caption_t5": [1.0, 2.0],
            # missing attention_mask_caption_t5
        }
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 0

    def test_mask_present(self) -> None:
        sel = self._make_selector(has_mask=True)
        sample: dict[str, Any] = {
            "caption": "a dog",
            "latent_caption_t5": [1.0, 2.0, 3.0],
            "attention_mask_caption_t5": [1, 1, 1],
        }
        valid = sel.get_valid_captions(sample)
        assert len(valid) == 1
        assert valid[0].weight == 1.0


# ---------------------------------------------------------------------------
# build_image_size_list (key transform utility)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildImageSizeList:
    def test_returns_non_empty(self) -> None:
        sizes = build_image_size_list(512, patch_size_pixels=16)
        assert len(sizes) > 0

    def test_all_divisible(self) -> None:
        sizes = build_image_size_list(512, patch_size_pixels=16, divisible_by=16)
        for w, h in sizes:
            assert w % 16 == 0
            assert h % 16 == 0

    def test_aspect_ratios_in_range(self) -> None:
        min_ar, max_ar = 0.5, 2.0
        sizes = build_image_size_list(512, patch_size_pixels=16, min_ar=min_ar, max_ar=max_ar)
        for w, h in sizes:
            ar = w / h
            assert ar >= min_ar - 0.1  # small tolerance for rounding
            assert ar <= max_ar + 0.1

    def test_contains_square(self) -> None:
        sizes = build_image_size_list(512, patch_size_pixels=16)
        # There should be at least one roughly square size
        ars = [w / h for w, h in sizes]
        assert any(0.9 <= ar <= 1.1 for ar in ars)
