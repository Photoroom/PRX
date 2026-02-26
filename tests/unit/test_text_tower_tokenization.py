"""Unit tests for TextTower text cleaning and tokenization (no GPU, no model weights)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from prx.models.text_tower import TextTower, TokenResult


# ---------------------------------------------------------------------------
# Text cleaning (static / instance methods, no model needed)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTextCleaning:
    """Test the text cleaning pipeline directly. These don't need a model at all."""

    def test_basic_clean_strips_whitespace(self) -> None:
        assert TextTower.basic_clean("  hello  ") == "hello"

    def test_basic_clean_fixes_unicode(self) -> None:
        # ftfy should fix mojibake-style issues
        result = TextTower.basic_clean("schön")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_basic_clean_unescapes_html(self) -> None:
        assert TextTower.basic_clean("&amp;") == "&"
        assert TextTower.basic_clean("&lt;b&gt;") == "<b>"

    def _make_tower_for_cleaning(self) -> TextTower:
        """Create a TextTower with mocked model creation so we only test clean_text."""
        with patch.object(TextTower, "create_model", return_value=(MagicMock(), None)):
            tower = TextTower.__new__(TextTower)
            tower.skip_text_cleaning = False
            # Compile the regex class attribute (it's on the class, not instance)
        return tower

    def test_clean_text_lowercases(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("HELLO World")
        assert result == result.lower()

    def test_clean_text_removes_urls(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("check out https://example.com/page for info")
        assert "https://" not in result
        assert "example.com" not in result

    def test_clean_text_removes_at_mentions(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("photo by @photographer123")
        assert "@photographer123" not in result

    def test_clean_text_removes_filenames(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("see image file photo.jpg for details")
        assert "photo.jpg" not in result

    def test_clean_text_normalizes_dashes(self) -> None:
        tower = self._make_tower_for_cleaning()
        # em-dash (U+2014) should become a regular hyphen
        result = tower.clean_text("word\u2014word")
        assert "\u2014" not in result
        assert "-" in result

    def test_clean_text_removes_shipping_text(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("buy now worldwide free shipping")
        assert "shipping" not in result

    def test_clean_text_handles_empty_string(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("")
        assert result == ""

    def test_clean_text_removes_ip_addresses(self) -> None:
        tower = self._make_tower_for_cleaning()
        result = tower.clean_text("server at 192.168.1.1 is down")
        assert "192.168.1.1" not in result


# ---------------------------------------------------------------------------
# Tokenization (mocked tokenizer)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTextToToken:
    """Test text_to_token with a mocked tokenizer (no HF download needed)."""

    def _make_tower_with_mock_tokenizer(
        self, max_tokens: int = 128, unpadded: bool = False
    ) -> TextTower:
        """Build a TextTower with a mock tokenizer and no actual model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        def tokenize_side_effect(
            texts: list[str], **kwargs: object
        ) -> dict[str, torch.Tensor]:
            bs = len(texts)
            seq_len = max_tokens if kwargs.get("padding") == "max_length" else min(10, max_tokens)
            return {
                "input_ids": torch.ones(bs, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(bs, seq_len, dtype=torch.long),
            }

        mock_tokenizer.side_effect = tokenize_side_effect
        mock_tokenizer.__call__ = tokenize_side_effect

        with patch.object(TextTower, "create_model", return_value=(mock_tokenizer, None)):
            tower = TextTower.__new__(TextTower)
            # Manually set all required attributes
            tower.only_tokenizer = True
            tower.use_attn_mask = True
            tower.use_last_hidden_state = True
            tower.torch_dtype = torch.float32
            tower.unpadded = unpadded
            tower.skip_text_cleaning = True  # skip cleaning to avoid regex deps in mock
            tower.tokenizer = mock_tokenizer
            tower.text_encoder = None
            tower.tokenizer_max_length = max_tokens
            tower.hidden_size = 0

        return tower

    def test_token_result_type(self) -> None:
        tower = self._make_tower_with_mock_tokenizer()
        result = tower.text_to_token("hello world")
        assert isinstance(result, TokenResult)
        assert isinstance(result.tokens, torch.Tensor)
        assert isinstance(result.attention_mask, torch.Tensor)

    def test_token_shapes_padded(self) -> None:
        max_tokens = 64
        tower = self._make_tower_with_mock_tokenizer(max_tokens=max_tokens)
        result = tower.text_to_token(["prompt one", "prompt two"])
        assert result.tokens.shape == (2, max_tokens)
        assert result.attention_mask.shape == (2, max_tokens)
        assert result.attention_mask.dtype == torch.bool

    def test_single_string_becomes_batch(self) -> None:
        tower = self._make_tower_with_mock_tokenizer(max_tokens=32)
        result = tower.text_to_token("single prompt")
        assert result.tokens.dim() == 2
        assert result.tokens.shape[0] == 1

    def test_num_pad_tokens_is_none_for_padded_mode(self) -> None:
        tower = self._make_tower_with_mock_tokenizer(max_tokens=32, unpadded=False)
        result = tower.text_to_token("test")
        assert result.num_pad_tokens is None
