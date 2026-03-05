import logging
import random
import string
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from composer.utils import dist
from torch.utils.data import DataLoader, default_collate
from torchvision import tv_tensors
from diffusers.utils.torch_utils import randn_tensor

from .constants import BatchKeys
from .transforms import image_to_tensor

logger = logging.getLogger(__name__)

# Constants
EMPTY_SAMPLING_WEIGHT = 1e-9  # Near-zero weight for empty captions to avoid crashing on samples with only empty captions
DEFAULT_DATA_AUG_TARGETS = ["image"]
DEFAULT_VAE_SPATIAL_SCALE_FACTOR = 8


# ============================================================================
# Caption Handling
# ============================================================================

CaptionKeys = str | Sequence[str] | Sequence[tuple[str, float]]


class CaptionKeyAndWeight(NamedTuple):
    key: str
    weight: float


def parse_caption_keys(caption_keys: CaptionKeys) -> list[CaptionKeyAndWeight]:
    """Parse caption keys into standardized CaptionKeyAndWeight tuples."""
    if isinstance(caption_keys, str):
        return [CaptionKeyAndWeight(caption_keys, 1.0)]

    if not caption_keys:
        return []

    if isinstance(caption_keys[0], str):
        keys = cast(Sequence[str], caption_keys)
        return [CaptionKeyAndWeight(k, 1.0) for k in keys]

    keys = cast(Sequence[tuple[str, float]], caption_keys)
    return [CaptionKeyAndWeight(k, float(w)) for k, w in keys]


class CaptionSelector:
    """Handles caption selection logic with weights."""

    def __init__(
        self,
        caption_keys_and_weights: list[CaptionKeyAndWeight],
        has_text_latents: bool,
        has_mask_text_latents: bool,
        text_tower_name: str,
    ):
        self.caption_keys_and_weights = caption_keys_and_weights
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents
        self.text_tower_name = text_tower_name

    def get_valid_captions(self, sample: dict[str, Any]) -> list[CaptionKeyAndWeight]:
        """Return list of valid CaptionKeyAndWeight tuples for this sample."""
        if self.has_text_latents:
            return self._get_valid_latent_captions(sample)
        return self._get_valid_text_captions(sample)

    def _get_valid_latent_captions(self, sample: dict[str, Any]) -> list[CaptionKeyAndWeight]:
        """Get valid captions with precomputed latents."""
        valid = []
        for key, weight in self.caption_keys_and_weights:
            latent_key = f"latent_{key}_{self.text_tower_name}"
            if latent_key not in sample:
                continue

            # Check mask if required
            if self.has_mask_text_latents:
                mask_key = f"attention_mask_{key}_{self.text_tower_name}"
                if mask_key not in sample:
                    continue

            # Downweight empty captions
            is_empty = len(sample[latent_key]) <= 1
            if self.has_mask_text_latents:
                mask_key = f"attention_mask_{key}_{self.text_tower_name}"
                is_empty = is_empty or len(sample[mask_key]) <= 1

            final_weight = EMPTY_SAMPLING_WEIGHT if is_empty else weight
            valid.append(CaptionKeyAndWeight(key, final_weight))

        return valid

    def _get_valid_text_captions(self, sample: dict[str, Any]) -> list[CaptionKeyAndWeight]:
        """Get valid text captions (no latents)."""
        valid = []
        for key, weight in self.caption_keys_and_weights:
            if key not in sample:
                continue

            # Downweight empty captions
            is_empty = len(sample[key]) == 0
            final_weight = EMPTY_SAMPLING_WEIGHT if is_empty else weight
            valid.append(CaptionKeyAndWeight(key, final_weight))

        return valid

    def select_caption(self, sample: dict[str, Any]) -> str:
        """Select a random caption weighted by configured weights."""
        valid_captions = self.get_valid_captions(sample)
        if not valid_captions:
            raise ValueError(f"No valid captions found. Available keys: {list(sample.keys())}")

        keys, weights = zip(*valid_captions)
        return random.choices(list(keys), weights=list(weights))[0]


# ============================================================================
# Sample Processing
# ============================================================================

def sample_latent(moments: torch.Tensor) -> torch.Tensor:
    """Sample from a latent distribution given mean and std.

    Args:
        moments: Tensor containing concatenated mean and std

    Returns:
        Sampled latent tensor
    """
    with torch.no_grad():
        # Get a sample out of the distribution
        mean, std = torch.chunk(moments, 2)
        sample = randn_tensor(mean.shape, generator=None, device=moments.device, dtype=moments.dtype)
        x = mean + std * sample
    return x.contiguous()


class SampleProcessor:
    """Processes raw samples into model inputs."""

    def __init__(
        self,
        caption_selector: CaptionSelector,
        text_tower_name: str,
        has_text_latents: bool,
        has_mask_text_latents: bool,
        prompt_max_tokens: int = 256,
        transforms: list[Callable] | None = None,
        transform_targets: list[str] = None,
    ):
        self.caption_selector = caption_selector
        self.text_tower_name = text_tower_name
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents
        self.prompt_max_tokens = prompt_max_tokens
        self.transforms = transforms or []
        self.transform_targets = transform_targets or DEFAULT_DATA_AUG_TARGETS

    def process(self, raw_sample: dict[str, Any]) -> dict[BatchKeys, Any]:
        """Process raw sample into model input format."""
        # Clean up invalid values
        sample = self._remove_invalid_values(raw_sample)

        # Select caption
        caption_key = self.caption_selector.select_caption(sample)

        # Apply transforms if needed
        if self.transforms:
            sample = self._apply_transforms(sample, caption_key)

        # Build output batch
        output = {}

        # Add image or image latent
        self._add_image_data(sample, output)

        # Add text data
        self._add_text_data(sample, caption_key, output)

        # Add metadata
        self._add_metadata(sample, output)

        return output

    def _remove_invalid_values(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Remove entries with NaN values."""
        def is_valid(value: Any) -> bool:
            if isinstance(value, np.ndarray):
                return not np.isnan(value).any()
            if isinstance(value, torch.Tensor):
                return not torch.isnan(value).any()
            return True

        return {k: v for k, v in sample.items() if is_valid(v)}

    def _apply_transforms(self, sample: dict[str, Any], caption_key: str) -> dict[str, Any]:
        """Apply augmentation transforms to images."""
        # Convert images to tensors
        images = {}
        for target in self.transform_targets:
            if target in sample:
                img = sample[target]
                if not isinstance(img, torch.Tensor):
                    img = image_to_tensor(img)
                images[target] = img

        # Apply each transform
        for transform in self.transforms:
            images = {k: tv_tensors.Image(v) for k, v in images.items()}
            images = transform(images)

        # Update sample with transformed images
        for target, img in images.items():
            sample[target] = img

        # Remove any precomputed latents since images changed
        sample.pop("img_latent", None)

        return sample

    def _add_image_data(self, sample: dict[str, Any], output: dict[BatchKeys, Any]) -> None:
        """Add image or latent to output."""
        # Try to use precomputed latent if available
        if "img_latent" in sample:
            moments = torch.tensor(sample["img_latent"], dtype=torch.float32)
            output[BatchKeys.IMAGE_LATENT] = sample_latent(moments)
            return

        # Otherwise use image
        if "image" in sample:
            output[BatchKeys.IMAGE] = image_to_tensor(sample["image"])

    def _add_text_data(self, sample: dict[str, Any], caption_key: str, output: dict[BatchKeys, Any]) -> None:
        """Add text prompt and embeddings to output."""
        output[BatchKeys.CAPTION_KEY] = caption_key
        output[BatchKeys.PROMPT] = sample[caption_key]

        if not self.has_text_latents:
            return

        # Get text latent
        latent_key = f"latent_{caption_key}_{self.text_tower_name}"
        latent = sample[latent_key]
        if isinstance(latent, np.ndarray):
            latent = torch.tensor(latent, dtype=torch.float32)
        output[BatchKeys.PROMPT_EMBEDDING] = latent

        if not self.has_mask_text_latents:
            # Verify shape matches expected length
            if latent.shape[0] != self.prompt_max_tokens:
                raise ValueError(
                    f"Prompt embedding length {latent.shape[0]} doesn't match "
                    f"expected {self.prompt_max_tokens} for {self.text_tower_name}"
                )
            return

        # Get attention mask
        mask_key = f"attention_mask_{caption_key}_{self.text_tower_name}"
        mask = torch.tensor(sample[mask_key], dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.squeeze(0)
        output[BatchKeys.PROMPT_EMBEDDING_MASK] = mask

        # Pad embedding if needed
        if mask.shape[0] != latent.shape[0]:
            if mask.sum() != latent.shape[0]:
                raise ValueError(
                    f"Valid tokens ({mask.sum()}) doesn't match embedding length ({latent.shape[0]})"
                )
            padding_len = mask.shape[0] - latent.shape[0]
            padding = torch.zeros((padding_len, latent.shape[1]), dtype=torch.float32)
            output[BatchKeys.PROMPT_EMBEDDING] = torch.cat([latent, padding], dim=0)

    def _add_metadata(self, sample: dict[str, Any], output: dict[BatchKeys, Any]) -> None:
        """Add resolution metadata."""
        # Get image size
        if BatchKeys.IMAGE in output:
            h, w = output[BatchKeys.IMAGE].shape[-2:]
        elif BatchKeys.IMAGE_LATENT in output:
            h, w = output[BatchKeys.IMAGE_LATENT].shape[-2:]
            h, w = h * DEFAULT_VAE_SPATIAL_SCALE_FACTOR, w * DEFAULT_VAE_SPATIAL_SCALE_FACTOR
        else:
            raise ValueError("No image or latent found")

        # Use original size if available
        if "original_height" in sample and "original_width" in sample:
            h = min(h, sample["original_height"])
            w = min(w, sample["original_width"])

        output[BatchKeys.RESOLUTION] = torch.tensor([h, w])


class ProcessedDataset:
    """Base class for processed datasets with common functionality."""

    def __init__(
        self,
        caption_keys: CaptionKeys = "caption",
        text_tower: str = "t5gemma2b-256-bf16",
        prompt_max_tokens: int = 256,
        has_text_latents: bool = True,
        has_mask_text_latents: bool = True,
        transforms: list[Callable] | None = None,
        transforms_targets: list[str] | str = DEFAULT_DATA_AUG_TARGETS,
    ):
        self.text_tower_name = text_tower
        self.prompt_max_tokens = prompt_max_tokens
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents

        caption_keys_and_weights = parse_caption_keys(caption_keys)
        if isinstance(transforms_targets, str):
            transforms_targets = [transforms_targets]

        self.processor = SampleProcessor(
            caption_selector=CaptionSelector(
                caption_keys_and_weights=caption_keys_and_weights,
                has_text_latents=has_text_latents,
                has_mask_text_latents=has_mask_text_latents,
                text_tower_name=text_tower,
            ),
            text_tower_name=text_tower,
            has_text_latents=has_text_latents,
            has_mask_text_latents=has_mask_text_latents,
            prompt_max_tokens=prompt_max_tokens,
            transforms=transforms,
            transform_targets=transforms_targets,
        )

    def __getitem__(self, index: int) -> dict[BatchKeys, Any] | None:
        """Get processed sample at index."""
        try:
            raw_sample = self._get_raw_item(index)
            return self.processor.process(raw_sample)
        except Exception as e:
            logger.error(f"Error processing sample {index}: {e}")
            logger.debug("Traceback:", exc_info=True)
            return None

    def _get_raw_item(self, index: int) -> dict[str, Any]:
        """Get raw sample - must be implemented by subclasses."""
        raise NotImplementedError

    def _get_sampler(self, shuffle: bool) -> torch.utils.data.Sampler | None:
        """Get sampler for this dataset. Override in subclasses if needed."""
        return dist.get_sampler(self, shuffle=shuffle)

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle samples.
            drop_last: Whether to drop the last incomplete batch.
            **dataloader_kwargs: Additional arguments passed to DataLoader.

        Returns:
            Configured DataLoader instance.
        """
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self._get_sampler(shuffle),
            drop_last=drop_last,
            collate_fn=lambda batch: default_collate([x for x in batch if x is not None]),
            **dataloader_kwargs,
        )


# ============================================================================
# Dummy Dataset
# ============================================================================

class DummyDataset(torch.utils.data.Dataset):
    """Dataset that generates random data for testing.

    Args:
        num_samples: Number of samples in the dataset.
        image_size: Size of generated images as (height, width).
        text_tower: Name of the text encoder preset for latent dimensions.
        has_text_latents: Whether to generate text latents.
        has_mask_text_latents: Whether to generate attention masks for text latents.
    """

    def __init__(
        self,
        num_samples: int = 1_000_000,
        image_size: tuple[int, int] = (256, 256),
        text_tower: str = "t5gemma2b-256-bf16",
        prompt_max_tokens: int = 256,
        hidden_dim: int = 2304,
        has_text_latents: bool = True,
        has_mask_text_latents: bool = False,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.text_tower = text_tower
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents
        self.seq_len = prompt_max_tokens
        self.hidden_dim = hidden_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[BatchKeys, Any]:
        sample = {
            BatchKeys.IMAGE: torch.rand(3, *self.image_size),
            BatchKeys.PROMPT: "".join(random.choices(string.ascii_lowercase + " ", k=32)),
        }

        if self.has_text_latents:
            sample[BatchKeys.PROMPT_EMBEDDING] = torch.randn(self.seq_len, self.hidden_dim)
            if self.has_mask_text_latents:
                sample[BatchKeys.PROMPT_EMBEDDING_MASK] = torch.ones(self.seq_len, dtype=torch.bool)

        return sample

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle samples.
            drop_last: Whether to drop the last incomplete batch.
            **dataloader_kwargs: Additional arguments passed to DataLoader.

        Returns:
            Configured DataLoader instance.
        """
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=dist.get_sampler(self, shuffle=shuffle),
            drop_last=drop_last,
            collate_fn=lambda batch: default_collate([x for x in batch if x is not None]),
            **dataloader_kwargs,
        )


