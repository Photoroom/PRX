import io
import logging
import random
import string
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from composer.utils import dist
from torch.utils.data import DataLoader, default_collate
from torchvision import tv_tensors
import torchvision.transforms.functional as TF
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from models.text_tower import TextTowerPresets

from .constants import BatchKeys

logger = logging.getLogger(__name__)

# Constants
EMPTY_SAMPLING_WEIGHT = 1e-9
DEFAULT_DATA_AUG_TARGETS = ["image"]


# ============================================================================
# Caption Handling
# ============================================================================

def parse_caption_keys(caption_keys: Union[str, List[str], List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    """Parse caption keys into standardized (key, weight) tuples."""
    if isinstance(caption_keys, str):
        return [(caption_keys, 1.0)]
    if isinstance(caption_keys[0], str):
        return [(key, 1.0) for key in caption_keys]  # type: ignore
    return caption_keys  # type: ignore


class CaptionSelector:
    """Handles caption selection logic with weights."""

    def __init__(
        self,
        caption_keys_and_weights: List[Tuple[str, float]],
        has_text_latents: bool,
        has_mask_text_latents: bool,
        text_tower_name: str,
    ):
        self.caption_keys_and_weights = caption_keys_and_weights
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents
        self.text_tower_name = text_tower_name

    def get_valid_captions(self, sample: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Return list of valid (caption_key, weight) tuples for this sample."""
        if self.has_text_latents:
            return self._get_valid_latent_captions(sample)
        return self._get_valid_text_captions(sample)

    def _get_valid_latent_captions(self, sample: Dict[str, Any]) -> List[Tuple[str, float]]:
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
            valid.append((key, final_weight))

        return valid

    def _get_valid_text_captions(self, sample: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get valid text captions (no latents)."""
        valid = []
        for key, weight in self.caption_keys_and_weights:
            if key not in sample:
                continue

            # Downweight empty captions
            is_empty = len(sample[key]) == 0
            final_weight = EMPTY_SAMPLING_WEIGHT if is_empty else weight
            valid.append((key, final_weight))

        return valid

    def select_caption(self, sample: Dict[str, Any]) -> str:
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


def image_to_tensor(image: Union[bytes, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert various image formats to torch tensor.

    Args:
        image: Image as bytes, PIL Image, numpy array, or torch tensor

    Returns:
        Torch tensor in [0, 1] range with shape (C, H, W)
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))

    if isinstance(image, Image.Image):
        image = TF.pil_to_tensor(image).to(torch.float32).div(255.0)
    elif isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32).div(255.0)
        if image.dim() == 3:
            image = image.permute(2, 0, 1).contiguous()
        else:
            image = image.unsqueeze(0)

    # Clamp to overcome possible issues due to numerical errors in previous resizing steps
    image = torch.clamp(image, 0.0, 1.0)
    return image


class SampleProcessor:
    """Processes raw samples into model inputs."""

    def __init__(
        self,
        caption_selector: CaptionSelector,
        text_tower_name: str,
        has_text_latents: bool,
        has_mask_text_latents: bool,
        transforms: Optional[List[Callable]] = None,
        transform_targets: List[str] = None,
    ):
        self.caption_selector = caption_selector
        self.text_tower_name = text_tower_name
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents
        self.transforms = transforms or []
        self.transform_targets = transform_targets or DEFAULT_DATA_AUG_TARGETS

    def process(self, raw_sample: Dict[str, Any]) -> Dict[BatchKeys, Any]:
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

    def _remove_invalid_values(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Remove entries with NaN values."""
        def is_valid(value: Any) -> bool:
            if isinstance(value, np.ndarray):
                return not np.isnan(value).any()
            if isinstance(value, torch.Tensor):
                return not torch.isnan(value).any()
            return True

        return {k: v for k, v in sample.items() if is_valid(v)}

    def _apply_transforms(self, sample: Dict[str, Any], caption_key: str) -> Dict[str, Any]:
        """Apply augmentation transforms to images."""
        # Convert images to tensors
        images = {}
        for target in self.transform_targets:
            if target in sample:
                img = sample[target]
                if isinstance(img, bytes):
                    img = image_to_tensor(img)
                images[target] = img

        # Apply each transform
        for transform in self.transforms:
            # Check condition if transform is condition-aware
            if hasattr(transform, 'check_condition'):
                if not transform.check_condition(sample):
                    continue

            # Wrap as tv_tensors and apply
            images = {k: tv_tensors.Image(v) for k, v in images.items()}
            images = transform(images)

        # Update sample with transformed images
        for target, img in images.items():
            sample[target] = img

        # Remove any precomputed latents since images changed
        sample.pop("img_latent", None)

        return sample

    def _add_image_data(self, sample: Dict[str, Any], output: Dict[BatchKeys, Any]) -> None:
        """Add image or latent to output."""
        # Try to use precomputed latent if available
        if "img_latent" in sample:
            moments = torch.tensor(sample["img_latent"], dtype=torch.float32)
            output[BatchKeys.image_latent] = sample_latent(moments)
            return

        # Otherwise use image
        if "image" in sample:
            output[BatchKeys.image] = image_to_tensor(sample["image"])

    def _add_text_data(self, sample: Dict[str, Any], caption_key: str, output: Dict[BatchKeys, Any]) -> None:
        """Add text prompt and embeddings to output."""
        output[BatchKeys.caption_key] = caption_key
        output[BatchKeys.prompt] = sample[caption_key]

        if not self.has_text_latents:
            return

        # Get text latent
        latent_key = f"latent_{caption_key}_{self.text_tower_name}"
        latent = sample[latent_key]
        if isinstance(latent, np.ndarray):
            latent = torch.tensor(latent, dtype=torch.float32)
        output[BatchKeys.prompt_embedding] = latent

        if not self.has_mask_text_latents:
            # Verify shape matches expected length
            expected_len = TextTowerPresets[self.text_tower_name]["model_max_length"]
            if latent.shape[0] != expected_len:
                raise ValueError(
                    f"Prompt embedding length {latent.shape[0]} doesn't match "
                    f"expected {expected_len} for {self.text_tower_name}"
                )
            return

        # Get attention mask
        mask_key = f"attention_mask_{caption_key}_{self.text_tower_name}"
        mask = torch.tensor(sample[mask_key], dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.squeeze(0)
        output[BatchKeys.prompt_embedding_mask] = mask

        # Pad embedding if needed
        if mask.shape[0] != latent.shape[0]:
            if mask.sum() != latent.shape[0]:
                raise ValueError(
                    f"Valid tokens ({mask.sum()}) doesn't match embedding length ({latent.shape[0]})"
                )
            padding_len = mask.shape[0] - latent.shape[0]
            padding = torch.zeros((padding_len, latent.shape[1]), dtype=torch.float32)
            output[BatchKeys.prompt_embedding] = torch.cat([latent, padding], dim=0)

    def _add_metadata(self, sample: Dict[str, Any], output: Dict[BatchKeys, Any]) -> None:
        """Add resolution metadata."""
        # Get image size
        if BatchKeys.image in output:
            h, w = output[BatchKeys.image].shape[-2:]
        elif BatchKeys.image_latent in output:
            h, w = output[BatchKeys.image_latent].shape[-2:]
            h, w = h * 8, w * 8
        else:
            raise ValueError("No image or latent found")

        # Use original size if available
        if "original_height" in sample and "original_width" in sample:
            h = min(h, sample["original_height"])
            w = min(w, sample["original_width"])

        output[BatchKeys.resolution] = torch.tensor([h, w])


class ProcessedDataset:
    """Base class for processed datasets with common functionality."""

    def __init__(
        self,
        caption_keys: Union[str, List[str], List[Tuple[str, float]]] = "caption",
        text_tower: str = "t5gemma2b-256-bf16",
        has_text_latents: bool = True,
        has_mask_text_latents: bool = True,
        transforms: Optional[List[Callable]] = None,
        transforms_targets: Union[List[str], str] = DEFAULT_DATA_AUG_TARGETS,
    ):
        if text_tower not in TextTowerPresets:
            raise ValueError(f"Unknown text tower: {text_tower}")

        self.text_tower_name = text_tower
        self.has_text_latents = has_text_latents
        self.has_mask_text_latents = has_mask_text_latents

        caption_keys_and_weights = parse_caption_keys(caption_keys)
        if isinstance(transforms_targets, str):
            transforms_targets = [transforms_targets]

        self.caption_selector = CaptionSelector(
            caption_keys_and_weights=caption_keys_and_weights,
            has_text_latents=has_text_latents,
            has_mask_text_latents=has_mask_text_latents,
            text_tower_name=text_tower,
        )

        self.processor = SampleProcessor(
            caption_selector=self.caption_selector,
            text_tower_name=text_tower,
            has_text_latents=has_text_latents,
            has_mask_text_latents=has_mask_text_latents,
            transforms=transforms,
            transform_targets=transforms_targets,
        )

    def __getitem__(self, index: int) -> Optional[Dict[BatchKeys, Any]]:
        """Get processed sample at index."""
        try:
            raw_sample = self._get_raw_item(index)
            return self.processor.process(raw_sample)
        except Exception as e:
            logger.error(f"Error processing sample {index}: {e}")
            logger.debug("Traceback:", exc_info=True)
            return None

    def _get_raw_item(self, index: int) -> Dict[str, Any]:
        """Get raw sample - must be implemented by subclasses."""
        raise NotImplementedError

# ============================================================================
# Synthetic Dataset
# ============================================================================

def build_synthetic_dataloader(
    batch_size: int,
    image_size: tuple[int, int] = (256, 256),
    text_tower: str = "t5gemma2b-256-bf16",
    num_samples: int = 1_000_000,
    has_text_latents: bool = True,
    has_mask_text_latents: bool = False,
    prefetch_factor: int = 4,
    num_workers: int = 32,
    **_kwargs: Any,
) -> DataLoader:
    """Build a synthetic dataloader for testing."""

    class SyntheticDataset(torch.utils.data.Dataset):
        """Synthetic dataset for testing."""

        def __init__(self, num_samples: int, image_size: tuple[int, int], text_tower: str,
                     has_text_latents: bool, has_mask_text_latents: bool):
            self.num_samples = num_samples
            self.image_size = image_size
            self.text_tower = text_tower
            self.has_text_latents = has_text_latents
            self.has_mask_text_latents = has_mask_text_latents
            self.seq_len = TextTowerPresets[text_tower]["model_max_length"]

            # Get embedding dimension
            if "t5xxl" in text_tower:
                self.hidden_dim = 4096
            elif "t5gemma2b" in text_tower:
                self.hidden_dim = 2304
            else:
                raise ValueError(f"Unknown text tower: {text_tower}")

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> Dict[BatchKeys, Any]:
            sample = {
                BatchKeys.image: torch.rand(3, *self.image_size),
                BatchKeys.prompt: "".join(random.choices(string.ascii_lowercase + " ", k=32)),
            }

            if self.has_text_latents:
                sample[BatchKeys.prompt_embedding] = torch.randn(self.seq_len, self.hidden_dim)
                if self.has_mask_text_latents:
                    sample[BatchKeys.prompt_embedding_mask] = torch.ones(self.seq_len, dtype=torch.bool)

            return sample

    dataset = SyntheticDataset(
        num_samples=num_samples,
        image_size=image_size,
        text_tower=text_tower,
        has_text_latents=has_text_latents,
        has_mask_text_latents=has_mask_text_latents,
    )

    sampler = dist.get_sampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=lambda batch: default_collate([x for x in batch if x is not None]),
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )


