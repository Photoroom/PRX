import io
import math
from collections.abc import Iterable
from functools import lru_cache
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as transforms
from PIL import Image

T = TypeVar("T", torch.Tensor, Image.Image)


class ToTensorImage:
    """
    Convert a PIL image or torch tensor to an image tensor (from torchvision.tv_tensors).
    """

    def __init__(self) -> None:
        self.transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])

    def __call__(self, *img: Image.Image | torch.Tensor) -> torch.Tensor:
        return self.transform(*img)


def image_to_tensor(image: bytes | Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
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


def build_image_size_list(
    default_image_size: int, patch_size_pixels: int, min_ar: float = 0.5, max_ar: float = 2.0, divisible_by: int = 16
) -> set[tuple[int, int]]:
    """
    Build a list of image sizes to cover multiple aspect ratios (non exhaustive).
    All generated sizes have approximately the same total number of patches.
    For all possible patch grid widths, we take the longest possible height, and conversely.
    With the default values we get 46 (h, w) tuples.

    Args:
        default_image_size: Base image size (e.g., 512)
        patch_size_pixels: Size of each patch in pixels (e.g., 16 means 16x16 pixel patches)
        min_ar: Minimum aspect ratio (width/height)
        max_ar: Maximum aspect ratio (width/height)
        divisible_by: All dimensions must be divisible by this value
    """
    base_num_patches = default_image_size / patch_size_pixels  # e.g., 512/16 = 32 patches per side
    image_list = []

    min_num_patches_w = math.ceil(math.sqrt(base_num_patches**2 * min_ar))
    max_num_patches_w = math.floor(math.sqrt(base_num_patches**2 * max_ar))
    for num_patches_w in range(min_num_patches_w, max_num_patches_w + 1):  # go over all possible patch grid widths
        num_patches_h = math.floor(base_num_patches**2 / num_patches_w)  # get max height
        img_w, img_h = num_patches_w * patch_size_pixels, num_patches_h * patch_size_pixels
        if img_w % divisible_by != 0 or img_h % divisible_by != 0:
            continue
        image_list.append((img_w, img_h))

    min_num_patches_h = math.ceil(math.sqrt(base_num_patches**2 * 1.0 / max_ar))
    max_num_patches_h = math.floor(math.sqrt(base_num_patches**2 * 1.0 / min_ar))
    for num_patches_h in range(min_num_patches_h, max_num_patches_h + 1):  # go over all possible patch grid heights
        num_patches_w = math.floor(base_num_patches**2 / num_patches_h)  # get max width
        img_w, img_h = num_patches_w * patch_size_pixels, num_patches_h * patch_size_pixels
        if img_w % divisible_by != 0 or img_h % divisible_by != 0:
            continue
        image_list.append((img_w, img_h))

    return set(image_list)


class ArAwareTransform(torch.nn.Module, ABC):
    """
    Abstract base class for aspect-ratio-aware transforms.

    Builds a mapping of aspect ratios to target image sizes. Subclasses must
    implement __call__ to define the actual transform behavior.
    """

    def __init__(
        self,
        default_image_size: int = 512,
        patch_size_pixels: int = 16,
        min_ar: float = 0.5,
        max_ar: float = 2.0,
        divisible_by: int = 16,
    ):
        """
        Crop to the closest aspect ratio (or leave unchanged)
        Aspect Ratio is defined as width / height
        """
        super().__init__()
        self.default_image_size = default_image_size
        self.patch_size_pixels = patch_size_pixels
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.divisible_by = divisible_by
        self.target_image_sizes: set[tuple[int, int]] = build_image_size_list(
            default_image_size=default_image_size,
            patch_size_pixels=patch_size_pixels,
            min_ar=min_ar,
            max_ar=max_ar,
            divisible_by=divisible_by,
        )
        self.ar_to_size = {w / h: (w, h) for w, h in self.target_image_sizes}

    @lru_cache(maxsize=128)
    def get_closest_ar(self, image_width: int, image_height: int) -> float:
        image_ar = image_width / image_height
        return min(self.ar_to_size.keys(), key=lambda x: abs(x - image_ar))

    def __repr__(self) -> str:
        return f"{self.__class__}.\n Aspect Ratio : image size. \n" + "\n".join(
            [f"{k} : {v}" for k, v in sorted(self.ar_to_size.items())]
        )

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> Any:
        pass


def _get_shape(image: torch.Tensor | Image.Image) -> tuple[int, int]:
    # Image.size is [w, h], while a torch tensor is typically [h, w]
    if isinstance(image, torch.Tensor):
        shape: tuple[int, int] = image.shape[-2:]
        return shape
    shape: tuple[int, int] = image.size[::-1]
    return shape


class ArAwareCenterCrop(ArAwareTransform):
    """
    Crop to the closest aspect ratio to match one of the aspect ratio listed.
    Applicable to PIL.Image and torch.Tensor
    """

    @lru_cache(maxsize=128)
    def _get_target_ar(self, image_h: int, image_w: int) -> float:
        image_ar = float(image_w) / image_h
        # If w > h make sure to crop on w
        if image_w >= image_h:
            selected_ar = max([a for a in self.ar_to_size.keys() if a <= image_ar])  # biggest ar smaller than image_ar
        else:  # else crop on h
            selected_ar = min([a for a in self.ar_to_size.keys() if a >= image_ar])
        return selected_ar

    @lru_cache(maxsize=128)
    def _get_crop_coord(self, image_h: int, image_w: int, ar: float) -> tuple[int, int]:
        # If w > h make sure to crop on w
        if image_w >= image_h:
            crop_h = image_h
            crop_w = round(ar * image_h)
        else:  # else crop on h
            crop_w = image_w
            crop_h = round(image_w / ar)
        return crop_h, crop_w

    def get_target_ar(self, image: torch.Tensor | Image.Image) -> float:
        image_h, image_w = _get_shape(image)
        return self._get_target_ar(image_h, image_w)

    def get_crop_coord(self, image: torch.Tensor | Image.Image, target_ar: float) -> tuple[int, int]:
        image_h, image_w = _get_shape(image)
        return self._get_crop_coord(image_h, image_w, target_ar)

    def __call__(
        self, images: torch.Tensor | Iterable[torch.Tensor]
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        """
        Crop each image tensor to the target size based on the closest aspect ratio.

        Args:
            images: An image tensor of shape (.. x H x W), or a list thereof.

        Returns:
            A single cropped image tensor if one tensor is passed,
            otherwise a list of cropped image tensors.
        """

        # Crop a single image
        def crop_image(image: torch.Tensor, target_ar: float | None = None) -> torch.Tensor:
            # Either compute the target ar on the fly, or use the one provided
            target_h, target_w = self.get_crop_coord(image, target_ar or self.get_target_ar(image))
            return transforms.functional.center_crop(image, (target_h, target_w))

        # return a single tensor if only one is passed
        if isinstance(images, torch.Tensor):
            return crop_image(images)

        elif isinstance(images, dict):
            target_ar = self.get_target_ar(next(iter(images.values())))
            return {k: crop_image(v, target_ar) for k, v in images.items()}
        elif isinstance(images, Sequence):
            target_ar = self.get_target_ar(images[0])
            return [crop_image(image, target_ar) for image in images]
        else:
            raise ValueError(f"Invalid input type: {type(images)}, expected torch.Tensor, dict, or list")


class ArAwareResize(ArAwareTransform):
    """
    Resize to the closest aspect ratio available.
    Applicable to PIL.Image and torch.Tensor
    """

    def get_target_size(self, image: torch.Tensor | Image.Image) -> tuple[int, int]:
        image_height, image_width = _get_shape(image)
        return self.ar_to_size[self.get_closest_ar(image_width, image_height)]

    def __call__(
        self, images: torch.Tensor | Iterable[torch.Tensor]
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        """
        Resize each image tensor to the target size.

        Args:
            images: An image tensor of shape (.. x H x W), or a list thereof.

        Returns:
            A single resized image tensor if one tensor is passed,
            otherwise a list of resized image tensors.
        """

        # Resize a single image
        def resize_image(image: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
            # Either compute the target size on the fly, or use the one provided
            target_w, target_h = target_size or self.get_target_size(image)

            # Note: Dont use antialias if the image has 1 channel (mask, etc.)
            kwargs = {}
            if isinstance(image, torch.Tensor):
                kwargs["antialias"] = image.shape[-3] != 1

            return transforms.functional.resize(image, (target_h, target_w), **kwargs)

        # Return a single tensor if only one is passed
        if isinstance(images, torch.Tensor):
            return resize_image(images)

        # Handle both dict and list/iterable inputs
        elif isinstance(images, dict):
            first_image = next(iter(images.values()))
            size = self.get_target_size(first_image)
            return {k: resize_image(v, target_size=size) for k, v in images.items()}
        elif isinstance(images, Sequence):
            first_image = images[0]
            size = self.get_target_size(first_image)
            return [resize_image(image, target_size=size) for image in images]
        else:
            raise ValueError(f"Invalid input type: {type(images)}, expected torch.Tensor, dict, or list")
        
