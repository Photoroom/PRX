"""Feature extractors for image quality metrics (CLIP, DINOv2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPFeatureExtractor(nn.Module):
    """CLIP-based feature extractor for CMMD computation.

    Extracts visual features from images using CLIP's vision encoder from transformers.
    Compatible with torchmetrics KernelInceptionDistance as a custom feature extractor.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14-336", use_pil: bool = True):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.eval()

        self.use_pil = use_pil

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor (N, 3, H, W), uint8 in [0,255]

        Returns:
            (N, D) float tensor, L2-normalized.
        """
        if self.use_pil:
            from torchvision.transforms.functional import to_pil_image

            imgs_cpu = images.detach().cpu()
            pil_images = [to_pil_image(img) for img in imgs_cpu]
            inputs = self.processor(images=pil_images, return_tensors="pt")
        else:
            x = images.float() / 255.0
            inputs = self.processor(images=[img for img in x], return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        emb = outputs.image_embeds
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        return emb


class DINOFeatureExtractor(nn.Module):
    """DINOv2 feature extractor for DINO-MMD computation."""

    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",
        resize_size: int = 518,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()

        self.resize_size = int(resize_size)
        self.normalize_embeddings = bool(normalize_embeddings)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor (N, 3, H, W), uint8 in [0,255]
        Returns:
            Tensor (N, D) float, optionally L2-normalized.
        """
        x = images.float() / 255.0

        x = F.interpolate(x, size=(self.resize_size, self.resize_size), mode="bicubic", align_corners=False)

        x = (x - self.mean) / self.std

        feats = self.model(x)

        if self.normalize_embeddings:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return feats
