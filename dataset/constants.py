from enum import Enum


class BatchKeys(str, Enum):
    """Keys used for batch data in txt2img dataset."""
    # Core image keys
    image = "image"
    image_latent = "image_latent"

    # Text/prompt keys
    prompt = "prompt"
    negative_prompt = "negative_prompt"
    prompt_embedding = "prompt_embedding"
    prompt_embedding_mask = "prompt_embedding_mask"

    # Image metadata
    original_height = "original_height"
    original_width = "original_width"
    resolution = "resolution"

    # Task and logging
    caption_key = "caption_key"

    # Noise for generation
    noise = "noise"
    
    # Repa keys
    target_representation = "target_representation"
