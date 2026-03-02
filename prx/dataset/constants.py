from enum import StrEnum, auto


class BatchKeys(StrEnum):
    """Keys used for batch data in txt2img dataset."""
    # Core image keys
    IMAGE = auto()
    IMAGE_LATENT = auto()

    # Text/prompt keys
    PROMPT = auto()
    NEGATIVE_PROMPT = auto()
    PROMPT_EMBEDDING = auto()
    PROMPT_EMBEDDING_MASK = auto()

    # Image metadata
    ORIGINAL_HEIGHT = auto()
    ORIGINAL_WIDTH = auto()
    RESOLUTION = auto()

    # Task and logging
    CAPTION_KEY = auto()

    # Noise for generation
    NOISE = auto()

    # Repa keys
    TARGET_REPRESENTATION = auto()
