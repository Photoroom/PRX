from .constants import BatchKeys
from .dataset import (
    CaptionSelector,
    DummyDataset,
    ProcessedDataset,
    SampleProcessor,
)

__all__ = [
    # Constants
    "BatchKeys",
    # Dataset classes
    "CaptionSelector",
    "DummyDataset",
    "ProcessedDataset",
    "SampleProcessor",
    "StreamingProcessedDataset",
]


def __getattr__(name: str):
    if name == "StreamingProcessedDataset":
        from .mds_dataset import StreamingProcessedDataset

        return StreamingProcessedDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
