from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from prx.dataset.mds_dataset import DataLoaderConfig, ProcessedConfig, StreamingConfig

CONFIG_DIR = Path(__file__).parents[2] / "configs" / "yamls" / "dataset"
DATASET_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _fake_streaming_init(instance, **kwargs):
    instance.samples_per_stream = [10]
    instance.streams = kwargs["streams"]


@pytest.mark.unit
@pytest.mark.parametrize("config_path", DATASET_CONFIGS, ids=lambda p: p.stem)
@patch("prx.dataset.mds_dataset.StreamingProcessedDataset.size", new_callable=PropertyMock, return_value=10)
@patch("prx.dataset.mds_dataset.StreamingProcessedDataset.__len__", return_value=10)
@patch("prx.dataset.mds_dataset.ProcessedDataset.__init__", return_value=None)
@patch("prx.dataset.mds_dataset.StreamingDataset.__init__", side_effect=_fake_streaming_init)
@patch("prx.dataset.mds_dataset.PatchedStream")
@patch("prx.dataset.mds_dataset.get_stream_iterator", return_value=iter([(None, "local", "index.json", None)]))
def test_shipped_dataset_config_instantiates(
    mock_iter, mock_stream, mock_streaming_init, mock_processed_init, mock_len, mock_size, config_path
) -> None:
    """Each shipped dataset YAML must instantiate through Hydra with defaults filled in."""
    # Wrap in a parent so the ${image_size}-style interpolations resolve.
    cfg = OmegaConf.create(
        {
            "image_size": 256,
            "patch_size_pixels": 32,
            "diffusion_text_tower": {"preset_name": "t5gemma2b-256-bf16"},
            "dataset": OmegaConf.load(config_path),
        }
    )

    dataset = instantiate(cfg.dataset)

    # Sections must arrive as real dataclasses, not raw DictConfigs, so defaults apply.
    for section, expected in [
        ("streaming", StreamingConfig),
        ("processed", ProcessedConfig),
        ("dataloader", DataLoaderConfig),
    ]:
        kwargs = instantiate(cfg.dataset[section])
        assert isinstance(kwargs, expected), f"{config_path.stem}.{section} is {type(kwargs).__name__}"

    assert dataset._dataloader_kwargs["num_workers"] >= 0