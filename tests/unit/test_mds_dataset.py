from unittest.mock import PropertyMock, patch

import pytest

from prx.dataset.mds_dataset import (
    DataLoaderConfig,
    ProcessedConfig,
    StreamingConfig,
    StreamingProcessedDataset
)

@pytest.mark.unit
class TestStreamingProcessedDataset:
    @patch("prx.dataset.mds_dataset.StreamingProcessedDataset.size", new_callable = PropertyMock)
    @patch("prx.dataset.mds_dataset.StreamingProcessedDataset.__len__", return_value = 10)
    @patch("prx.dataset.mds_dataset.ProcessedDataset.__init__", return_value = None)
    @patch("prx.dataset.mds_dataset.StreamingDataset.__init__")
    @patch("prx.dataset.mds_dataset.PatchedStream")
    @patch("prx.dataset.mds_dataset.get_stream_iterator")
    

    def test_forwards_config_values(
        self,
        mock_get_stream_iterator,
        mock_patched_stream,
        mock_streaming_init,
        mock_processed_init,
        mock_len,
        mock_size
    ) -> None:
        """Verify that configuration dataclasses are forwarded correctly to the main dataset"""

        def fake_streaming_init(instance, **kwargs):
            instance.samples_per_stream = [10]
            instance.streams = kwargs["streams"]

        mock_size.return_value = 10
        mock_streaming_init.side_effect = fake_streaming_init
        mock_get_stream_iterator.return_value = iter([("remote/path", "local/path", "custom_index.json", 0.75)])

        streaming = StreamingConfig(
            download_retry = 5,
            download_timeout = 30.0,
            predownload = 8,
            cache_limit = "10gb",
            num_canonical_nodes = 2,
            batch_size = 16,
            shuffle = True,
            shuffle_seed = 4096,
            batching_method = "random"
        )

        processed = ProcessedConfig(
            caption_keys = ["caption", "alt_text"],
            text_tower = "test-tower",
            prompt_max_tokens = 128,
            has_text_latents = False,
            has_mask_text_latents = True,
            transforms = None,
            transforms_targets = ["image"]
        )

        dataloader = DataLoaderConfig(
            drop_last = False,
            prefetch_factor = 4,
            num_workers = 2,
            persistent_workers = True,
            pin_memory = True,
        )

        dataset = StreamingProcessedDataset(
            local = "local/path",
            remote = "remote/path",
            proportions = 0.75,
            streaming = streaming,
            processed = processed,
            dataloader = dataloader,
        )

        mock_get_stream_iterator.assert_called_once_with(
            "local/path",
            "remote/path",
            0.75,
        )

        mock_patched_stream.assert_called_once_with(
            remote="remote/path",
            local="local/path",
            download_retry=5,
            download_timeout=30.0,
            index_file="custom_index.json",
            proportion=0.75,
        )

        mock_streaming_init.assert_called_once_with(
            dataset,
            streams=[mock_patched_stream.return_value],
            remote=None,
            local=None,
            split=None,
            download_retry=5,
            download_timeout=30.0,
            validate_hash=None,
            keep_zip=False,
            predownload=8,
            cache_limit="10gb",
            num_canonical_nodes=2,
            batch_size=16,
            shuffle=True,
            shuffle_seed=4096,
            batching_method="random",
        )

        mock_processed_init.assert_called_once_with(
            dataset,
            caption_keys=["caption", "alt_text"],
            text_tower="test-tower",
            prompt_max_tokens=128,
            has_text_latents=False,
            has_mask_text_latents=True,
            transforms=None,
            transforms_targets=["image"],
        )

        assert dataset._dataloader_kwargs == {
            "drop_last": False,
            "num_workers": 2,
            "persistent_workers": True,
            "pin_memory": True,
            "prefetch_factor": 4,
        }

    @patch("prx.dataset.mds_dataset.StreamingProcessedDataset.size",new_callable=PropertyMock)
    @patch("prx.dataset.mds_dataset.StreamingProcessedDataset.__len__",return_value=10)
    @patch("prx.dataset.mds_dataset.ProcessedDataset.__init__",return_value=None)
    @patch("prx.dataset.mds_dataset.StreamingDataset.__init__")
    @patch("prx.dataset.mds_dataset.PatchedStream")
    @patch("prx.dataset.mds_dataset.get_stream_iterator")

    def test_omits_prefetch_factor_when_none(
        self,
        mock_get_stream_iterator,
        mock_patched_stream,
        mock_streaming_init,
        mock_processed_init,
        mock_len,
        mock_size,
    ) -> None:
        """Verify that prefetch_factor gets omitted when set to None"""
        def fake_streaming_init(instance, **kwargs):
            instance.samples_per_stream = [10]
            instance.streams = kwargs["streams"]

        mock_size.return_value = 10
        mock_streaming_init.side_effect = fake_streaming_init

        mock_get_stream_iterator.return_value = iter([(None,"local/path","index.json",None)])

        streaming = StreamingConfig()

        processed = ProcessedConfig(
            caption_keys=["caption"],
            text_tower="test-tower",
            prompt_max_tokens=128,
            has_text_latents=False,
            has_mask_text_latents=False,
            transforms=None,
            transforms_targets=["image"],
        )

        dataloader = DataLoaderConfig(
            drop_last=False,
            prefetch_factor=None,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )

        dataset = StreamingProcessedDataset(
            local="local/path",
            streaming=streaming,
            processed=processed,
            dataloader=dataloader,
        )

        assert dataset._dataloader_kwargs == {
            "drop_last": False,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
        }

        assert "prefetch_factor" not in dataset._dataloader_kwargs

    @patch("prx.dataset.mds_dataset.StreamingProcessedDataset.size", new_callable=PropertyMock)
    @patch("prx.dataset.mds_dataset.StreamingProcessedDataset.__len__", return_value=10)
    @patch("prx.dataset.mds_dataset.ProcessedDataset.__init__", return_value=None)
    @patch("prx.dataset.mds_dataset.StreamingDataset.__init__")
    @patch("prx.dataset.mds_dataset.PatchedStream")
    @patch("prx.dataset.mds_dataset.get_stream_iterator")

    def test_invalid_proportion_raises(
        self,
        mock_get_stream_iterator,
        mock_patched_stream,
        mock_streaming_init,
        mock_processed_init,
        mock_len,
        mock_size,
    ):
        """Verify that proportions are getting rejected if negative"""
        mock_size.return_value = 10

        mock_get_stream_iterator.return_value = iter([(None,"local/path","index.json",-1.0)])

        streaming = StreamingConfig()
        processed = ProcessedConfig(
            caption_keys=["caption"],
            text_tower="test-tower",
            prompt_max_tokens=128,
            has_text_latents=False,
            has_mask_text_latents=False,
            transforms=None,
            transforms_targets=["image"],
        )
        dataloader = DataLoaderConfig()

        with pytest.raises(ValueError, match="Proportion must be positive"):
            StreamingProcessedDataset(
                local="local/path",
                streaming=streaming,
                processed=processed,
                dataloader=dataloader,
            )