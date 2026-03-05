import json
import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any


from torch.utils.data import DataLoader

from streaming.base.constant import TICK
from streaming.base.format import Reader, reader_from_json
from streaming import Stream, StreamingDataset
from streaming.base.util import wait_for_file_to_exist
from streaming.base.world import World

from .constants import BatchKeys
from .dataset import ProcessedDataset, DEFAULT_DATA_AUG_TARGETS
from .dataset import logger
from .mds_patches import *  # noqa: F401,F403 — applies encoding + streaming patches on import

INDEX_FILE = "index.json"
INDEX_FILE_SUFFIX = "_index.json"

def get_nb_samples_in_stream(index_file: str) -> int:
    total_samples = 0
    with open(index_file, "r") as f:
        index = json.load(f)
    for shard in index["shards"]:
        total_samples += shard["samples"]
    return total_samples


def get_split_folders(path: str, file_name: str = INDEX_FILE) -> list[str]:
    """Get list of folders containing the specified index file.

    Args:
        path: Local filesystem path to search
        file_name: Name of the index file to look for (default: INDEX_FILE)

    Returns:
        List of folder paths containing the index file
    """
    if os.path.isfile(os.path.join(path, file_name)):
        # Index file exists in the root path
        return [path]

    # Check if the index file exists in any subfolder
    sub_folders = []
    for sub_folder in [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]:
        if os.path.isfile(os.path.join(path, sub_folder, file_name)):
            sub_folders.append(os.path.join(path, sub_folder))

    return sub_folders


def get_stream_iterator(
    local: str | list[str],
    remote: str | list[str] | None,
    proportions: float | list[float] | None,
) -> Iterator[tuple[str | None, str, str, float | None]]:
    """Get iterator over (remote, local, index_file, proportion) tuples."""
    if remote is None:
        return get_local_iterator(local, proportions)

    if proportions is not None:
        warnings.warn("Proportions are ignored when using remote datasets")
    return get_remote_iterator(remote, local)


def get_local_iterator(
    local_paths: str | list[str],
    proportions: float | list[float] | None
) -> Iterator[tuple[str | None, str, str, float | None]]:
    """Iterate over local dataset paths."""
    if isinstance(local_paths, str):
        local_paths = [local_paths]

    for idx, local_path in enumerate(local_paths):
        # Get proportion for this path
        if isinstance(proportions, Sequence):
            proportion = proportions[idx]
        else:
            proportion = proportions

        # Check for index files in root
        index_files = [f for f in os.listdir(local_path) if f.endswith(INDEX_FILE_SUFFIX)]

        if index_files:
            # Multiple streams in root - use cache workaround
            logger.info(
                "Multiple streams in root directory. Using remote/local cache workaround."
            )
            tmp_path = local_path.rstrip("/").split("/")[-1]
            stream_props = split_proportion(proportion, index_files) if proportion else [None] * len(index_files)

            for i, (index_file, prop) in enumerate(zip(index_files, stream_props)):
                yield local_path, f"/tmp/{tmp_path}_{i}/", index_file, prop
        else:
            # Check subfolders
            folders = get_split_folders(local_path, INDEX_FILE)
            stream_props = split_proportion(proportion, folders) if proportion else [None] * len(folders)

            for folder, prop in zip(folders, stream_props):
                yield None, folder, INDEX_FILE, prop


def get_remote_iterator(
    remote_paths: str | list[str],
    local_paths: str | list[str]
) -> Iterator[tuple[str, str, str, None]]:
    """Iterate over remote dataset paths with local cache."""
    # Normalize to lists
    if isinstance(remote_paths, str) and isinstance(local_paths, str):
        remote_paths, local_paths = [remote_paths], [local_paths]

    # Handle length mismatch
    if isinstance(remote_paths, Sequence) and isinstance(local_paths, Sequence):
        if len(remote_paths) != len(local_paths):
            if len(remote_paths) < len(local_paths):
                raise ValueError("More local paths than remote paths")

            # Create cache subfolders
            local_paths = list(local_paths)
            base_local = local_paths[-1]
            local_paths[-1] = f"{base_local}/cache_0"

            for i in range(len(remote_paths) - len(local_paths)):
                local_paths.append(f"{base_local}/cache_{i + 1}")

            warnings.warn(f"Created cache subfolders: {local_paths}")

            # Ensure directories exist
            for path in local_paths:
                os.makedirs(path, exist_ok=True)

    # Iterate over pairs
    for remote_path, local_path in zip(remote_paths, local_paths):
        remote_folders = get_split_folders(remote_path, INDEX_FILE)
        local_folders = [os.path.join(local_path, str(i)) for i in range(len(remote_folders))]

        for remote, local in zip(remote_folders, local_folders):
            yield remote, local, INDEX_FILE, None


def split_proportion(proportion: float | None, items: list[Any]) -> list[float | None]:
    """Split a proportion across multiple items based on their sample counts."""
    if proportion is None:
        return [None] * len(items)

    # Get sample counts
    counts = [get_nb_samples_in_stream(item) for item in items]
    total = sum(counts)

    # Distribute proportion
    return [(count / total) * proportion for count in counts]


class PatchedStream(Stream):
    """
    Update the get_shards method to take any index file not only index.json
    """

    def __init__(self, index_file: str = INDEX_FILE, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.index_file = index_file

    def get_shards(self, world: World, allow_unsafe_types: bool) -> list[Reader]:
        """Load this Stream's index, retrieving its shard readers.

        Args:
            world (World): Distributed context.
            allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
                execution during deserialization, whether to keep going if ``True`` or raise an
                error.

        Returns:
            `list[Reader]: Shard readers.
        """
        basename = self.index_file
        filepath = os.path.join(self.local, self.split, basename)  # pyright: ignore
        if not os.path.exists(filepath):
            if world.is_local_leader:
                if self.remote:
                    # Downloads the `index.json` as `index.json.tmp` fully and then rename it to
                    # `index.json` since only one process downloads the `index.json` file while
                    # other processes wait for it to get downloaded. Hence, It avoids loading the
                    # in-progress downloading `index.json`.
                    tmp_filepath = self._download_file(basename, basename + ".tmp")
                    os.rename(tmp_filepath, filepath)
                else:
                    if not os.path.exists(filepath):
                        raise RuntimeError(
                            f"No `remote` provided, but local file {filepath} " + "does not exist either"
                        )
            else:
                wait_for_file_to_exist(
                    filepath,
                    TICK,
                    self.download_timeout,
                    f"Index file {os.path.join(self.remote or '', self.split or '', basename)} "
                    + f"-> {filepath} took too long to download or failed to download. Either increase the "
                    + "`download_timeout` value or check the local rank 0 traceback.",
                )
        try:
            obj = json.load(open(filepath))
        except json.decoder.JSONDecodeError as error:
            error.args = (f"Index file at {filepath} is empty or corrupted. " + error.args[0],)
            raise error

        # Version check.
        if obj["version"] != 2:
            raise ValueError(f"Unsupported streaming data version: {obj['version']}. Expected version 2.")

        # Initialize shard readers according to the loaded info.
        shards = []
        for info in obj["shards"]:
            shard = reader_from_json(self.local, self.split, info)
            shard.validate(allow_unsafe_types)
            shards.append(shard)

        return shards

    def get_index_size(self) -> int:
        """Get the size of the index file in bytes.

        Returns:
            int: Size in bytes.
        """
        filepath = os.path.join(self.local, self.split, self.index_file)
        return os.stat(filepath).st_size


class StreamingProcessedDataset(StreamingDataset, ProcessedDataset):
    """Dataset that combines MosaicML streaming with sample processing.

    Args:
        local: Local path(s) to dataset folders.
        remote: Remote path(s) to dataset folders (optional).
        caption_keys: Caption field name(s), optionally with sampling weights.
        text_tower: Name of the text encoder preset for latent lookups.
        prompt_max_tokens: Maximum sequence length for text embeddings.
        download_retry: Number of download retries per shard.
        download_timeout: Timeout in seconds for shard downloads.
        predownload: Number of shards to pre-download per worker.
        cache_limit: Maximum cache size (e.g., "1tb", "500gb").
        num_canonical_nodes: Number of canonical nodes for deterministic shuffling.
        batch_size: Batch size for batching method calculations.
        shuffle: Whether to shuffle samples.
        shuffle_seed: Random seed for shuffling.
        proportions: Sampling proportions for each path.
        has_text_latents: Whether samples contain precomputed text latents.
        has_mask_text_latents: Whether samples contain attention masks for text latents.
        batching_method: How to batch across streams ("per_stream" or "random").
        transforms: List of transforms to apply to images.
        transforms_targets: Image keys to apply transforms to.
        drop_last: Whether to drop the last incomplete batch (default: True).
        prefetch_factor: Number of batches to prefetch per worker (optional).
        num_workers: Number of worker processes for data loading (default: 0).
        persistent_workers: Whether to keep workers alive between epochs (default: False).
        pin_memory: Whether to use pinned memory for faster GPU transfer (default: False).
    """

    def __init__(
        self,
        local: str | list[str],
        remote: str | list[str] | None = None,
        caption_keys: str | list[str] | list[tuple[str, float]] = "caption",
        text_tower: str = "t5gemma2b-256-bf16",
        prompt_max_tokens: int = 256,
        download_retry: int = 2,
        download_timeout: float = 120,
        predownload: int | None = None,
        cache_limit: str = "1tb",
        num_canonical_nodes: int | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 9146,
        proportions: float | list[float] | None = None,
        has_text_latents: bool = True,
        has_mask_text_latents: bool = False,
        batching_method: str = "per_stream",
        transforms: list[Callable] | None = None,
        transforms_targets: list[str] | str = DEFAULT_DATA_AUG_TARGETS,
        drop_last: bool = True,
        prefetch_factor: int | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ):
        # Build streams from paths
        streams = []
        for r_path, l_path, index_file, proportion in get_stream_iterator(local, remote, proportions):
            if proportion is not None and proportion <= 0:
                raise ValueError(f"Proportion must be positive, got {proportion}")
            streams.append(
                PatchedStream(
                    remote=r_path,
                    local=l_path,
                    download_retry=download_retry,
                    download_timeout=download_timeout,
                    index_file=index_file,
                    proportion=proportion,
                )
            )

        StreamingDataset.__init__(
            self,
            streams=streams,
            remote=None,
            local=None,
            split=None,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            keep_zip=False,
            predownload=predownload,
            cache_limit=cache_limit,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            batching_method=batching_method,
        )
        ProcessedDataset.__init__(
            self,
            caption_keys=caption_keys,
            text_tower=text_tower,
            prompt_max_tokens=prompt_max_tokens,
            has_text_latents=has_text_latents,
            has_mask_text_latents=has_mask_text_latents,
            transforms=transforms,
            transforms_targets=transforms_targets,
        )

        # Store dataloader kwargs
        self._dataloader_kwargs = {
            "drop_last": drop_last,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
        }
        if prefetch_factor is not None:
            self._dataloader_kwargs["prefetch_factor"] = prefetch_factor

        # Log summary
        logger.info(
            "--- Dataset Summary ---\n"
            f"Remote: {remote} | Local: {local}\n"
            f"Total size: {self.size} | This rank: {len(self)}\n"
            f"Sum of stream samples: {sum(self.samples_per_stream)}"
        )
        for i, (stream, n_samples) in enumerate(zip(self.streams, self.samples_per_stream)):
            location = stream.remote or stream.local
            index = getattr(stream, 'index_file', INDEX_FILE)
            logger.info(f"  Stream {i}: {location}/{index} - {n_samples} samples")
        logger.info("-" * 23)

    def __getitem__(self, index: int) -> dict[BatchKeys, Any] | None:
        return ProcessedDataset.__getitem__(self, index)

    def _get_raw_item(self, index: int) -> dict[str, Any]:
        return StreamingDataset.__getitem__(self, index)

    def _get_sampler(self, shuffle: bool) -> None:
        # StreamingDataset is an IterableDataset — it handles shuffling/sharding
        # internally, so DataLoader must not receive a sampler.
        return None

    def get_dataloader(self, batch_size: int, **kwargs: Any) -> DataLoader:
        """Create a DataLoader using stored kwargs from config."""
        # StreamingDataset needs batch_size set before iteration for deterministic
        # resumption and optimal worker partitioning.
        self.batch_size = batch_size
        merged_kwargs = {**self._dataloader_kwargs, **kwargs}
        return super().get_dataloader(batch_size=batch_size, **merged_kwargs)
