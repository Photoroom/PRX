"""
MDS Streaming Patch

Patches ``device_per_stream`` batching in MosaicML Streaming so that streams
with too few samples are gracefully skipped instead of crashing.  The patch
is applied automatically when this module is imported.

Usage:
    from prx.dataset.mds_patches import streaming_patch  # patch applied on import
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from streaming.base.batching import batching_methods
from streaming.base.partition import get_partitions
from streaming.base.shuffle import get_shuffle
from streaming.base.world import World

if TYPE_CHECKING:
    from streaming.base.dataset import StreamingDataset

logger = logging.getLogger(__name__)


def _patched_generate_work_device_per_stream_batching(
    dataset: StreamingDataset, world: World, epoch: int, sample_in_epoch: int
) -> NDArray[np.int64]:
    """Generate this epoch's sample arrangement for ``device_per_stream`` batching.

    Drop-in replacement for the upstream implementation that adds two safety
    guards so training does not crash when a stream has very few samples:

    1. Streams with fewer samples than ``num_canonical_nodes`` are skipped
       (with a warning) instead of producing empty/broken partitions.
    2. ``shuffle_block_portion`` is clamped to ``max(1, ...)`` to avoid a
       zero-size shuffle block when a stream's proportion is tiny.
    """
    # Ensure that num_canonical_nodes has been set.
    if dataset.num_canonical_nodes is None:
        raise RuntimeError("`num_canonical_nodes` can never be None. Provide a positive integer.")

    if dataset.num_canonical_nodes % world.num_nodes != 0:
        logger.warning(
            "For `device_per_stream` batching, num_canonical_nodes must be divisible by physical nodes. "
            + f"Got {dataset.num_canonical_nodes} canonical nodes and {world.num_nodes} physical nodes. "
            + f"Setting num_canonical_nodes to {world.num_nodes}."
        )
        dataset.num_canonical_nodes = world.num_nodes

    partition_per_stream = []

    batch_size = dataset.batch_size
    assert isinstance(batch_size, int), f"Batch size must be an integer. Got {type(batch_size)}."

    for stream_id, stream in enumerate(dataset.streams):
        shuffle_units, small_per_big = dataset.resample_streams(epoch, stream_id)
        samples_in_stream = len(small_per_big)
        stream_partition = get_partitions(
            dataset.partition_algo,
            samples_in_stream,
            dataset.num_canonical_nodes,
            dataset.num_canonical_nodes,
            world.ranks_per_node,
            world.workers_per_rank,
            1,
            0,
            dataset.initial_physical_nodes,
        )
        if dataset.shuffle:
            # Skip streams that have fewer samples than canonical nodes –
            # they would produce empty/broken partitions.
            if samples_in_stream < dataset.num_canonical_nodes:
                logger.warning(
                    f"Because of the `device_per_stream` batching method, stream with index {stream_id} "
                    + f"has fewer samples ({samples_in_stream}) than canonical nodes "
                    + f"({dataset.num_canonical_nodes}); stream will be dropped."
                )
                continue

            if not isinstance(dataset.shuffle_block_size, int):
                raise TypeError(
                    "Dataset `shuffle_block_size` must be an integer. "
                    + f"Got {type(dataset.shuffle_block_size)} instead."
                )
            shuffle_block_portion = max(1, int(dataset.shuffle_block_size * stream.proportion))
            stream_shuffle = get_shuffle(
                dataset.shuffle_algo,
                shuffle_units,
                dataset.num_canonical_nodes,
                dataset.shuffle_seed,
                epoch,
                shuffle_block_portion,
            )
            stream_partition = np.where(stream_partition != -1, stream_shuffle[stream_partition], -1)
        partition_per_stream.append(np.where(stream_partition != -1, small_per_big[stream_partition], -1))

    # Merge per-stream partitions so each device batch has samples from a single stream.
    batches_per_stream = []
    batches_from_partitions = []
    ncn_per_node = dataset.num_canonical_nodes // world.num_nodes
    for node in range(world.num_nodes):
        per_node_stream_partitions = []
        per_node_batches_per_stream = []
        for stream_idx, partition in enumerate(partition_per_stream):
            stream_samples_inorder = (
                partition[node * ncn_per_node : (node + 1) * ncn_per_node].transpose(3, 2, 0, 1, 4).flatten()
            )
            padding_samples = batch_size - (stream_samples_inorder.size % batch_size)
            stream_samples_inorder = np.concatenate((stream_samples_inorder, np.full(padding_samples, -1)))
            stream_samples_inorder = stream_samples_inorder.reshape(-1, batch_size)
            num_full_batches = np.count_nonzero(np.min(stream_samples_inorder, axis=1) >= 0)
            per_node_batches_per_stream.append(num_full_batches)
            if num_full_batches != stream_samples_inorder.shape[0]:
                logger.warning(
                    "Because of the `device_per_stream` batching method, some batches with an inadequate "
                    + f"number of samples from stream with index {stream_idx} will be dropped."
                )
            if num_full_batches > 0:
                per_node_stream_partitions.append(stream_samples_inorder[:num_full_batches])
            else:
                logger.warning(
                    f"Stream with index {stream_idx} does not have an adequate number of "
                    + f"samples to construct even a single device batch of size {batch_size}. "
                    + "Training will occur without any samples from this stream!"
                )

        batches_per_stream.append(per_node_batches_per_stream)
        batches_from_partitions.append(per_node_stream_partitions)

    # Combine all device batches from all streams into one array, per node.
    all_partition_batches = []
    for node in range(world.num_nodes):
        all_partition_batches.append(np.concatenate(batches_from_partitions[node]))

    # Truncate all nodes to the minimum batch count so every node processes the
    # exact same number of real batches.  The previous approach padded shorter
    # nodes with -1 sentinel indices, but those are silently skipped by the
    # Streaming iterator, re-introducing a per-node batch-count imbalance that
    # leads to NCCL collective timeouts at epoch boundaries.
    min_device_batches = min(node_batches.shape[0] for node_batches in all_partition_batches)
    num_devices = world.num_nodes * world.ranks_per_node
    # Round down to a multiple of num_devices so the final reshape is clean.
    min_device_batches -= min_device_batches % num_devices

    epoch_seed = dataset.shuffle_seed + epoch if dataset.epoch_seed_change else dataset.shuffle_seed
    epoch_rng = np.random.default_rng(epoch_seed)

    for node in range(world.num_nodes):
        stream_origins = np.concatenate([np.full(n_batch, i) for i, n_batch in enumerate(batches_per_stream[node])])
        epoch_rng.shuffle(stream_origins)

        batch_indices = np.zeros(stream_origins.shape[0]).astype(np.int64)
        batch_offset = 0
        for i, n_device_batch in enumerate(batches_per_stream[node]):
            batch_indices[stream_origins == i] += batch_offset + np.arange(n_device_batch)
            batch_offset += n_device_batch

        all_partition_batches[node] = all_partition_batches[node][batch_indices]

        # Truncate to the synchronized batch count.
        all_partition_batches[node] = all_partition_batches[node][:min_device_batches]

    all_partition_batches_arr: np.ndarray = np.stack(all_partition_batches, axis=1).reshape(-1, batch_size)

    global_batch_size = batch_size * world.num_nodes * world.ranks_per_node
    if sample_in_epoch % global_batch_size != 0:
        logger.warning(
            "Because of the `device_per_stream` batching method, resumption may only occur on a sample "
            "that is a multiple of the current global batch size of "
            + str(global_batch_size)
            + ". Resuming training after the most recently finished global batch."
        )

    all_partition_batches_arr = all_partition_batches_arr.reshape(-1, global_batch_size)

    resumption_batch = sample_in_epoch // global_batch_size
    all_partition_batches_arr = all_partition_batches_arr[resumption_batch:]

    current_samples = all_partition_batches_arr.size
    divisibility_requirement = world.num_nodes * world.ranks_per_node * world.workers_per_rank * batch_size
    if current_samples % divisibility_requirement != 0:
        samples_needed = divisibility_requirement - (current_samples % divisibility_requirement)
        padding_batches_needed = samples_needed // global_batch_size
        all_partition_batches_arr = np.concatenate(
            (all_partition_batches_arr, np.full((padding_batches_needed, global_batch_size), -1))
        )

    return all_partition_batches_arr.reshape(
        -1, world.workers_per_rank, world.num_nodes, world.ranks_per_node, batch_size
    ).transpose(2, 3, 1, 0, 4)


def patch_mosaic_streaming() -> None:
    """Replace ``device_per_stream`` batching with a version that skips
    streams with too few samples instead of crashing."""
    batching_methods["device_per_stream"] = _patched_generate_work_device_per_stream_batching


# Auto-apply on import
patch_mosaic_streaming()
