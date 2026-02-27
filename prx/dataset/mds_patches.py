"""
MDS Streaming Patches

This module provides patches to MosaicML Streaming:
1. Custom MDS encoding support (e.g. bfloat16 tensors)
2. Patched ``device_per_stream`` batching that gracefully skips streams with too
   few samples instead of crashing.

Usage:
    from dataset.mds_patches import patch_mosaic_streaming

    # Apply all streaming patches (encoding + batching)
    patch_mosaic_streaming()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from streaming.base.batching import batching_methods
from streaming.base.partition import get_partitions
from streaming.base.shuffle import get_shuffle
from streaming.base.world import World

if TYPE_CHECKING:
    from streaming.base.dataset import StreamingDataset

try:
    from streaming.base.format.mds.encodings import Encoding
except ImportError as e:
    raise ImportError(
        "MDS patches require the 'streaming' library to be installed. "
        "Install it with: pip install mosaicml-streaming"
    ) from e

logger = logging.getLogger(__name__)


# Global state for patching
_custom_encodings = {}
_is_patched = False


class BFloat16TensorEncoding(Encoding):
    """
    Custom encoding for bfloat16 tensors following MDS Encoding pattern.

    Creates a compact binary format:
    - Number of dimensions (4 bytes): uint32
    - Shape (ndim * 4 bytes): int32 array
    - Data (variable): raw tensor bytes as uint16
    """

    def encode(self, obj: torch.Tensor) -> bytes:
        """Encode a bfloat16 tensor to bytes."""
        if not isinstance(obj, torch.Tensor) or obj.dtype != torch.bfloat16:
            raise TypeError(f"Expected bfloat16 tensor, got {type(obj)} with dtype {getattr(obj, 'dtype', 'N/A')}")

        # Pack: ndim + shape + data
        shape = np.array(obj.shape, dtype=np.int32)
        ndim = np.uint32(len(shape))
        tensor_bytes = obj.view(torch.uint16).numpy().tobytes()

        result: bytes = ndim.tobytes() + shape.tobytes() + tensor_bytes
        return result

    def decode(self, data: bytes) -> torch.Tensor:
        """Decode bytes back to a bfloat16 tensor."""
        if not isinstance(data, bytes):
            raise ValueError("Expected bytes data")

        offset = 0

        # Read number of dimensions
        ndim = np.frombuffer(data[offset : offset + 4], dtype=np.uint32)[0]
        offset += 4

        # Read shape
        shape = np.frombuffer(data[offset : offset + ndim * 4], dtype=np.int32)
        offset += ndim * 4

        # Read tensor data
        tensor_bytes = data[offset:]
        flat_uint16 = np.frombuffer(tensor_bytes, dtype=np.uint16).copy()
        uint16_tensor = torch.from_numpy(flat_uint16)

        return uint16_tensor.view(torch.bfloat16).reshape(tuple(shape.astype(int)))


def register_custom_encoding(name: str, encoding_class: type[Encoding]) -> None:
    """
    Register a custom encoding class (not instance) with the given name.

    Args:
        name: The encoding name (e.g., "bf16")
        encoding_class: Encoding implementation (class) with encode() and decode() methods, subclass of Encoding
    """
    _custom_encodings[name] = encoding_class

    # If patches are already applied, add to the MDS encodings dict immediately
    if _is_patched:
        try:
            import streaming.base.format.mds.encodings as mds_encodings

            mds_encodings._encodings[name] = encoding_class
        except ImportError:
            pass  # Streaming library not available


def patch_mds_encoding() -> None:
    """
    Patch the MDS encoding system to support custom types by directly extending
    the encodings dictionary.

    This is much simpler than patching individual functions and automatically
    works everywhere the _encodings dict is used.

    Example:
        patch_mds_encoding()

        # Writing data
        sample = {'tensor': torch.randn(3, 4, dtype=torch.bfloat16)}
        columns = {'tensor': 'bf16'}

        with MDSWriter(out=path, columns=columns) as writer:
            writer.write(sample)  # Automatically encodes the bfloat16 tensor

        # Reading data - automatic deserialization
        dataset = StreamingDataset(local=path)
        for item in dataset:
            bf16_tensor = item['tensor']  # Automatically decoded as torch.bfloat16
            assert bf16_tensor.dtype == torch.bfloat16
    """
    global _is_patched

    if _is_patched:
        return  # Already patched

    try:
        import streaming.base.format.mds.encodings as mds_encodings

        # Add all our custom encodings to the MDS encodings registry
        for name, encoding in _custom_encodings.items():
            mds_encodings._encodings[name] = encoding

        _is_patched = True
        print("✅ MDS encoding patches applied")

    except ImportError:
        print("⚠️  Warning: streaming library not available, patches not applied")


def unpatch_mds_encoding() -> None:
    """
    Remove custom encodings from the MDS encoding system.

    Useful for testing or cleanup.
    """
    global _is_patched

    if not _is_patched:
        return  # Not patched

    try:
        import streaming.base.format.mds.encodings as mds_encodings

        # Remove all our custom encodings from the MDS encodings registry
        for name in _custom_encodings.keys():
            if name in mds_encodings._encodings:
                del mds_encodings._encodings[name]

        _is_patched = False
        print("🧹 MDS encoding patches removed")

    except ImportError:
        print("⚠️  Warning: streaming library not available, cannot unpatch")


def is_patched() -> bool:
    """Check if MDS encoding is currently patched."""
    return _is_patched


# Register the bfloat16 encoding class (not instance)
register_custom_encoding("bf16", BFloat16TensorEncoding)


# ---------------------------------------------------------------------------
# Patched device_per_stream batching
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def patch_mosaic_streaming() -> None:
    """Apply all MosaicML Streaming patches.

    1. Replace ``device_per_stream`` batching with a version that skips
       streams with too few samples instead of crashing.
    2. Register custom MDS encodings (e.g. bfloat16 tensor support).
    """
    batching_methods["device_per_stream"] = _patched_generate_work_device_per_stream_batching
    patch_mds_encoding()
