"""
MDS Encoding Patch

Adds custom encoding support (e.g. bfloat16 tensors) to MosaicML Streaming's
MDS format. The patch is applied automatically when this module is imported.

Usage:
    from prx.dataset.mds_patches import encoding_patch  # patch applied on import
"""

import numpy as np
import torch

try:
    from streaming.base.format.mds.encodings import Encoding
except ImportError as e:
    raise ImportError(
        "MDS patches require the 'streaming' library to be installed. "
        "Install it with: pip install mosaicml-streaming"
    ) from e


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

# Auto-apply on import
patch_mds_encoding()
