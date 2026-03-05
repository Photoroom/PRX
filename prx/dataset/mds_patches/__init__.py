"""
MDS Patches

Individual patches for MosaicML Streaming. Each submodule auto-applies its
patch on import:

    from prx.dataset.mds_patches import encoding_patch   # encoding only
    from prx.dataset.mds_patches import streaming_patch   # batching only
    from prx.dataset.mds_patches import *                 # both patches
"""

__all__ = ["encoding_patch", "streaming_patch"]
