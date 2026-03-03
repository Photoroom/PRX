# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mosaicml-streaming>=0.9",
#     "pillow>=10.0",
# ]
# ///
"""Convert locally-downloaded fine-t2i tar files to AR-bucketed MDS.

Splits all tar files across N worker processes. Each worker streams its
tars with tarfile, resizes images to 1024-base AR buckets (27 buckets,
patch_size=32), and writes to its own MDS shard directory. A final merge
step combines per-worker indexes into one index per AR bucket.

Download the dataset first:
    HF_HUB_CACHE=/path/to/cache huggingface-cli download ma-xu/fine-t2i --repo-type dataset

Then run:
    uv run fine_t2i_to_mds.py \\
        --input /path/to/cache/hub/datasets--ma-xu--fine-t2i/snapshots/<hash> \\
        --output /path/to/output/fine-t2i \\
        --workers 60
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import tarfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from glob import glob
from multiprocessing import Process, Value
from typing import Any

from PIL import Image
from streaming import MDSWriter

logger = logging.getLogger(__name__)

MDS_COLUMNS: dict[str, str] = {
    "image": "jpeg",
    "width": "int",
    "height": "int",
    "original_width": "int",
    "original_height": "int",
    "id": "str",
    "prompt": "str",
    "enhanced_prompt": "str",
    "aesthetic_predictor_v_2_5_score": "str",
    "image_generator": "str",
    "prompt_generator": "str",
    "prompt_category": "str",
    "style": "str",
    "task": "str",
    "enhancer": "str",
    "image_aspect_ratio": "str",
    "image_generated_with_enhanced_prompt": "str",
    "length": "int",
    "enhanced_length": "int",
    "subset": "str",
}

JPEG_QUALITY = 95
SHARD_SIZE = 1 << 27  # 128 MB


# ---------------------------------------------------------------------------
# AR bucketing at 1024 base, patch_size=32 → 27 buckets
# ---------------------------------------------------------------------------


def build_ar_to_size(
    base: int = 1024,
    patch: int = 32,
    min_ar: float = 0.5,
    max_ar: float = 2.0,
    div: int = 16,
) -> dict[float, tuple[int, int]]:
    bp = base / patch
    sizes: set[tuple[int, int]] = set()
    for pw in range(
        math.ceil(math.sqrt(bp**2 * min_ar)),
        math.floor(math.sqrt(bp**2 * max_ar)) + 1,
    ):
        ph = math.floor(bp**2 / pw)
        w, h = pw * patch, ph * patch
        if w % div == 0 and h % div == 0:
            sizes.add((w, h))
    for ph in range(
        math.ceil(math.sqrt(bp**2 / max_ar)),
        math.floor(math.sqrt(bp**2 / min_ar)) + 1,
    ):
        pw = math.floor(bp**2 / ph)
        w, h = pw * patch, ph * patch
        if w % div == 0 and h % div == 0:
            sizes.add((w, h))
    return {w / h: (w, h) for w, h in sizes}


AR_TO_SIZE = build_ar_to_size()
AR_KEYS = tuple(sorted(AR_TO_SIZE.keys()))


def closest_bucket(w: int, h: int) -> tuple[float, int, int]:
    ar = w / h
    best = min(AR_KEYS, key=lambda x: abs(x - ar))
    tw, th = AR_TO_SIZE[best]
    return best, tw, th


# ---------------------------------------------------------------------------
# Tar file streaming
# ---------------------------------------------------------------------------


def iter_tar_samples(tar_path: str) -> Iterator[dict[str, bytes]]:
    """Stream samples from a tar one at a time.

    WebDataset tars group consecutive files by key (e.g. 00001.jpg, 00001.json).
    Yields dicts with at least 'jpg' and 'json' keys.
    """
    current_key: str | None = None
    current: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            basename = member.name.rsplit("/", 1)[-1] if "/" in member.name else member.name
            key, _, ext = basename.rpartition(".")
            if not key:
                continue
            if key != current_key:
                if current_key is not None and "jpg" in current and "json" in current:
                    yield current
                current_key = key
                current = {}
            f = tf.extractfile(member)
            if f is not None:
                current[ext] = f.read()
        if current_key is not None and "jpg" in current and "json" in current:
            yield current


# ---------------------------------------------------------------------------
# Process one sample
# ---------------------------------------------------------------------------


def process_sample(
    image_bytes: bytes,
    meta_bytes: bytes,
    subset: str,
    jpeg_quality: int,
) -> tuple[dict[str, Any], str] | None:
    """Decode, resize to AR bucket, re-encode. Returns (mds_sample, ar_str) or None."""
    try:
        meta = json.loads(meta_bytes)

        img_res = meta.get("image_resolution")
        if img_res and len(img_res) == 2:
            orig_w, orig_h = int(img_res[0]), int(img_res[1])
        else:
            img = Image.open(io.BytesIO(image_bytes))
            orig_w, orig_h = img.size

        ar, tw, th = closest_bucket(orig_w, orig_h)
        ar_str = f"{ar:.3f}"

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((tw, th), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        jpeg_pil = Image.open(buf)
        jpeg_pil.load()

        task_val = meta.get("task")
        sample: dict[str, Any] = {
            "image": jpeg_pil,
            "width": tw,
            "height": th,
            "original_width": orig_w,
            "original_height": orig_h,
            "id": str(meta.get("id", "")),
            "prompt": str(meta.get("prompt", "")),
            "enhanced_prompt": str(meta.get("enhanced_prompt", "")),
            "aesthetic_predictor_v_2_5_score": str(meta.get("aesthetic_predictor_v_2_5_score", "")),
            "image_generator": str(meta.get("image_generator", "")),
            "prompt_generator": str(meta.get("prompt_generator", "")),
            "prompt_category": str(meta.get("prompt_category", "")),
            "style": str(meta.get("style", "")),
            "task": json.dumps(task_val) if task_val is not None else "",
            "enhancer": str(meta.get("enhancer", "")),
            "image_aspect_ratio": str(meta.get("image_aspect_ratio", "")),
            "image_generated_with_enhanced_prompt": str(
                meta.get("image_generated_with_enhanced_prompt", "")
            ),
            "length": int(meta.get("length", 0)),
            "enhanced_length": int(meta.get("enhanced_length", 0)),
            "subset": subset,
        }
        return sample, ar_str
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def worker_fn(
    worker_id: int,
    tar_files: list[str],
    output_root: str,
    jpeg_quality: int,
    counter: "Value[int]",
) -> None:
    log = logging.getLogger(f"w{worker_id}")
    writers: dict[str, MDSWriter] = {}
    n_written = 0
    n_failed = 0
    start = time.time()

    try:
        for tar_path in tar_files:
            subset = tar_path.rsplit("/", 2)[-2]

            for raw in iter_tar_samples(tar_path):
                result = process_sample(raw["jpg"], raw["json"], subset, jpeg_quality)
                if result is None:
                    n_failed += 1
                    continue

                sample, ar_str = result
                if ar_str not in writers:
                    out = os.path.join(output_root, ar_str, f"w{worker_id}")
                    os.makedirs(out, exist_ok=True)
                    writers[ar_str] = MDSWriter(
                        out=out, columns=MDS_COLUMNS, size_limit=SHARD_SIZE,
                    )
                writers[ar_str].write(sample)
                n_written += 1

                with counter.get_lock():
                    counter.value += 1

            if n_written > 0 and n_written % 1000 < 100:
                elapsed = time.time() - start
                log.info(
                    f"w{worker_id}: {n_written:,} written, "
                    f"{n_failed} failed, {n_written / elapsed:.0f}/s"
                )

    finally:
        for w in writers.values():
            w.finish()
        elapsed = time.time() - start
        rate = n_written / elapsed if elapsed > 0 else 0
        log.info(
            f"w{worker_id} done: {n_written:,} written, "
            f"{n_failed} failed in {elapsed:.0f}s ({rate:.0f}/s)"
        )


# ---------------------------------------------------------------------------
# Index merging
# ---------------------------------------------------------------------------


def merge_indexes(output_root: str) -> None:
    """Merge per-worker index.json files into one index per AR bucket."""
    import streaming.base.util

    for ar_dir in sorted(os.listdir(output_root)):
        ar_path = os.path.join(output_root, ar_dir)
        if not os.path.isdir(ar_path):
            continue
        worker_indexes = sorted(glob(os.path.join(ar_path, "w*", "index.json")))
        if not worker_indexes:
            continue
        logger.info(f"Merging {len(worker_indexes)} indexes for AR {ar_dir}")
        streaming.base.util.merge_index(worker_indexes, ar_path, keep_local=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    input_dir: str, output_root: str, num_workers: int, jpeg_quality: int,
) -> None:
    os.makedirs(output_root, exist_ok=True)

    tar_files = sorted(glob(os.path.join(input_dir, "*/train-*.tar")))
    logger.info(
        f"Found {len(tar_files)} tar files | "
        f"AR buckets: {len(AR_TO_SIZE)} | workers: {num_workers} | "
        f"shard_size: {SHARD_SIZE // (1 << 20)} MB"
    )

    # Round-robin distribute tars across workers
    worker_tars: list[list[str]] = [[] for _ in range(num_workers)]
    for i, tf in enumerate(tar_files):
        worker_tars[i % num_workers].append(tf)

    counter: Value[int] = Value("i", 0)
    start = time.time()

    processes: list[Process] = []
    for i in range(num_workers):
        if not worker_tars[i]:
            continue
        p = Process(
            target=worker_fn,
            args=(i, worker_tars[i], output_root, jpeg_quality, counter),
        )
        p.start()
        processes.append(p)

    logger.info(f"Launched {len(processes)} workers")

    while any(p.is_alive() for p in processes):
        time.sleep(30)
        elapsed = time.time() - start
        total = counter.value
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(f"Progress: {total:,} samples, {rate:.0f}/s, {elapsed / 60:.0f}m elapsed")

    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        logger.error(f"{len(failed)} workers failed")

    elapsed = time.time() - start
    total = counter.value
    logger.info(
        f"Processing done: {total:,} samples in {elapsed / 60:.0f}m ({total / elapsed:.0f}/s)"
    )

    logger.info("Merging AR indexes...")
    merge_indexes(output_root)

    logger.info(f"Output at {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fine-t2i WebDataset tars to AR-bucketed MDS"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing subset/train-*.tar files",
    )
    parser.add_argument("--output", default="/mnt/data/datasets/gen_ai/fine-t2i")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(args.input, args.output, args.workers, args.jpeg_quality)


if __name__ == "__main__":
    main()

