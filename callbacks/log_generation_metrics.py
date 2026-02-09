"""
Callback for computing generation quality metrics (FID, CMMD, DINO-MMD) during evaluation,
with configurable frequency like '100ba', '1ep', '5000sp', or an int (batches).

CMMD here matches sayakpaul/cmmd-pytorch's distance.py:
- Gaussian RBF kernel
- _SIGMA = 10
- _SCALE = 1000
- "minimum-variance/biased" style estimator (mean over ALL pairs, including diagonals)
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from composer import Callback, Logger, State
from composer.core import TimeUnit
from composer.utils import dist
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance

from callbacks.feature_extractors import CLIPFeatureExtractor, DINOFeatureExtractor
from dataset.constants import BatchKeys
from pipeline.pipeline import LatentDiffusion

_logger = logging.getLogger(__name__)


def _rbf_kernel_sum_chunked(x: torch.Tensor, y: torch.Tensor, gamma: float, chunk_size: int) -> torch.Tensor:
    """Returns sum_{i,j} exp(-gamma * ||x_i - y_j||^2) computed in chunks over x to save memory."""
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[1]
    n, d = x.shape
    m, _ = y.shape

    # Precompute norms for y once
    y_sq = (y * y).sum(dim=1).unsqueeze(0)  # (1, m)

    total = x.new_zeros(())
    for i in range(0, n, chunk_size):
        x_chunk = x[i : i + chunk_size]  # (c, d)
        x_sq = (x_chunk * x_chunk).sum(dim=1).unsqueeze(1)  # (c, 1)
        # dist^2 = ||x||^2 + ||y||^2 - 2 x y^T
        dist2 = x_sq + y_sq - 2.0 * (x_chunk @ y.t())
        total = total + torch.exp(-gamma * dist2).sum()

    return total


def _cmmd_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 10.0,
    scale: float = 1000.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """CMMD MMD estimator:
    - k_xx = mean over ALL (i,j) in x×x (includes diagonal)
    - k_yy = mean over ALL (i,j) in y×y (includes diagonal)
    - k_xy = mean over ALL (i,j) in x×y
    - return scale * (k_xx + k_yy - 2*k_xy)
    """
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    n = x.shape[0]
    m = y.shape[0]
    if n == 0 or m == 0:
        return torch.tensor(float("nan"), device=x.device)

    gamma = 1.0 / (2.0 * (sigma**2))

    # sums
    s_xx = _rbf_kernel_sum_chunked(x, x, gamma=gamma, chunk_size=chunk_size)
    s_yy = _rbf_kernel_sum_chunked(y, y, gamma=gamma, chunk_size=chunk_size)
    s_xy = _rbf_kernel_sum_chunked(x, y, gamma=gamma, chunk_size=chunk_size)

    k_xx = s_xx / float(n * n)
    k_yy = s_yy / float(m * m)
    k_xy = s_xy / float(n * m)

    return x.new_tensor(scale) * (k_xx + k_yy - 2.0 * k_xy)


def _gather_cat_to_rank0(t: torch.Tensor) -> Optional[torch.Tensor]:
    """Gather variable-length 2D tensors (N_i, D) from all ranks to rank 0 and concatenate."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return t

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor to gather, got shape {tuple(t.shape)}")

    # Gather sizes
    n_local = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(n_local) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, n_local)
    sizes_int = [int(s.item()) for s in sizes]
    max_n = max(sizes_int)

    # Pad to max_n along dim 0
    if t.shape[0] < max_n:
        pad = torch.zeros((max_n - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t

    if rank == 0:
        gather_list = [torch.empty_like(t_pad) for _ in range(world_size)]
        torch.distributed.gather(t_pad, gather_list=gather_list, dst=0)
        pieces = [g[:sz] for g, sz in zip(gather_list, sizes_int)]
        return torch.cat(pieces, dim=0)
    else:
        torch.distributed.gather(t_pad, gather_list=None, dst=0)
        return None


class RBFMMDMetric(Metric):
    """A torchmetrics.Metric that accumulates embeddings and computes CMMD-style MMD at compute().

    - update(images, real=True/False): extracts embeddings, stores them (optionally on CPU)
    - compute(): gathers embeddings to rank 0, computes MMD, broadcasts scalar back to all ranks
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        feature_extractor: nn.Module,
        sigma: float = 10.0,
        scale: float = 1000.0,
        chunk_size: int = 1024,
        store_on_cpu: bool = True,
    ):
        super().__init__(dist_sync_on_step=False)

        self.feature_extractor = feature_extractor
        self.sigma = float(sigma)
        self.scale = float(scale)
        self.chunk_size = int(chunk_size)
        self.store_on_cpu = bool(store_on_cpu)

        # Keep per-rank lists; we'll do our own distributed gather on compute.
        self.add_state("real_feats", default=[], dist_reduce_fx=None)
        self.add_state("fake_feats", default=[], dist_reduce_fx=None)

    @torch.inference_mode()
    def update(self, images: torch.Tensor, real: bool) -> None:
        feats = self.feature_extractor(images)  # (N, D), on extractor device
        feats = feats.detach().float()

        if self.store_on_cpu:
            feats = feats.cpu()

        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        if len(self.real_feats) == 0 or len(self.fake_feats) == 0:
            return torch.tensor(float("nan"), device=self.device)

        real_local = torch.cat(list(self.real_feats), dim=0)
        fake_local = torch.cat(list(self.fake_feats), dim=0)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            backend_str = backend if isinstance(backend, str) else str(backend)
            backend_str = backend_str.lower()

            if backend_str == "nccl":
                if not torch.cuda.is_available():
                    raise RuntimeError("NCCL backend requires CUDA tensors, but CUDA is not available.")
                compute_device = torch.device("cuda", torch.cuda.current_device())
            else:
                compute_device = torch.device("cpu")

            real_local = real_local.to(compute_device, non_blocking=True)
            fake_local = fake_local.to(compute_device, non_blocking=True)

            rank = torch.distributed.get_rank()
            real_global = _gather_cat_to_rank0(real_local)  # must be on compute_device
            fake_global = _gather_cat_to_rank0(fake_local)

            if rank == 0:
                out = _cmmd_distance(
                    real_global, fake_global, sigma=self.sigma, scale=self.scale, chunk_size=self.chunk_size
                ).detach()
            else:
                out = torch.zeros((), device=compute_device, dtype=torch.float32)

            torch.distributed.broadcast(out, src=0)
            return out

        # single-process
        device = real_local.device
        return _cmmd_distance(
            real_local.to(device), fake_local.to(device), sigma=self.sigma, scale=self.scale, chunk_size=self.chunk_size
        )

        def reset(self) -> None:
            super().reset()
            self.real_feats.clear()
            self.fake_feats.clear()


@dataclass
class _Frequency:
    value: int
    unit: TimeUnit  # BA, EP, or SP

    @classmethod
    def from_input(cls, x: Union[int, str]) -> "_Frequency":
        if isinstance(x, int):
            return cls(x, TimeUnit.BATCH)

        if isinstance(x, str):
            m = re.fullmatch(r"\s*([\d_]+)\s*(ba|ep|sp)\s*", x.lower())
            if not m:
                raise ValueError(f"Invalid frequency string: {x!r}. Use like '100ba', '1ep', or '5000sp'.")
            v_str, u = m.group(1), m.group(2)
            v = int(v_str.replace("_", ""))
            unit = {"ba": TimeUnit.BATCH, "ep": TimeUnit.EPOCH, "sp": TimeUnit.SAMPLE}[u]
            return cls(v, unit)


class LogQualityMetrics(Callback):
    """
    Computes generation quality metrics (FID, CMMD, DINO-MMD) during evaluation.

    This callback is designed to periodically compute and log various image generation
    quality metrics, including Frechet Inception Distance (FID), CLIP-based MMD (CMMD),
    and DINO-based MMD, during evaluation phases of model training or testing. It supports
    flexible scheduling via configurable frequency, and can evaluate multiple guidance
    scales.

    Attributes:
        frequency (Union[int, str]): How often to compute metrics (e.g. '1ep', '1000ba').
        guidance_scales (List[float]): Guidance scale values to use for evaluation.
        seed (int): Random seed for evaluation sample generation.
        num_inference_steps (int): Number of inference steps for sampling.
        max_samples (int): Maximum number of samples to use for metric computation.
        compute_fid (bool): Whether to compute the Frechet Inception Distance (FID).
        compute_cmmd (bool): Whether to compute CLIP-based Maximum Mean Discrepancy metric (CMMD).
        compute_dino_mmd (bool): Whether to compute DINO-based MMD metric.
        clip_model_name (str): Name or path for loading the CLIP model.
        dino_model_name (str): Name or path for loading the DINO model.
        dino_resize_size (int): Input resize size for the DINO model.
        mmd_sigma (float): Sigma value for MMD kernels.
        mmd_scale (float): Scaling value for MMD kernels.
        mmd_chunk_size (int): Chunk size for MMD computation (for efficiency).
        clip_use_pil (bool): Whether to use PIL preprocessing for CLIP feature extraction.
        store_feats_on_cpu (bool): Whether to store extracted features on CPU to save GPU memory.

    Typical usage:
        Add this callback to the list of callbacks for your training or evaluation run.
        The callback will collect real and generated samples and periodically compute the
        relevant quality metrics, logging them through the provided logger.
    """

    def __init__(
        self,
        frequency: Union[int, str] = "1ep",
        guidance_scales: Sequence[float] = (3.5,),
        seed: int = 1138,
        num_inference_steps: int = 28,
        max_samples: int = 10000,
        compute_fid: bool = True,
        compute_cmmd: bool = True,
        compute_dino_mmd: bool = True,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        dino_model_name: str = "dinov2_vitl14_reg",
        dino_resize_size: int = 518,
        mmd_sigma: float = 10.0,
        mmd_scale: float = 1000.0,
        mmd_chunk_size: int = 1024,
        clip_use_pil: bool = True,
        store_feats_on_cpu: bool = False,
    ) -> None:
        self.frequency = _Frequency.from_input(frequency)
        self.guidance_scales = list(guidance_scales)
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.max_samples = max_samples

        self.compute_fid = compute_fid
        self.compute_cmmd = compute_cmmd
        self.compute_dino_mmd = compute_dino_mmd

        self.clip_model_name = clip_model_name
        self.dino_model_name = dino_model_name
        self.dino_resize_size = dino_resize_size

        self.mmd_sigma = float(mmd_sigma)
        self.mmd_scale = float(mmd_scale)
        self.mmd_chunk_size = int(mmd_chunk_size)

        self.clip_use_pil = bool(clip_use_pil)
        self.store_feats_on_cpu = bool(store_feats_on_cpu)

        self.metrics: Optional[Dict[str, Dict[float, Metric]]] = None
        self._metrics_initialized = False

        self.clip_extractor: Optional[CLIPFeatureExtractor] = None
        self.dino_extractor: Optional[DINOFeatureExtractor] = None

        self._run_this_eval = False
        self._samples_processed = 0
        self._per_worker_limit = 0
        self._metric_computation_done = False

    def get_model(self, state: State) -> LatentDiffusion:
        if isinstance(state.model, DistributedDataParallel):
            return state.model.module
        return state.model

    def _should_run_now(self, state: State) -> bool:
        cur_ba = state.timestamp.batch.value
        cur_ep = int(state.timestamp.epoch.value)
        cur_sp = state.timestamp.sample.value

        freq = self.frequency
        if freq.unit == TimeUnit.BATCH:
            return bool(cur_ba > 0 and (cur_ba % freq.value == 0))
        if freq.unit == TimeUnit.EPOCH:
            return bool(cur_ep > 0 and (cur_ep % freq.value == 0))
        if freq.unit == TimeUnit.SAMPLE:
            return bool(cur_sp > 0 and (cur_sp % freq.value == 0))
        return False

    def eval_start(self, state: State, logger: Logger) -> None:
        self._run_this_eval = self._should_run_now(state)
        if not self._run_this_eval:
            return

        self._samples_processed = 0
        self._metric_computation_done = False

        world_size = dist.get_world_size()
        self._per_worker_limit = max(1, self.max_samples // world_size)

        if dist.get_global_rank() == 0:
            _logger.info(
                "Global max_samples: %d, World size: %d, Per-worker limit: %d",
                self.max_samples, world_size, self._per_worker_limit,
            )

        model = self.get_model(state)
        device = model.denoiser_device

        if not self._metrics_initialized:
            self.metrics = {}

            if self.compute_fid:
                self.metrics["FID"] = {
                    scale: FrechetInceptionDistance(normalize=True).to(device) for scale in self.guidance_scales
                }

            if self.compute_cmmd:
                self.clip_extractor = CLIPFeatureExtractor(
                    model_name=self.clip_model_name,
                    use_pil=self.clip_use_pil,
                ).to(device)

                self.metrics["CMMD"] = {
                    scale: RBFMMDMetric(
                        feature_extractor=self.clip_extractor,
                        sigma=self.mmd_sigma,
                        scale=self.mmd_scale,
                        chunk_size=self.mmd_chunk_size,
                        store_on_cpu=self.store_feats_on_cpu,
                    ).to(device)
                    for scale in self.guidance_scales
                }

            if self.compute_dino_mmd:
                self.dino_extractor = DINOFeatureExtractor(
                    model_name=self.dino_model_name,
                    resize_size=self.dino_resize_size,
                    normalize_embeddings=True,
                ).to(device)

                self.metrics["DINO_MMD"] = {
                    scale: RBFMMDMetric(
                        feature_extractor=self.dino_extractor,
                        sigma=self.mmd_sigma,
                        scale=self.mmd_scale,
                        chunk_size=self.mmd_chunk_size,
                        store_on_cpu=self.store_feats_on_cpu,
                    ).to(device)
                    for scale in self.guidance_scales
                }

            self._metrics_initialized = True
        else:
            if self.clip_extractor is not None:
                self.clip_extractor.to(device)
            if self.dino_extractor is not None:
                self.dino_extractor.to(device)

    @torch.inference_mode()  # type: ignore
    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if not self._run_this_eval:
            return
        if self._metric_computation_done:
            return
        assert self.metrics is not None, "Metrics should be initialized in eval_start."

        model = self.get_model(state)
        batch = state.batch

        batch_size = (
            batch[BatchKeys.IMAGE].shape[0] if BatchKeys.IMAGE in batch else batch[BatchKeys.IMAGE_LATENT].shape[0]
        )

        if self._samples_processed >= self._per_worker_limit:
            if dist.get_global_rank() == 0:
                _logger.info(
                    "Rank 0: Reached per-worker limit (%d samples). "
                    "Computing metrics (global total: ~%d)...",
                    self._per_worker_limit, self._per_worker_limit * dist.get_world_size(),
                )
            self._compute_and_log_metrics(state, logger)
            self._metric_computation_done = True
            return

        image_h, image_w = model.get_image_size_from_batch(batch)
        image_size = (image_h, image_w)

        # Real images
        if BatchKeys.IMAGE in batch:
            real_images = batch[BatchKeys.IMAGE]
        else:
            real_images = model.latent_to_image(batch[BatchKeys.IMAGE_LATENT])

        real_images = torch.clamp(real_images * 255.0, 0, 255).to(torch.uint8)
        if real_images.ndim == 4 and real_images.shape[-1] in (1, 3):
            real_images = real_images.permute(0, 3, 1, 2).contiguous()

        for guidance_scale in self.guidance_scales:
            gen_images = model.generate(
                batch=batch,
                image_size=image_size,
                guidance_scale=guidance_scale,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                progress_bar=False,
                denoiser=model.ema_denoiser
                if getattr(model, "ema_denoiser", None) and model.ema_denoiser.is_active
                else model.denoiser,
            )

            gen_images = torch.clamp(gen_images * 255.0, 0, 255).to(torch.uint8)
            if gen_images.ndim == 4 and gen_images.shape[-1] in (1, 3):
                gen_images = gen_images.permute(0, 3, 1, 2).contiguous()

            for metric_type, metrics_dict in self.metrics.items():
                metric = metrics_dict[guidance_scale]

                # Determine where the metric lives (so we don't accidentally copy huge tensors too much)
                metric_device = None
                try:
                    metric_device = next(metric.parameters()).device  # type: ignore
                except StopIteration:
                    metric_device = real_images.device

                real_slice = real_images.to(metric_device, non_blocking=True)
                gen_slice = gen_images.to(metric_device, non_blocking=True)

                metric.update(real_slice, real=True)
                metric.update(gen_slice, real=False)

        self._samples_processed += batch_size

    def _compute_and_log_metrics(self, state: State, logger: Logger) -> None:
        if self.metrics is None:
            return

        for metric_type, metrics_dict in self.metrics.items():
            for scale, metric in metrics_dict.items():
                try:
                    score = metric.compute()
                    score_val = score.detach().float().cpu().item()

                    scale_str = str(scale).replace(".", "p")
                    metric_name = f"metrics/eval/{metric_type}_scale_{scale_str}"

                    if dist.get_global_rank() == 0:
                        _logger.info("Logging %s: %s", metric_name, score_val)

                    logger.log_metrics({metric_name: score_val}, step=state.timestamp.batch.value)

                except RuntimeError as e:
                    _logger.error("Rank %d: Error computing %s for scale %s: %s",
                                  dist.get_global_rank(), metric_type, scale, e)
                except Exception as e:
                    _logger.error("Rank %d: Unexpected error computing %s for scale %s: %s: %s",
                                  dist.get_global_rank(), metric_type, scale, type(e).__name__, e)
                finally:
                    metric.reset()

    def eval_end(self, state: State, logger: Logger) -> None:
        if not self._run_this_eval:
            return
        if self.metrics is None:
            return

        if not self._metric_computation_done:
            if dist.get_global_rank() == 0:
                global_total = self._samples_processed * dist.get_world_size()
                _logger.info(
                    "Computing metrics after processing %d samples per worker "
                    "(global total: ~%d)...",
                    self._samples_processed, global_total,
                )
            self._compute_and_log_metrics(state, logger)

        # Move feature extractors back to CPU to free GPU memory during training
        if self.clip_extractor is not None:
            self.clip_extractor.to("cpu")
        if self.dino_extractor is not None:
            self.dino_extractor.to("cpu")
