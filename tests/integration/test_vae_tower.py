"""Integration tests for VaeTower."""

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from prx.models.vae_tower import VaeTower

IDENTITY_CONFIG_REL_PATH: str = "configs/yamls/vae/identity.yaml"
FLUX_CONFIG_REL_PATH: str = "configs/yamls/vae/flux_dev.yaml"
FLUX2_CONFIG_REL_PATH: str = "configs/yamls/vae/flux2.yaml"
DC_AE_CONFIG_REL_PATH: str = "configs/yamls/vae/dc_ae_sana.yaml"
REPA_E_CONFIG_REL_PATH: str = "configs/yamls/vae/flux_repa_e.yaml"


def _load_cfg(repo_root: Path, rel_path: str) -> dict[str, Any]:
    with open(repo_root / rel_path) as f:
        return yaml.safe_load(f)


def _build_tower(cfg: dict[str, Any]) -> VaeTower:
    torch_dtype: torch.dtype = getattr(torch, cfg["torch_dtype"].replace("torch.", ""))
    return VaeTower(
        model_name=cfg["model_name"],
        model_class=cfg["model_class"],
        default_channels=cfg["default_channels"],
        torch_dtype=torch_dtype,
    )


# ── Identity VAE (no download, no GPU) ──────────────────────────────


@pytest.fixture(scope="module")
def identity_cfg(repo_root: Path) -> dict[str, Any]:
    return _load_cfg(repo_root, IDENTITY_CONFIG_REL_PATH)


@pytest.fixture(scope="module")
def identity_tower(identity_cfg: dict[str, Any]) -> VaeTower:
    return _build_tower(identity_cfg)


@pytest.mark.integration
class TestVaeTowerIdentity:

    def test_latent_channels_positive(self, identity_tower: VaeTower) -> None:
        assert identity_tower.latent_channels > 0

    def test_encode_shapes(self, identity_tower: VaeTower) -> None:
        image: torch.Tensor = torch.rand(2, 3, 64, 64)
        latent: torch.Tensor = identity_tower.encode(image)
        # Identity VAE: output shape == input shape
        assert latent.shape == (2, 3, 64, 64)

    def test_encode_no_nan_inf(self, identity_tower: VaeTower) -> None:
        image: torch.Tensor = torch.rand(1, 3, 32, 32)
        latent: torch.Tensor = identity_tower.encode(image)
        assert not torch.isnan(latent).any(), "NaN detected in identity VAE latent"
        assert not torch.isinf(latent).any(), "Inf detected in identity VAE latent"


# ── FLUX.1-dev VAE (real model, GPU) ────────────────────────────────


@pytest.fixture(scope="module")
def flux_cfg(repo_root: Path) -> dict[str, Any]:
    return _load_cfg(repo_root, FLUX_CONFIG_REL_PATH)


@pytest.fixture(scope="module")
def flux_tower(flux_cfg: dict[str, Any], device: torch.device) -> VaeTower:
    tower: VaeTower = _build_tower(flux_cfg)
    return tower.to(device)


@pytest.mark.integration
@pytest.mark.gpu
class TestVaeTowerFlux:

    def test_latent_channels_positive(self, flux_tower: VaeTower) -> None:
        assert flux_tower.latent_channels > 0

    def test_spatial_compression_ratio(self, flux_tower: VaeTower) -> None:
        assert flux_tower.spatial_compression_ratio >= 1

    def test_encode_shapes(self, flux_tower: VaeTower) -> None:
        h, w = 256, 256
        image: torch.Tensor = torch.rand(1, 3, h, w, device=flux_tower.device)
        latent: torch.Tensor = flux_tower.encode(image)
        ratio: int = flux_tower.spatial_compression_ratio
        expected_h: int = h // ratio
        expected_w: int = w // ratio
        assert latent.shape == (1, flux_tower.latent_channels, expected_h, expected_w)

    def test_encode_no_nan_inf(self, flux_tower: VaeTower) -> None:
        image: torch.Tensor = torch.rand(1, 3, 128, 128, device=flux_tower.device)
        latent: torch.Tensor = flux_tower.encode(image)
        assert not torch.isnan(latent).any(), "NaN detected in FLUX VAE latent"
        assert not torch.isinf(latent).any(), "Inf detected in FLUX VAE latent"


# ── FLUX.2-dev VAE (real model, GPU) ─────────────────────────────────


@pytest.fixture(scope="module")
def flux2_cfg(repo_root: Path) -> dict[str, Any]:
    return _load_cfg(repo_root, FLUX2_CONFIG_REL_PATH)


@pytest.fixture(scope="module")
def flux2_tower(flux2_cfg: dict[str, Any], device: torch.device) -> VaeTower:
    tower: VaeTower = _build_tower(flux2_cfg)
    return tower.to(device)


@pytest.mark.integration
@pytest.mark.gpu
class TestVaeTowerFlux2:

    def test_latent_channels(self, flux2_tower: VaeTower) -> None:
        assert flux2_tower.latent_channels == 32

    def test_spatial_compression_ratio(self, flux2_tower: VaeTower) -> None:
        assert flux2_tower.spatial_compression_ratio >= 1

    def test_encode_shapes(self, flux2_tower: VaeTower) -> None:
        h, w = 256, 256
        image: torch.Tensor = torch.rand(1, 3, h, w, device=flux2_tower.device)
        latent: torch.Tensor = flux2_tower.encode(image)
        ratio: int = flux2_tower.spatial_compression_ratio
        assert latent.shape == (1, flux2_tower.latent_channels, h // ratio, w // ratio)

    def test_encode_no_nan_inf(self, flux2_tower: VaeTower) -> None:
        image: torch.Tensor = torch.rand(1, 3, 128, 128, device=flux2_tower.device)
        latent: torch.Tensor = flux2_tower.encode(image)
        assert not torch.isnan(latent).any(), "NaN detected in FLUX.2 VAE latent"
        assert not torch.isinf(latent).any(), "Inf detected in FLUX.2 VAE latent"


# ── DC-AE Sana VAE (real model, GPU) ─────────────────────────────────


@pytest.fixture(scope="module")
def dc_ae_cfg(repo_root: Path) -> dict[str, Any]:
    return _load_cfg(repo_root, DC_AE_CONFIG_REL_PATH)


@pytest.fixture(scope="module")
def dc_ae_tower(dc_ae_cfg: dict[str, Any], device: torch.device) -> VaeTower:
    tower: VaeTower = _build_tower(dc_ae_cfg)
    return tower.to(device)


@pytest.mark.integration
@pytest.mark.gpu
class TestVaeTowerDcAe:

    def test_latent_channels(self, dc_ae_tower: VaeTower) -> None:
        assert dc_ae_tower.latent_channels == 32

    def test_spatial_compression_ratio(self, dc_ae_tower: VaeTower) -> None:
        assert dc_ae_tower.spatial_compression_ratio == 32

    def test_encode_shapes(self, dc_ae_tower: VaeTower) -> None:
        # DC-AE has 32x compression, so input must be at least 32x32
        h, w = 256, 256
        image: torch.Tensor = torch.rand(1, 3, h, w, device=dc_ae_tower.device)
        latent: torch.Tensor = dc_ae_tower.encode(image)
        ratio: int = dc_ae_tower.spatial_compression_ratio
        assert latent.shape == (1, dc_ae_tower.latent_channels, h // ratio, w // ratio)

    def test_encode_no_nan_inf(self, dc_ae_tower: VaeTower) -> None:
        image: torch.Tensor = torch.rand(1, 3, 256, 256, device=dc_ae_tower.device)
        latent: torch.Tensor = dc_ae_tower.encode(image)
        assert not torch.isnan(latent).any(), "NaN detected in DC-AE latent"
        assert not torch.isinf(latent).any(), "Inf detected in DC-AE latent"


# ── REPA-E FLUX VAE (real model, GPU) ────────────────────────────────


@pytest.fixture(scope="module")
def repa_e_cfg(repo_root: Path) -> dict[str, Any]:
    return _load_cfg(repo_root, REPA_E_CONFIG_REL_PATH)


@pytest.fixture(scope="module")
def repa_e_tower(repa_e_cfg: dict[str, Any], device: torch.device) -> VaeTower:
    tower: VaeTower = _build_tower(repa_e_cfg)
    return tower.to(device)


@pytest.mark.integration
@pytest.mark.gpu
class TestVaeTowerRepaE:

    def test_latent_channels(self, repa_e_tower: VaeTower) -> None:
        assert repa_e_tower.latent_channels == 16

    def test_spatial_compression_ratio(self, repa_e_tower: VaeTower) -> None:
        assert repa_e_tower.spatial_compression_ratio >= 1

    def test_encode_shapes(self, repa_e_tower: VaeTower) -> None:
        h, w = 256, 256
        image: torch.Tensor = torch.rand(1, 3, h, w, device=repa_e_tower.device)
        latent: torch.Tensor = repa_e_tower.encode(image)
        ratio: int = repa_e_tower.spatial_compression_ratio
        assert latent.shape == (1, repa_e_tower.latent_channels, h // ratio, w // ratio)

    def test_encode_no_nan_inf(self, repa_e_tower: VaeTower) -> None:
        image: torch.Tensor = torch.rand(1, 3, 128, 128, device=repa_e_tower.device)
        latent: torch.Tensor = repa_e_tower.encode(image)
        assert not torch.isnan(latent).any(), "NaN detected in REPA-E VAE latent"
        assert not torch.isinf(latent).any(), "Inf detected in REPA-E VAE latent"
