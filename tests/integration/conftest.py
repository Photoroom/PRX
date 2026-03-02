"""Shared fixtures for integration tests."""

from pathlib import Path

import pytest
import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    return torch.device("cuda")
