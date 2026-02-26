"""Root conftest: register custom markers so pytest doesn't warn about them."""


def pytest_configure(config: object) -> None:
    config.addinivalue_line("markers", "unit: fast unit tests (no GPU, no model downloads)")  # type: ignore[attr-defined]
    config.addinivalue_line("markers", "integration: integration tests (load real models, slow)")  # type: ignore[attr-defined]
    config.addinivalue_line("markers", "gpu: tests that require a CUDA GPU")  # type: ignore[attr-defined]
