import pytest


def pytest_collection_modifyitems(config, items):
    """Skip `gpu`-marked tests when CUDA is unavailable (or torch missing)."""
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu:
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
