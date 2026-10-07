"""Skip integration tests when no Bitdeer AI API key is available."""

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every integration test when `BITDEER_API_KEY` is unset."""
    if os.environ.get("BITDEER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="BITDEER_API_KEY is not set")
    for item in items:
        item.add_marker(skip)
