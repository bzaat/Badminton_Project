"""Shared test setup.

BFMD_ROOT (environment) overrides the data path of every config. That is wanted for the real-data
regression tests (marked `data`), e.g. on Colab where the data lives in /content/BFMD_data, but it
must not leak into the synthetic tests, which build their own config around a tiny fake dataset.
"""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_bfmd_root(request, monkeypatch):
    if request.node.get_closest_marker("data") is None:
        monkeypatch.delenv("BFMD_ROOT", raising=False)
