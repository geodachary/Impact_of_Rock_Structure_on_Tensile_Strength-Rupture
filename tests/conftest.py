"""Shared pytest fixtures and path setup for the test suite."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def lithologies():
    from tools import lithology
    return lithology


@pytest.fixture(scope="session")
def pairing():
    """The authoritative 14-specimen observed/predicted pairing."""
    from tools.lithology import pairing_table
    return pairing_table()
