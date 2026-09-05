"""Re-running a generator must not change the figure unless the data changed.

The PDF backend stamps a CreationDate, so an identical plot produced a
byte-different file on every run. Two consequences: the sync stage recopied all
54 cited figures whether or not anything had moved, and a checksum could not
tell a refreshed figure from a changed one, which is the only cheap way to know
whether a rebuild is needed.

``tools/__init__`` pins SOURCE_DATE_EPOCH, which matplotlib honours for that
stamp, so figure bytes now depend on the data alone.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "outputs" / "figures" / "delta_failure_vs_spacing.pdf"
GEN = REPO / "scripts" / "make_sensitivity_figures.py"


def test_the_epoch_is_pinned_at_package_import():
    out = subprocess.run(
        [sys.executable, "-c", "import tools, os; print(os.environ['SOURCE_DATE_EPOCH'])"],
        cwd=REPO, capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "SOURCE_DATE_EPOCH"})
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().isdigit(), (
        "importing tools no longer pins SOURCE_DATE_EPOCH; figure bytes will "
        "change on every run again")


def test_an_explicit_epoch_still_wins():
    """setdefault, so a dated release build can override it."""
    env = dict(os.environ, SOURCE_DATE_EPOCH="1234567890")
    out = subprocess.run(
        [sys.executable, "-c", "import tools, os; print(os.environ['SOURCE_DATE_EPOCH'])"],
        cwd=REPO, capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "1234567890"


@pytest.mark.slow
@pytest.mark.skipif(not GEN.is_file(), reason="generator absent")
def test_rerunning_a_generator_reproduces_the_same_bytes():
    digest = lambda: hashlib.sha256(FIG.read_bytes()).hexdigest()
    for _ in range(2):
        r = subprocess.run([sys.executable, str(GEN)], cwd=REPO,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    first = digest()
    r = subprocess.run([sys.executable, str(GEN)], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert digest() == first, (
        "the same generator produced different bytes from the same data; "
        "figure output is no longer reproducible")
