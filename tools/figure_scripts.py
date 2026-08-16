"""Run the figure-building scripts in ``scripts/`` and show what they produced.

Both lithology notebooks and the general notebook ended with the same small
helper pasted inline. It is here so the notebooks keep their promise of holding
no definitions, and so a change to how scripts are invoked is made once.

The scripts write **both** lithologies in one pass, so running them from either
lithology notebook also refreshes the other's panels. That is deliberate: it is
what stops one rock's figures being regenerated against a newer model than the
other's.

Because the scripts run in a subprocess, their figures land on disk and nothing
appears in the notebook. A reader then has to go and find the files to see what
a cell did. :func:`run` therefore notices which figures a script wrote and
displays them inline, so the notebook shows the result next to the call that
produced it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import output_dirs

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Everywhere a generator may leave a figure. The k_max sweep still writes to
#: the repository root, hence the trailing entry.
FIGURE_DIRS = (output_dirs.FIGURE_DIR, output_dirs.DOC_DIR, ".")

SUFFIXES = (".pdf", ".png")

#: Rendering resolution for inline display. High enough to read an axis label,
#: low enough that a notebook holding dozens of panels stays a sane size.
DISPLAY_DPI = 110


def _snapshot(root=None):
    """Modification times of every figure currently on disk."""
    root = Path(root) if root is not None else REPO_ROOT
    out = {}
    for folder in FIGURE_DIRS:
        d = root / folder
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in SUFFIXES and p.is_file():
                out[p] = p.stat().st_mtime
    return out


def show(paths, dpi=DISPLAY_DPI):
    """Display figures inline, rendering PDF pages to images as needed.

    Silently does nothing outside a notebook, so the same helper is safe to
    call from a plain script.
    """
    try:
        from IPython import get_ipython
        from IPython.display import display, Image, Markdown
    except ImportError:                                   # pragma: no cover
        return
    # IPython is importable in a plain interpreter too, where display() only
    # prints an object repr. Showing figures is a notebook behaviour, so check
    # for a live kernel rather than for the import.
    if get_ipython() is None:
        return

    for path in paths:
        path = Path(path)
        try:
            if path.suffix.lower() == ".png":
                display(Markdown(f"**{path.name}**"))
                display(Image(filename=str(path)))
                continue
            import fitz                                   # PyMuPDF
            with fitz.open(path) as doc:
                for n, page in enumerate(doc):
                    label = path.name if len(doc) == 1 else f"{path.name} (page {n + 1})"
                    display(Markdown(f"**{label}**"))
                    display(Image(data=page.get_pixmap(dpi=dpi).tobytes("png")))
        except Exception as exc:                          # pragma: no cover
            # A figure that cannot be rendered must not stop the notebook; say
            # so and carry on, because the file itself is still on disk.
            print(f"  [display] {path.name}: {type(exc).__name__}: {exc}")


def run(script: str, root: Path | None = None, display_figures: bool = True) -> list:
    """Run ``scripts/<script>.py``, echo its output, and show what it wrote.

    Returns the figures the run created or modified. Only the tail of stderr is
    shown on failure: these scripts are chatty and the useful part is at the end.
    """
    root = Path(root) if root is not None else REPO_ROOT
    before = _snapshot(root)
    result = subprocess.run(
        [sys.executable, f"scripts/{script}.py"],
        capture_output=True, text=True, cwd=root)
    print(result.stdout.strip() or result.stderr.strip()[-800:])
    if result.returncode:
        raise RuntimeError(f"scripts/{script}.py failed ({result.returncode})")

    after = _snapshot(root)
    written = sorted(p for p, t in after.items()
                     if p not in before or t > before[p] + 1e-9)
    # One figure often lands in two folders; show each distinct name once.
    seen, unique = set(), []
    for p in written:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    if display_figures and unique:
        show(unique)
    return unique


def run_all(scripts, root: Path | None = None, display_figures: bool = True) -> list:
    """Run several scripts in order, returning everything they wrote."""
    written = []
    for script in scripts:
        written.extend(run(script, root=root, display_figures=display_figures))
    return written


def report_expected(filenames, folder: str = "manuscript",
                    root: Path | None = None) -> list:
    """Print whether each expected figure exists; return the missing ones.

    Reporting rather than asserting is intentional: a missing figure at this
    point usually means the script that writes it has not been run yet, and the
    reader is better served by the full list than by the first failure.
    """
    root = Path(root) if root is not None else REPO_ROOT
    missing = []
    for name in filenames:
        path = root / folder / name
        ok = path.exists()
        print(f"  {'ok     ' if ok else 'MISSING'} {name}")
        if not ok:
            missing.append(name)
    return missing
