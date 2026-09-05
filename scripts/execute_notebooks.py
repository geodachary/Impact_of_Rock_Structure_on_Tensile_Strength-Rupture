#!/usr/bin/env python3
"""Execute the three notebooks in order and store their outputs.

Order matters. The two lithology notebooks export the field archives that
``Tensile_general_plots.ipynb`` compares, and the general notebook asserts up
front that all fourteen are present, so it runs last.

Execution is in-place: the notebooks are the deliverable, and a published
notebook is expected to show its figures and tables without the reader running
anything. Each notebook is executed in the repository root, which is what makes
the relative paths in ``tools/`` resolve.

    python scripts/execute_notebooks.py [--only NOTEBOOK ...] [--timeout SECONDS]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ORDER = [
    "Tensile_augen_gneiss.ipynb",
    "Tensile_psammitic_schist.ipynb",
    "Tensile_general_plots.ipynb",
]


def strip_embedded_images(nb):
    """Drop rendered images from cell outputs, keep the printed text.

    Every figure a cell draws is also written to ``outputs/figures`` as PDF and
    PNG, so embedding it here stores the same picture twice: 36 MB of base64
    across the three notebooks against 60 kB of printed output. Base64 in JSON
    does not diff, so each run rewrites all of it as a new blob.

    ``build_notebooks.py`` strips images when it carries outputs across a
    rebuild, but that does not help here: executing the notebooks puts them
    straight back. The stripping has to happen where the outputs are produced,
    which is here.
    """
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            data = out.get("data")
            if data is None:
                kept.append(out)
                continue
            trimmed = {k: v for k, v in data.items() if not k.startswith("image/")}
            if not trimmed:
                continue                      # the output was only a picture
            out["data"] = trimmed
            meta = out.get("metadata")
            if isinstance(meta, dict):
                out["metadata"] = {k: v for k, v in meta.items()
                                   if not k.startswith("image/")}
            kept.append(out)
        cell["outputs"] = kept


def execute(name, timeout, kernel):
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = REPO / name
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name=kernel,
        resources={"metadata": {"path": str(REPO)}},
        allow_errors=False)

    start = time.time()
    try:
        client.execute()
        status, detail = "ok", ""
    except CellExecutionError as exc:
        status, detail = "FAIL", str(exc)
    except BaseException as exc:
        # Anything else means the run did not merely raise inside a cell -- most
        # often the kernel died, in which case nbclient's own cleanup asserts on
        # a missing kernel manager and buries the cause. Say which cell was
        # reached, because that is the part worth acting on.
        done = sum(1 for c in nb.cells
                   if c.get("cell_type") == "code" and c.get("outputs"))
        status = "FAIL"
        detail = (f"{type(exc).__name__}: {exc}\n"
                  f"The kernel stopped after {done} code cells produced output. "
                  f"A bare AssertionError from nbclient's cleanup means the "
                  f"kernel process died (killed, out of memory, or a crash in a "
                  f"native library) rather than raising inside a cell.")
    finally:
        # Write whatever ran, so a failure still leaves the successful cells'
        # outputs on disk and the failing cell's traceback visible in place.
        strip_embedded_images(nb)
        nbformat.write(nb, path)

    elapsed = time.time() - start
    n_out = sum(1 for c in nb.cells if c.get("cell_type") == "code" and c.get("outputs"))
    n_code = sum(1 for c in nb.cells if c.get("cell_type") == "code")
    print(f"  {status:4s} {name:34s} {elapsed / 60:5.1f} min   "
          f"{n_out}/{n_code} code cells produced output", flush=True)
    return status, detail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=14400,
                    help="per-cell timeout in seconds (default 4 hours)")
    ap.add_argument("--kernel", default="viscoquake")
    args = ap.parse_args(argv)

    names = args.only or ORDER
    failures = []
    for name in names:
        status, detail = execute(name, args.timeout, args.kernel)
        if status == "FAIL":
            failures.append((name, detail))
            # The later notebooks read what the earlier ones export, so a
            # failure upstream makes everything after it meaningless.
            print(f"  -- stopping: {name} failed, and the rest depend on it")
            break

    if failures:
        for name, detail in failures:
            tail = "\n".join(detail.strip().splitlines()[-25:])
            print(f"\n===== {name}\n{tail}")
        return 1
    print(f"\nall {len(names)} notebooks executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
