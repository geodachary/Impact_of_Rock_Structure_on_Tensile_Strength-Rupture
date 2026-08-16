#!/usr/bin/env python3
"""Run every analysis section once and report which ones work.

Each section is run in isolation so one failure does not hide the next, and
rock-scoped sections are run for both lithologies -- the point of the refactor
is that they behave for either, and only running both proves it.

    python scripts/smoke_sections.py [--only name ...]
"""
from __future__ import annotations

import argparse
import inspect
import io
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

import matplotlib
matplotlib.use("Agg")

from tools import lithology, analysis  # noqa: E402

ROCKS = [lithology.AUGEN_GNEISS, lithology.PSAMMITIC_SCHIST]


def run_one(module, rock=None):
    label = module if rock is None else f"{module}[{rock.key}]"
    mod = getattr(analysis, module)
    fn = getattr(mod, "main", None)
    if fn is None:
        return label, "library", 0.0, ""
    buf = io.StringIO()
    start = time.time()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            fn(rock) if rock is not None else fn()
        return label, "ok", time.time() - start, ""
    except BaseException:
        return label, "FAIL", time.time() - start, traceback.format_exc()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    names = args.only or list(analysis.__all__)
    results = []
    for name in names:
        mod = getattr(analysis, name)
        fn = getattr(mod, "main", None)
        if fn is None:
            results.append(run_one(name))
            continue
        takes_rock = "rock" in inspect.signature(fn).parameters
        targets = ROCKS if takes_rock else [None]
        for rock in targets:
            label, status, secs, tb = run_one(name, rock)
            print(f"  {status:7s} {label:44s} {secs:7.1f}s", flush=True)
            results.append((label, status, secs, tb))

    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} ok, {len(failed)} failed")
    for label, _, _, tb in failed:
        print(f"\n===== {label}\n{tb.strip().splitlines()[-1]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
