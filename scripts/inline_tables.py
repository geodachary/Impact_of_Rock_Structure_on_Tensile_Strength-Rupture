#!/usr/bin/env python3
"""Fold the generated supplementary tables into the manuscript source.

The journal's submission portal prefers one or two ``.tex`` files, so the
tables live inside ``manscript_revision_001.tex`` rather than in six separate
files pulled in with ``\\input``.

Inlining generated content usually costs you regeneration: once the table is
part of the manuscript, re-running the script that computes it no longer
reaches the paper, and the numbers go stale without anyone noticing. That is
avoided here by giving each table a marked block. The scripts write their
tables to ``manuscript/tables/<name>.txt``; this script replaces the text
between the markers with whatever those files currently hold. Regeneration
therefore still works, and the manuscript stays a single file.

Everything outside the markers is untouched, so prose around the tables is
safe to edit by hand.

    python scripts/inline_tables.py            # fold current tables in
    python scripts/inline_tables.py --check    # report drift, change nothing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "manuscript" / "manscript_revision_001.tex"
TABLE_DIR = REPO / "manuscript" / "tables"

#: In the order they appear in the supplement.
TABLES = [
    "table_S_strength_by_angle",
    "table_S_ati_envelope",
    "table_S_mirror_symmetry",
    "table_S_fig20_GGc",
    "table_S_fig20_bands",
    "table_S_fabric_tractions",
    "table_S_eshelby_bound",
    "table_S_trace_validation",
    "table_S_energy_localization",
]

BEGIN = "% >>> BEGIN GENERATED TABLE: {name} (scripts/inline_tables.py) <<<"
END = "% >>> END GENERATED TABLE: {name} <<<"


def block(name, body):
    return (BEGIN.format(name=name) + "\n"
            + "% Edit the producing script, not this block; it is overwritten.\n"
            + body.rstrip() + "\n"
            + END.format(name=name))


def marker_pattern(name):
    return re.compile(
        re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)),
        re.S)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report which tables would change, write nothing")
    args = ap.parse_args(argv)

    # This step edits the document rather than producing analysis output, so a
    # checkout without one has nothing for it to do. Skipping keeps it safe to
    # leave in the pipeline: the tables are still written to
    # manuscript/tables/*.txt by the scripts that compute them.
    if not MANUSCRIPT.is_file():
        print(f"  no document at {MANUSCRIPT.name}; "
              "tables written to manuscript/tables/ and left there")
        return 0

    text = MANUSCRIPT.read_text(encoding="utf-8")
    changed, missing, added = [], [], []

    for name in TABLES:
        src = TABLE_DIR / f"{name}.txt"
        if not src.exists():
            missing.append(name)
            continue
        body = src.read_text(encoding="utf-8")
        new = block(name, body)

        pattern = marker_pattern(name)
        if pattern.search(text):
            if pattern.search(text).group(0) != new:
                text = pattern.sub(lambda _: new, text, count=1)
                changed.append(name)
            continue

        # First run: replace the \input line that used to pull the file in.
        legacy = re.compile(r"\\input\{" + re.escape(name) + r"\}")
        if legacy.search(text):
            text = legacy.sub(lambda _: new, text, count=1)
            added.append(name)
        else:
            missing.append(f"{name} (no marker and no \\input to replace)")

    for label, names in (("inlined", added), ("updated", changed),
                         ("MISSING", missing)):
        if names:
            print(f"  {label}: {', '.join(names)}")
    if not (added or changed or missing):
        print("  all tables already current")

    if not args.check and (added or changed):
        MANUSCRIPT.write_text(text, encoding="utf-8")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
