#!/usr/bin/env python3
"""One-off: move the scattered output folders into ``outputs/``.

Before this, the repository root carried eight directories of generated
content, each named after the section that wrote it rather than after what it
held. ``stress_tensors/`` held a single spreadsheet of strength metrics,
``Output_Figures/`` held three CSVs and two solver caches, and figures were
spread across five of them. ``tools/output_dirs.py`` now names three
directories, split by kind, and this moves the existing files to match.

Each file is routed by what it is, not by which folder it sat in, because the
folder is exactly the thing that had stopped being meaningful. ``git mv`` is
used where the file is tracked, so history follows it.

    python scripts/migrate_output_tree.py --dry-run
    python scripts/migrate_output_tree.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import output_dirs  # noqa: E402

FIG, TAB, FLD = (output_dirs.FIGURE_DIR, output_dirs.TABLE_DIR,
                 output_dirs.FIELD_DIR)

#: source folder -> (destination for most of it, per-suffix exceptions)
#:
#: The destination is per folder rather than per extension because extension
#: does not determine kind here: a CSV can be a result a reader opens or solver
#: state a later step reads back, and which one it is depends on what wrote it.
SOURCES = {
    "Output_Figures":            (FIG, {".csv": TAB}),
    "figures":                   (FIG, {}),
    "foliation_deviation_output": (FIG, {}),
    "stress_tensors":            (TAB, {}),
    "strain_partitioning":       (TAB, {}),
    "results":                   (TAB, {}),
    "revision_outputs":          (TAB, {}),
    # These two wrote figures and solver state side by side. The figures join
    # the figures; everything else is state a later step reads back.
    "physics_force":             (FLD, {".pdf": FIG}),
    "ddm_fields":                (FLD, {".pdf": FIG}),
}

#: Whole directories that move as a unit, keeping their name.
SUBTREES = {
    "ddm_fields/fields_npz": FLD + "/fields_npz",
    "Output_Figures/_cache_uv_profiles_v2": FLD + "/_cache_uv_profiles_v2",
    "Output_Figures/_cache_direction_circles_uv_v2":
        FLD + "/_cache_direction_circles_uv_v2",
    "results/executed_notebooks": TAB + "/executed_notebooks",
}

#: Directories whose *contents* are folded into the destination, because the
#: directory itself is the thing being retired.
FOLD = {
    "revision_outputs/figures": FIG,
}

#: Solver caches, matched by glob because each analysis names its own and the
#: set grows. They are ignored by git, so they move as plain directories.
CACHE_GLOBS = ["Output_Figures/_cache_*"]

#: Untracked staging copies whose contents already exist elsewhere under the
#: same names. Deleted rather than moved, so the move does not create a second
#: copy of a figure that is already accounted for.
DISCARD = ["results/_notebook_figures"]

#: Formats no longer produced. The seven-panel composites used to be saved as
#: PDF, SVG and PNG; the SVGs were never placed in anything and only tripled
#: the count of files under a figure's name, so they are dropped rather than
#: carried into the new tree.
DROP_SUFFIXES = {".svg"}


def tracked(path: Path) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                       cwd=REPO, capture_output=True)
    return r.returncode == 0


def move(src: Path, dest_dir: Path, dry: bool) -> str:
    dest = dest_dir / src.name
    rel_s, rel_d = src.relative_to(REPO), dest.relative_to(REPO)
    if dest.exists() and dest.is_dir() and src.is_dir():
        return f"  skip (dir exists)  {rel_s}"
    if dest.exists():
        # The old layout let two producers write the same filename into
        # different folders. Whichever was written last is the current one;
        # the other is a leftover from a superseded producer.
        if src.stat().st_mtime <= dest.stat().st_mtime:
            if not dry:
                if tracked(src):
                    subprocess.run(["git", "rm", "-q", "-f", str(rel_s)],
                                   cwd=REPO, check=True)
                else:
                    src.unlink()
            return f"  older dup, dropped  {rel_s}"
        if not dry:
            dest.unlink()
            dest_dir.mkdir(parents=True, exist_ok=True)
            if tracked(src):
                subprocess.run(["git", "mv", "-f", str(rel_s), str(rel_d)],
                               cwd=REPO, check=True)
            else:
                shutil.move(str(src), str(dest))
        return f"  newer dup, replaces  {rel_s}"
    if dry:
        return f"  {rel_s}  ->  {rel_d}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file() and tracked(src):
        subprocess.run(["git", "mv", str(rel_s), str(rel_d)], cwd=REPO, check=True)
    else:
        shutil.move(str(src), str(dest))
    return f"  {rel_s}  ->  {rel_d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    for d in output_dirs.ALL:
        if not args.dry_run:
            (REPO / d).mkdir(parents=True, exist_ok=True)

    for rel in DISCARD:
        d = REPO / rel
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  discard  {rel}/  ({n} duplicate files)")
            if not args.dry_run:
                shutil.rmtree(d)

    moved = dropped = 0
    for pattern in CACHE_GLOBS:
        parent, _, glob = pattern.rpartition("/")
        for src in sorted((REPO / parent).glob(glob)):
            if src.is_dir():
                print(move(src, REPO / FLD, args.dry_run))
                moved += 1

    for rel, dest_rel in SUBTREES.items():
        src = REPO / rel
        if not src.is_dir():
            continue
        print(move(src, (REPO / dest_rel).parent, args.dry_run))
        moved += 1

    for rel, dest_rel in FOLD.items():
        src = REPO / rel
        if not src.is_dir():
            continue
        for f in sorted(src.rglob("*")):
            if f.is_file() and f.name != ".DS_Store":
                print(move(f, REPO / dest_rel, args.dry_run))
                moved += 1
        if not args.dry_run:
            shutil.rmtree(src, ignore_errors=True)

    for folder, (default, overrides) in SOURCES.items():
        src_dir = REPO / folder
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.iterdir()):
            if not p.is_file() or p.name == ".DS_Store":
                continue
            if p.suffix.lower() in DROP_SUFFIXES:
                print(f"  drop     {p.relative_to(REPO)}  (format retired)")
                if not args.dry_run:
                    if tracked(p):
                        # -f because these carry local modifications from the
                        # last run; the file is being retired either way.
                        subprocess.run(["git", "rm", "-q", "-f",
                                        str(p.relative_to(REPO))],
                                       cwd=REPO, check=True)
                    else:
                        p.unlink()
                dropped += 1
                continue
            dest = overrides.get(p.suffix.lower(), default)
            print(move(p, REPO / dest, args.dry_run))
            moved += 1

    verb = " would move" if args.dry_run else " moved"
    print(f"\n{moved} files{verb}, {dropped} retired-format files dropped")

    if not args.dry_run:
        for folder in SOURCES:
            d = REPO / folder
            if d.is_dir():
                for junk in d.rglob(".DS_Store"):
                    junk.unlink(missing_ok=True)
                if not any(d.iterdir()):
                    d.rmdir()
                    print(f"  removed empty {folder}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
