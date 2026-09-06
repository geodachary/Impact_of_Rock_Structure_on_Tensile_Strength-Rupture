#!/usr/bin/env python3
"""One-command reproduction of the analysis outputs and publication figures.

Calls the same ``tools`` functions the notebooks use, so nothing is duplicated
here: this script orchestrates, it does not reimplement any science.

    python scripts/reproduce_all.py --mode full
    python scripts/reproduce_all.py --mode smoke      # fast structural check

Exits non-zero on any failed stage.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from tools import output_dirs  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 20260807


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {level:5s} {msg}", flush=True)


# ---------------------------------------------------------------------------
def validate_inputs() -> list[str]:
    """Every raw input the workflow needs must exist before anything runs."""
    from tools import lithology as lith

    missing = []
    for row in lith.pairing_table():
        for key in ("observed_path", "predicted_path"):
            if not Path(row[key]).exists():
                missing.append(str(row[key]))
    for extra in (REPO / "tensile_samples_data.csv",):
        if not extra.exists():
            missing.append(str(extra))
    n_fields = sum(1 for s in range(1, 15) if lith.field_cache_path(s).exists())
    log(f"14 observed + 14 predicted traces checked; {n_fields}/14 field caches present")
    return missing


def record_environment(out: Path) -> None:
    import matplotlib
    import scipy

    info = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        # Name only, not the full path: the absolute path to the interpreter
        # carries the author's home directory into a published record, and the
        # environment name plus the versions below is what a reader needs.
        "environment": Path(sys.executable).parent.parent.name,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": SEED,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / output_dirs.TABLE_DIR / "environment.json").write_text(json.dumps(info, indent=2))
    log(f"environment recorded -> {output_dirs.TABLE_DIR}/environment.json")


def stage_trace_validation(out: Path) -> dict:
    """Observed-vs-predicted orientation validation for all 14 specimens."""
    from tools import traces as tr
    from tools import export as tx

    rows = tr.compare_all(seed=SEED)
    # tables are written into the run's own output root so that two runs can be
    # compared numerically, not only by their figures
    tx.write_pairing_manifest(rows, out=out / output_dirs.TABLE_DIR)
    tx.write_trace_metrics(rows, out=out / output_dirs.TABLE_DIR)
    tx.write_orientation_validation(rows, out=out / output_dirs.TABLE_DIR)
    agg = tr.aggregate_statistics(rows)
    null = tr.null_model_statistics(rows)
    log(f"orientation validation: n={agg['n']}, MAE={agg['mae_deg']:.2f} deg, "
        f"within 5 deg {agg['n_within_5']}/{agg['n']}")
    log(f"loading-parallel null : MAE={null['mae_deg']:.2f} deg "
        f"({'model does NOT beat null' if null['mae_deg'] < agg['mae_deg'] else 'model beats null'})")
    return dict(rows=rows, agg=agg, null=null)


def stage_figures(out_root: Path, smoke: bool = False) -> list[Path]:
    """The four required seven-panel composite figures."""
    import matplotlib
    matplotlib.use("Agg")
    from tools import lithology as lith
    from tools import traces as tr
    from tools import plotting as tp

    formats = ("png",) if smoke else ("pdf", "png")
    written: list[Path] = []
    rows = tr.compare_all(seed=SEED)
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        sub = [r for r in rows if r["lithology_key"] == lit.key]
        stem = out_root / output_dirs.FIGURE_DIR / f"fracture_trace_overlay_{lit.key}_7panel"
        written += tp.seven_panel_trace_overlay(sub, lit, stem, formats=formats)
        log(f"overlay composite written for {lit.display_name} (7 panels)")

    # classification composites reuse the shared script so the science is not duplicated
    from tools import strain_partitioning as sp_mod

    strengths = sp_mod.specimen_strengths()
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        panels = [_classify_panel(s, strengths) for s in lit.sample_ids]
        ok = [p for p in panels if p.get("status") == "computed"]
        if len(ok) != 7:
            log(f"only {len(ok)}/7 classification panels for {lit.display_name}", "WARN")
            continue
        stem = out_root / output_dirs.FIGURE_DIR / f"failure_mechanism_classification_{lit.key}_7panel"
        written += tp.seven_panel_classification(ok, lit, stem, formats=formats)
        log(f"classification composite written for {lit.display_name} (7 panels)")
    return written


def stage_partitioning(out_root: Path, smoke: bool = False) -> list[Path]:
    """Four-mechanism strain-energy partitioning, both lithologies."""
    import matplotlib
    matplotlib.use("Agg")
    from tools import strain_partitioning as sp_mod
    from tools import plotting as tp

    df = sp_mod.partition_all()
    (out_root / output_dirs.TABLE_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / output_dirs.TABLE_DIR / "strain_partition_fourclass.csv", index=False)
    c = sp_mod.lithology_contrast(df)
    n_same = int(c["identical"].sum())
    log(f"strain partitioning: {len(c)} quantities compared, {n_same} identical "
        f"between lithologies "
        f"({'FAIL - rocks not distinguished' if n_same else 'rocks are distinguished'})")
    log(f"  MS  gneiss {c.loc['MS_pct','Augen gneiss']:.2f} % vs schist "
        f"{c.loc['MS_pct','Psammitic schist']:.2f} %")
    log(f"  WT  gneiss {c.loc['WT_pct','Augen gneiss']:.2f} % vs schist "
        f"{c.loc['WT_pct','Psammitic schist']:.2f} %")
    formats = ("png",) if smoke else ("pdf", "png")
    return tp.strain_partition_two_rocks(
        df, out_root / output_dirs.FIGURE_DIR / "strain_partitioning_two_rocks", formats=formats)


def _classify_panel(sample_id, strengths):
    """Four-mechanism classification of one specimen, for the composite figure."""
    import numpy as np
    from tools import lithology as lith
    from tools import failure_classification as fc
    from tools import strain_partitioning as sp_mod

    npz = lith.field_cache_path(sample_id)
    lit = lith.lithology_of_sample(sample_id)
    if not npz.exists():
        return dict(sample=sample_id, status="blocked", reason=f"{npz.name} absent")
    d = np.load(npz, allow_pickle=True)
    p = strengths[int(sample_id)]
    res = fc.classify(
        d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=sp_mod.WEAK_T_RATIO * p["T_m"], c_wp=sp_mod.WEAK_C_RATIO * p["c_m"],
        phi_wp=p["phi_m"], T_m=p["T_m"], c_m=p["c_m"], phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=sp_mod.ACTIVATION_FLOOR,
        threshold=sp_mod.THRESHOLD, eta_mix=sp_mod.ETA_MIX, n_theta=sp_mod.N_THETA)
    return dict(sample=sample_id, angle_deg=lit.angle_for_sample(sample_id),
                lithology=lit.display_name, X=d["X"], Y=d["Y"],
                mask=d["M"].astype(bool), mode_code=res["mode_code"],
                mixed_flag=res["mixed_flag"], status="computed")


def stage_derived_figures(smoke: bool = False) -> list[str]:
    """Figures and tables built from the exported DDM fields.

    These read ``outputs/fields/fields_npz`` rather than recomputing it, so they are
    cheap, but they must run after the notebooks: the stress-glyph panels, the
    fabric-resolved tractions and the inclusion bound all quote numbers that the
    manuscript reproduces verbatim.
    """
    import runpy

    done = []
    for name in ("make_stress_field_figures", "make_stress_profile_figures",
                 "make_failure_statistics_figures", "make_energy_localization_figure",
                 "make_weakening_figure", "make_eshelby_supplement"):
        if smoke:
            done.append(f"{name} (skipped in smoke mode)")
            continue
        # Each generator ends with ``raise SystemExit(main())``, and running it
        # under ``run_name="__main__"`` lets that SystemExit propagate. It did:
        # the last generator in this list terminated reproduce_all itself, with
        # status 0, so stage_checks and the COMPLETED line below have never
        # executed and the script reported success without verifying anything.
        # Catch it here and treat a non-zero status as the failure it is.
        try:
            runpy.run_path(str(REPO / "scripts" / f"{name}.py"),
                           run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            if code:
                raise RuntimeError(f"{name} exited with status {code}") from exc
        done.append(name)
        log(f"{name} completed")
    return done


def stage_checks(res: dict, figures: list[Path]) -> list[str]:
    """Assertions that must hold for the run to be declared good."""
    problems = []
    if res["agg"]["n"] != 14:
        problems.append(f"expected 14 computable specimens, got {res['agg']['n']}")
    if len(figures) < 4:
        problems.append(f"expected at least 4 figure files, got {len(figures)}")
    seen = {r["sample"] for r in res["rows"]}
    if seen != set(range(1, 15)):
        problems.append(f"specimen coverage wrong: {sorted(seen)}")
    return problems


# ---------------------------------------------------------------------------
def stage_sync_document_figures() -> int:
    """Copy every generated figure the document cites into the document folder.

    Most generators already write to both trees, but several do not, and the
    ones that do not are the ones that go stale: 28 of the 44 generated
    graphics the manuscript cites were up to 48 hours behind ``outputs/``
    after a full re-run, so the compiled PDF showed pre-run figures while
    every number in the text had been updated. Nothing warned, because LaTeX
    resolves the filename and does not care how old it is.

    Only files that exist in ``outputs/figures`` are touched. Photographs and
    hand-drawn schematics live in the document folder alone and are inputs;
    they are left untouched because there is nothing to copy over them.

    The digest comparison below is meaningful because ``tools/__init__`` pins
    SOURCE_DATE_EPOCH: matplotlib would otherwise stamp each PDF with the time
    of the run, so every figure would differ on every rebuild and be recopied
    whether or not the plot had changed. A non-zero copy count now means the
    data moved.
    """
    import hashlib
    import re
    import shutil

    tex = output_dirs.document_source()
    if tex is None:
        log("no document present; nothing to sync")
        return 0

    cited = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}",
                           tex.read_text(encoding="utf-8")))
    src_dir = REPO / output_dirs.FIGURE_DIR
    dst_dir = tex.parent
    digest = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()

    copied = 0
    for name in sorted(cited):
        stem = name if "." in name else name + ".pdf"
        src, dst = src_dir / stem, dst_dir / stem
        if not src.exists():
            continue                      # an input, not a generated figure
        if dst.exists() and digest(src) == digest(dst):
            continue
        shutil.copy2(src, dst)
        copied += 1
    log(f"document figures synchronised: {copied} copied, "
        f"{len(cited)} cited by the manuscript")
    return copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("full", "smoke"), default="full",
                    help="full: all formats and stages; smoke: fast structural check")
    ap.add_argument("--output-root", type=Path, default=REPO,
                    help="where generated tables and figures are written")
    args = ap.parse_args(argv)

    t0 = time.time()
    out = Path(args.output_root)
    (out / output_dirs.FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    (out / output_dirs.TABLE_DIR).mkdir(parents=True, exist_ok=True)
    log(f"repository root : {REPO}")
    log(f"output root     : {out}")
    log(f"mode            : {args.mode}")

    missing = validate_inputs()
    if missing:
        for m in missing[:10]:
            log(f"missing required input: {m}", "ERROR")
        return 2
    record_environment(out)

    res = stage_trace_validation(out)
    figures = stage_figures(out, smoke=(args.mode == "smoke"))
    figures += stage_partitioning(out, smoke=(args.mode == "smoke"))
    stage_derived_figures(smoke=(args.mode == "smoke"))
    stage_sync_document_figures()
    problems = stage_checks(res, figures)

    log("-" * 62)
    if problems:
        for p in problems:
            log(p, "FAIL")
        log(f"FAILED after {time.time() - t0:.1f} s", "ERROR")
        return 1
    log(f"{len(figures)} figure files written; "
        f"orientation MAE {res['agg']['mae_deg']:.2f} deg over {res['agg']['n']} specimens")
    log(f"COMPLETED in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
