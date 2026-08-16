"""Quantitative tests of the G/Gc profile claims.

Three claims have been read off the shape of the G/Gc curves: that the profiles
increase monotonically at low foliation angles, that oscillation is prominent
only at intermediate angles, and that Mode-I therefore dominates at low angles.
These are claims about curve shape, so they are settled by measuring the curves
rather than by inspecting them.

Four measurements, each aimed at one part of the claim:

``monotonic``
    Is the profile strictly increasing? This is the claim as stated. A profile
    that reverses even once is not monotonic.

``cv``
    Coefficient of variation of G/Gc along the path — an oscillation amplitude
    that does not depend on path length or on the absolute level of G. If
    oscillation were confined to intermediate angles, the low-angle band would
    show a markedly smaller CV. Sample standard deviation (``ddof=1``).

``n_below_Gc``
    Steps where G < Gc. Arrest-and-reinitiation requires the driving force to
    fall below toughness at least once.

``shear_fraction``
    Steps whose crack-tip field is shear-dominated (``kIIratio`` > 0.5). Mode-I
    dominance requires this to be small.

Note on Gc: every run in ``outputs/fields/run_summary.csv`` uses Gc0 = 1.0, so
G/Gc here is a *relative* measure along the path, not an absolute margin against
a measured toughness. Any statement about approaching or exceeding real
toughness needs a measured Gc, which this dataset does not contain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import lithology as lith
from . import output_dirs

#: Angle bands used to test "low angles differ from intermediate angles".
BANDS = {"low (0-30 deg)": (0, 30), "intermediate (45-60 deg)": (45, 60),
         "high (75-90 deg)": (75, 90)}

#: A step is shear-dominated when |K_II| / |K_I| exceeds this.
SHEAR_RATIO_THRESHOLD = 0.5

TRACE_GLOB = output_dirs.FIELD_DIR + "/smoke_trace_sid{sid}.csv"
RUN_SUMMARY = output_dirs.FIELD_DIR + "/run_summary.csv"


def band_of(angle_deg) -> str:
    for name, (lo, hi) in BANDS.items():
        if lo <= angle_deg <= hi:
            return name
    raise ValueError(f"angle {angle_deg} falls outside the defined bands")


def load_profile(sample_id, root=None) -> pd.DataFrame:
    """Step-by-step crack-path record for one specimen."""
    root = Path(root or lith.REPO_ROOT)
    path = root / TRACE_GLOB.format(sid=int(sample_id))
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def profile_metrics(sample_id, root=None) -> dict:
    """The four G/Gc shape measurements for one specimen."""
    d = load_profile(sample_id, root)
    lit = lith.lithology_of_sample(sample_id)
    raw = (d["G"] / d["Gc"]).to_numpy(float)
    finite = np.isfinite(raw)
    ratio = raw[finite]
    # Keep the step kind aligned with the ratios by masking it identically.
    # Older traces predate the column and were energy-selected throughout.
    step_mode = (d["mode"].astype(str).to_numpy() if "mode" in d.columns
                 else np.full(len(d), "ENERGY"))[finite]
    if ratio.size < 2:
        return dict(sample=int(sample_id), lithology=lit.display_name,
                    status="blocked", reason="fewer than two finite G/Gc steps")

    diffs = np.diff(ratio)
    angle = lit.angle_for_sample(sample_id)
    shear = d["kIIratio"].to_numpy(float)
    shear = shear[np.isfinite(shear)]

    # Split by how each step was chosen, because a sub-critical step means
    # different things in the two cases.
    #
    # An ENERGY step is selected by maximising G - Gc over admissible
    # directions, and candidates with G <= Gc are discarded outright. A PHYS
    # step is taken when *no* direction clears that bar: the stepper then
    # advances along the preferred-orientation field regardless.
    #
    # So a count of sub-Gc steps over the whole path is not a measurement. It
    # is near-zero if only ENERGY steps are recorded and necessarily large once
    # PHYS steps are, in both cases because of how steps are chosen rather than
    # because of what the crack does. Arrest has to be judged on the
    # energy-driven steps, where falling below Gc would actually be
    # informative -- see :func:`verdict`.
    energy = ratio[step_mode == "ENERGY"]

    return dict(
        sample=int(sample_id), lithology=lit.display_name, angle_deg=angle,
        band=band_of(angle), n_steps=int(ratio.size),
        n_steps_energy=int(energy.size),
        n_steps_phys=int(ratio.size - energy.size),
        phys_fraction=float(1.0 - energy.size / ratio.size) if ratio.size else np.nan,
        monotonic=bool(np.all(diffs > 0)),
        n_reversals=int((diffs < 0).sum()),
        reversal_fraction=float((diffs < 0).mean()),
        cv=float(ratio.std(ddof=1) / ratio.mean()),
        min_G_over_Gc=float(ratio.min()), max_G_over_Gc=float(ratio.max()),
        n_below_Gc=int((ratio < 1.0).sum()),
        min_G_over_Gc_energy=float(energy.min()) if energy.size else np.nan,
        n_below_Gc_energy=int((energy < 1.0).sum()),
        shear_fraction=(float((shear > SHEAR_RATIO_THRESHOLD).mean())
                        if shear.size else np.nan),
        status="computed", reason="")


def all_profiles(root=None) -> pd.DataFrame:
    """All 14 specimens, gneiss then schist, in specimen order."""
    return pd.DataFrame([profile_metrics(s, root) for s in range(1, 15)])


def band_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean oscillation amplitude per angle band.

    This is the comparison the "oscillatory only at intermediate angles" claim
    rests on: if it held, the low band would show a markedly smaller CV.
    """
    g = df[df.status == "computed"]
    out = g.groupby("band").agg(
        n_profiles=("cv", "size"), mean_CV=("cv", "mean"),
        mean_reversal_fraction=("reversal_fraction", "mean"),
        n_monotonic=("monotonic", "sum"),
        mean_shear_fraction=("shear_fraction", "mean")).reset_index()
    order = list(BANDS)
    return out.set_index("band").loc[[b for b in order if b in out.band.values]].reset_index()


def verdict(df: pd.DataFrame) -> dict:
    """Does the measured evidence support the monotonicity and Mode-I reading?"""
    g = df[df.status == "computed"]
    bands = band_summary(df).set_index("band")["mean_CV"]
    lo = bands.get("low (0-30 deg)", np.nan)
    mid = bands.get("intermediate (45-60 deg)", np.nan)
    return dict(
        n_profiles=int(len(g)),
        n_monotonic=int(g.monotonic.sum()),
        monotonic_claim_supported=bool(g.monotonic.any()),
        low_band_CV=float(lo), intermediate_band_CV=float(mid),
        CV_ratio_low_over_intermediate=float(lo / mid) if mid else np.nan,
        oscillation_confined_to_intermediate=bool(lo < 0.5 * mid),
        min_G_over_Gc=float(g.min_G_over_Gc.min()),
        min_G_over_Gc_energy=float(g.min_G_over_Gc_energy.min()),
        total_steps_below_Gc=int(g.n_below_Gc.sum()),
        total_steps_below_Gc_energy=int(g.n_below_Gc_energy.sum()),
        phys_fraction_mean=float(g.phys_fraction.mean()),
        # Judged on the energy-driven steps only. A sub-Gc step chosen *because*
        # nothing cleared Gc is a property of the fallback, not evidence that
        # the crack arrested; counting those would make the reading true for
        # every specimen by construction. On the energy-driven steps the
        # question is at least well posed, and the answer there is no.
        arrest_reinitiation_supported=bool(g.n_below_Gc_energy.sum() > 0),
        mean_shear_fraction_low_band=float(
            g[g.band == "low (0-30 deg)"].shear_fraction.mean()),
    )
