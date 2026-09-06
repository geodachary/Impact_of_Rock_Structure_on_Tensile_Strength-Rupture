"""Section 3.7 values that reach the paper as prose, not through a table.

Two were wrong and nothing could see it. The characteristic roots were
described as sharing an imaginary part and differing in the sign of the real
part at +/-0.46 and +/-0.69; the real part is not a per-lithology constant and
vanishes in specimen 8, where both roots are purely imaginary. And 96.9% of
tensile-governed gneiss points were said to resolve to WT, a figure from the
uncorrected cohesion, when matrix tensile was empty everywhere; it is 64%.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tools import export, failure_classification as fc, lithology as lith
from tools import strain_partitioning as sp
from tools.lekhnitskii_root import lekh_roots_p

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")

_have_fields = all(lith.field_cache_path(s).exists() for s in range(1, 15))
needs_fields = pytest.mark.skipif(not _have_fields, reason="fields not exported yet")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


@needs_fields
def test_the_roots_stay_separated_by_the_quoted_margin(tex):
    """The claim that makes H real and positive definite, over all fourteen."""
    # The elastic constants are properties of the lithology, so the separation
    # is one value per rock rather than a range over fourteen specimens. It was
    # a range only while the constants were re-read per specimen, which applied
    # the fabric orientation twice; see test_e2_provenance.
    said = re.search(r"\$0\.66\$ in the gneiss and\s*\n?\$([\d.]+)\$ in the schist", tex)
    assert said, "the root-separation sentence is no longer in its expected form"

    seps = {}
    for sid in range(1, 15):
        d = np.load(lith.field_cache_path(sid), allow_pickle=True)
        p1, p2 = lekh_roots_p(float(d["E1_MPa"]) * 1e6, float(d["E2_MPa"]) * 1e6,
                              float(d["nu12"]), float(d["G12_MPa"]) * 1e6)
        assert p1.imag > 0 and p2.imag > 0, f"specimen {sid}: root left the upper half plane"
        seps.setdefault(lith.lithology_of_sample(sid).display_name, []).append(abs(p1 - p2))

    for rock, vals in seps.items():
        assert max(vals) - min(vals) < 1e-6, (
            f"{rock}: the root separation still varies between specimens, "
            f"{min(vals):.4f} to {max(vals):.4f}; the constants should be "
            "lithology properties")
    assert seps["Augen gneiss"][0] == pytest.approx(0.66, abs=0.005)
    assert seps["Psammitic schist"][0] == pytest.approx(float(said.group(1)), abs=0.005)


@needs_fields
def test_a_purely_imaginary_pair_still_occurs():
    """Guard the guard: the case that falsified the old wording must persist.

    If every specimen regained a non-zero real part the corrected sentence
    would still be true, but the reason for correcting it would have vanished
    and a future revision could reasonably reinstate the old claim.
    """
    d = np.load(lith.field_cache_path(8), allow_pickle=True)
    p1, p2 = lekh_roots_p(float(d["E1_MPa"]) * 1e6, float(d["E2_MPa"]) * 1e6,
                          float(d["nu12"]), float(d["G12_MPa"]) * 1e6)
    assert abs(p1.real) < 1e-9 and abs(p2.real) < 1e-9, (
        f"specimen 8 no longer has purely imaginary roots: {p1}, {p2}")
    assert abs(p1 - p2) > 0.1, "the pair collapsed onto a repeated root"


@needs_fields
def test_tensile_failure_at_high_angle_is_entirely_weak_plane():
    """The locus claim, over the analysis interior the rest of the paper uses."""
    st = sp.specimen_strengths()
    counts = {}
    for sid in range(1, 15):
        p = export.classification_panel(sid, st)
        code = p["mode_code"][p["core_mask"]]
        counts[sid] = (int((code == fc.CLASS_CODES["WT"]).sum()),
                       int((code == fc.CLASS_CODES["MT"]).sum()))

    for sid in (6, 7):                       # gneiss at 75 and 90 degrees
        wt, mt = counts[sid]
        assert wt > 0 and mt == 0, (
            f"specimen {sid}: {wt} WT and {mt} MT points. Section 3.7 states "
            "the tensile-governed points at high angle are entirely WT.")

    # Matrix tensile is empty under the measured anisotropy ratios. It held
    # 0.120 of the gneiss interior at 0 degrees while the adopted ratios were
    # in force; Sections 3.7 and 4.6 now report the class as absent, so any
    # reappearance means the classification has moved again.
    gneiss_mt = {s: counts[s][1] for s in range(1, 8) if counts[s][1] > 0}
    assert not gneiss_mt, (
        f"matrix-tensile points now appear in gneiss specimens {sorted(gneiss_mt)}; "
        "Sections 3.7 and 4.6 state the class is empty.")
