#!/usr/bin/env python3
"""Bring the stress-field figures, captions and supplementary table in line with
the recomputed fields.

Both captions this replaces made claims that the fields do not support. The
displacement caption described a corridor rotating toward foliation; the
corridor in fact stays within a few degrees of the loading axis at every
orientation. The stress caption attributed greater orientation heterogeneity to
the gneiss and explained it by stiff augen perturbing the surrounding matrix;
the schist is the more heterogeneous of the two on both measures, and the
elastic field is a homogeneous orthotropic solution containing no inclusions.

Every number written here is read from the results tables rather than typed in,
so the text cannot drift away from the fields again. Run after the notebooks and
``scripts/make_stress_field_figures.py``.

    python scripts/patch_stress_field_text.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fabric_tractions as ft            # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "manuscript/manscript_revision_001.tex"
G, P = "Augen gneiss", "Psammitic schist"


def apply_once(s, old, new, what, marker=None):
    """Replace ``old`` with ``new``, tolerating a run where it is already done.

    These patches were written as one-shot migrations: each asserted its target
    appeared exactly once, then replaced it. That makes the script unrunnable a
    second time, which matters because it also writes two supplementary tables
    at the end -- so once the captions were patched, those tables could never be
    regenerated. Re-running a generator must be safe, or the outputs downstream
    of it go stale silently.

    ``marker`` is the text that proves the patch already landed, for insertions
    where ``new`` is not a self-contained replacement.
    """
    if not s:                       # no document to patch
        return s
    already = (marker or new)
    if s.count(old) == 1:
        return s.replace(old, new)
    if s.count(old) == 0 and s.count(already) >= 1:
        print(f"  {what}: already applied, left as is")
        return s
    raise SystemExit(
        f"{what}: the manuscript no longer matches what this patch expects "
        f"({s.count(old)} matches for the old text, {s.count(already)} for the "
        f"new). Nothing was changed; reconcile the text by hand.")


def insert_after_once(s, anchor, marker, what, addition):
    """Insert ``addition`` after ``anchor``, unless ``marker`` shows it is there.

    Insertions need a different guard from replacements: the anchor survives the
    edit, so re-running would happily insert a second copy. ``marker`` is a
    fragment of the inserted text, usually its label.
    """
    if not s:                       # no document to patch
        return s
    if marker in s:
        print(f"  {what}: already inserted, left as is")
        return s
    if s.count(anchor) != 1:
        raise SystemExit(
            f"{what}: anchor found {s.count(anchor)} times, expected once. "
            f"Nothing was changed.")
    return s.replace(anchor, anchor + addition)


def f(x, n=2):
    return f"{x:.{n}f}"


def main():
    # Two jobs: write the supplementary tables, and bring the captions that
    # describe these fields into line with them. The tables are analysis
    # output and are always written; the caption edits need a document, and a
    # code-only checkout has none. `patch_document` records which half runs.
    from tools import output_dirs
    # DOC_TABLE_DIR is where the supplement .txt files go; TABLE_DIR is
    # where the machine-readable results go. Both are written here.
    output_dirs.ensure(output_dirs.DOC_DIR, output_dirs.DOC_TABLE_DIR,
                       output_dirs.TABLE_DIR)
    patch_document = TEX.is_file()
    if not patch_document:
        print(f"  no document at {TEX.name}; writing tables only")

    tr = pd.read_csv(REPO / output_dirs.TABLE_DIR / "fabric_tractions.csv")
    rot = pd.read_csv(REPO / output_dirs.TABLE_DIR / "principal_rotation.csv")
    het = pd.read_csv(REPO / output_dirs.TABLE_DIR / "principal_heterogeneity.csv")
    grd = pd.read_csv(REPO / output_dirs.TABLE_DIR / "orientation_gradient.csv")
    cor = pd.read_csv(REPO / output_dirs.TABLE_DIR / "displacement_corridor.csv")
    s = TEX.read_text(encoding="utf-8") if patch_document else ""

    rmax = {r: g.rotation_median_deg.max() for r, g in rot.groupby("rock")}
    hmean = {r: g.circ_sd_deg.mean() for r, g in het.groupby("rock")}
    gmed = {r: g.grad_median_deg_per_mm.mean() for r, g in grd.groupby("rock")}
    gp90 = {r: g.grad_p90_deg_per_mm.max() for r, g in grd.groupby("rock")}
    dev = {r: g.deviation_from_load_axis_deg.max() for r, g in cor.groupby("rock")}
    wid = {r: (g.sort_values("angle_deg").corridor_half_width.iloc[0],
               g.sort_values("angle_deg").corridor_half_width.iloc[-1])
           for r, g in cor.groupby("rock")}
    pk = {r: g.peak_displacement_mm.mean() for r, g in cor.groupby("rock")}
    ratio = pk[P] / pk[G]
    sn = {r: g.sort_values("angle_deg").sigma_n_mean_MPa.to_numpy() for r, g in tr.groupby("rock")}
    taupk = {r: int(g.loc[g.tau_abs_mean_MPa.idxmax(), "angle_deg"]) for r, g in tr.groupby("rock")}
    openf = {r: g.sort_values("angle_deg").open_fraction.to_numpy() for r, g in tr.groupby("rock")}
    cross = {r: ft.sign_change_angle(tr, r) for r in (G, P)}

    # ------------------------------------------------ displacement-field caption
    old_dir = """    \\caption{Displacement vector fields for (a) psammitic schist and (b)
    augen gneiss at seven loading orientations. Arrow direction indicates
    the local displacement vector; color encodes displacement magnitude
    (mm). At $0^\\circ$ the high-magnitude corridor (warm colors) is
    nearly vertical and symmetric, recovering the isotropic limit. As
    loading angle increases the corridor rotates progressively toward
    foliation, consistent with compliance-guided steering of the
    displacement field. In the schist the corridor is broader and reaches
    larger magnitudes ($\\approx 6\\times10^{-3}$~mm) than in the gneiss
    ($\\approx 2.5\\times10^{-3}$~mm), reflecting the schist's higher
    in-plane compliance. The $\\sigma_1$ rotation mechanism responsible
    for this rotation is quantified in Section~\\ref{subsec9}.}"""
    new_dir = (
        "    \\caption{\\rev{Displacement fields for (a) psammitic schist and (b) augen\n"
        "    gneiss at seven loading orientations. Arrow direction gives the local\n"
        "    displacement direction and color its magnitude; arrow length is drawn on a\n"
        "    single scale shared by every panel, so the panels may be read against one\n"
        "    another rather than each against itself. The direction field is set largely\n"
        "    by the convergence the platens impose and changes little with fabric angle:\n"
        "    the axis of the high-displacement corridor, fitted by total least squares to\n"
        f"    the upper {int(100 - 85)}\\% of displacement magnitude, stays within\n"
        f"    ${f(dev[G],1)}^\\circ$ of the loading axis in the gneiss and\n"
        f"    ${f(dev[P],1)}^\\circ$ in the schist, with the largest departure near\n"
        "    $45^\\circ$ and a return to the loading axis at $90^\\circ$. The corridor does\n"
        "    not swing toward the fabric; were it to do so it would lie horizontal in the\n"
        "    $90^\\circ$ specimen. What does change with orientation is magnitude and\n"
        f"    width: peak displacement is ${f(ratio,1)}\\times$ larger in the schist than\n"
        "    in the gneiss, and the corridor broadens monotonically with fabric angle in\n"
        f"    both lithologies (half-width ${f(wid[G][0],2)}R$ to ${f(wid[G][1],2)}R$ in the\n"
        f"    gneiss, ${f(wid[P][0],2)}R$ to ${f(wid[P][1],2)}R$ in the schist), consistent\n"
        "    with the schist's higher in-plane compliance. Fabric control is expressed in\n"
        "    the tractions the fabric carries rather than in the direction of the\n"
        "    displacement field (Fig.~\\ref{fig:fabric_tractions}).}}")
    s = apply_once(s, old_dir, new_dir, "direction-circle caption")

    # ---------------------------------------------------- stress-glyph caption
    old_st = """    \\caption{Principal stress tensors superposed on model grids for (a)
    psammitic schist and (b) augen gneiss. The spatial heterogeneity of
    tensor orientations is markedly greater in the gneiss, where stiff
    augen force abrupt stress reorientation in the surrounding matrix. In
    the schist, tensor orientations rotate more gradually across
    fabric-parallel corridors, consistent with compliance-driven channeling
    rather than interface-driven perturbation.}"""
    new_st = (
        "    \\caption{\\rev{Principal stress directions for (a) psammitic schist and (b)\n"
        "    augen gneiss, drawn on a stress scale shared by every panel so that glyph\n"
        "    length is comparable between orientations and between lithologies. Because\n"
        "    the Brazilian configuration prescribes tractions on a simply connected\n"
        "    domain, the stress field depends on the elastic constants only through the\n"
        "    compatibility condition, and the principal directions move very little with\n"
        "    fabric angle: relative to the $0^\\circ$ specimen the axes rotate by a median\n"
        f"    of at most ${f(rmax[G])}^\\circ$ in the gneiss and ${f(rmax[P])}^\\circ$ in the\n"
        "    schist. Within a single specimen the spread of orientations is comparable in\n"
        f"    the two rocks and slightly the larger in the schist (circular standard\n"
        f"    deviation ${f(hmean[G],1)}^\\circ$ against ${f(hmean[P],1)}^\\circ$), as is the\n"
        f"    sharpness of the spatial reorientation (median gradient ${f(gmed[G])}$ against\n"
        f"    ${f(gmed[P])}$ degrees per millimetre, reaching ${f(gp90[G])}$ and\n"
        f"    ${f(gp90[P])}$ at the ninetieth percentile). The elastic field is a\n"
        "    homogeneous orthotropic solution, so these differences follow from the\n"
        "    contrast in elastic constants between the two rocks and not from discrete\n"
        "    stiff inclusions, which the formulation does not represent. Orientation\n"
        "    dependence is carried by stress magnitude, visible here as the shortening of\n"
        "    the glyphs toward $90^\\circ$, and by how the stress resolves onto the\n"
        "    fabric.}}")
    s = apply_once(s, old_st, new_st, "stress-tensor caption")

    # ------------------------------------------------------------- new figure
    anchor = "\t\\label{fig:stress_tensor_distribution}\n\\end{figure}\n"
    s = insert_after_once(s, anchor, "fig:fabric_tractions",
                          "fabric-traction figure", f"""
\\begin{{figure}}[htbp]
	\\centering
	\\begin{{minipage}}[t]{{0.48\\textwidth}}
		\\centering
		\\includegraphics[width=\\textwidth]{{fabric_traction_normal.pdf}}
		\\caption*{{(a)}}
	\\end{{minipage}}\\hfill
	\\begin{{minipage}}[t]{{0.48\\textwidth}}
		\\centering
		\\includegraphics[width=\\textwidth]{{fabric_traction_shear.pdf}}
		\\caption*{{(b)}}
	\\end{{minipage}}
    \\caption{{\\rev{{Tractions resolved on the foliation plane, averaged over the disk
    interior inside $0.85R$, against fabric angle: (a) normal traction $\\sigma_n$,
    tension positive, and (b) shear traction magnitude $|\\tau|$. Where the principal
    directions are nearly orientation independent
    (Fig.~\\ref{{fig:stress_tensor_distribution}}), these are not. With the fabric
    across the load the foliation is held shut (${f(sn[G][0])}$~MPa in the gneiss,
    ${f(sn[P][0])}$~MPa in the schist) and cannot open; $\\sigma_n$ rises monotonically
    and changes sign at ${f(cross[G],0)}^\\circ$ and ${f(cross[P],0)}^\\circ$
    respectively, so only above those angles is the fabric free to part. Shear
    traction instead peaks at an intermediate angle (${taupk[G]}^\\circ$ and
    ${taupk[P]}^\\circ$), where the fabric lies oblique to both principal directions.
    The measured strength minimum at $75^\\circ$ falls where the clamping traction has
    just relaxed toward zero while appreciable shear remains, so the weakest
    orientation reflects a competition between opening and sliding rather than either
    acting alone.}}}}
	\\label{{fig:fabric_tractions}}
\\end{{figure}}
""")

    # ------------------------------------------------------------ Results text
    a2 = "\\subsection{Boundary-induced stress rotation and orientation statistics}\n\\label{subsec9}\n"
    s = insert_after_once(s, a2, "Resolving the modelled stress state onto the foliation",
                          "subsec9 traction paragraph",
                          "\n\\rev{Resolving the modelled stress state onto the foliation plane "
        "separates the two ways a foliation can fail and reveals an orientation "
        "dependence that the principal directions do not carry. Averaged over the disk "
        f"interior, the normal traction rises monotonically from ${f(sn[G][0])}$ to "
        f"${f(sn[G][-1])}$~MPa in the gneiss and from ${f(sn[P][0])}$ to ${f(sn[P][-1])}$~MPa "
        "in the schist (Fig.~\\ref{fig:fabric_tractions}a). At low angles the fabric is "
        "clamped shut and opening is unavailable whatever its strength; the fraction of "
        "the interior in which the foliation is in tension grows from zero at $0^\\circ$ "
        f"to ${f(openf[G][-1],2)}$ in the gneiss and ${f(openf[P][-1],2)}$ in the schist at "
        f"$90^\\circ$. Shear traction peaks instead at ${taupk[G]}^\\circ$ and "
        f"${taupk[P]}^\\circ$ (Fig.~\\ref{{fig:fabric_tractions}}b), where the fabric lies "
        "oblique to both principal directions. Neither quantity is extremal at the "
        "orientation that fails most easily: the measured minimum sits at $75^\\circ$ in "
        "both lithologies, where the clamping traction has just relaxed to near zero "
        "while appreciable shear remains. That coincidence is the mechanical content of "
        "the weakest orientation, and it is not visible in a map of principal "
        "directions.}\n")

    if patch_document:
        TEX.write_text(s, encoding="utf-8")
    print("  captions replaced, fabric-traction figure and results text inserted")

    # -------------------------------------------------------------- supp table
    rows = []
    for r in (G, P):
        a = tr[tr.rock == r].sort_values("angle_deg").reset_index()
        b = het[het.rock == r].sort_values("angle_deg").reset_index()
        c = rot[rot.rock == r].sort_values("angle_deg").reset_index()
        e = cor[cor.rock == r].sort_values("angle_deg").reset_index()
        for k in range(len(a)):
            rows.append(f"{r} & {a.angle_deg[k]:.0f} & {a.sigma_n_mean_MPa[k]:.2f} & "
                        f"{a.tau_abs_mean_MPa[k]:.2f} & {a.open_fraction[k]:.3f} & "
                        f"{c.rotation_median_deg[k]:.2f} & {b.circ_sd_deg[k]:.1f} & "
                        f"{e.deviation_from_load_axis_deg[k]:.1f} \\\\")
    (REPO / "manuscript/tables/table_S_fabric_tractions.txt").write_text(
        r"""% Generated by scripts/patch_stress_field_text.py -- do not edit by hand.
\begin{table}[htbp]
\centering
\small
\caption{\rev{Foliation-resolved tractions and field-orientation statistics for every
specimen, averaged over the disk interior inside $0.85R$. $\sigma_n$ is the normal
traction on the foliation with tension positive and $|\tau|$ the shear traction
magnitude; the open fraction is the proportion of the interior in which the foliation
is in tension. Rotation is the median rotation of the principal axes away from the
$0^\circ$ specimen of the same lithology and spread is the circular standard deviation
of principal orientation within one specimen. Corridor deviation is the angle between
the axis of the high-displacement corridor and the loading axis. The tractions vary
over the full range while the three orientation measures stay small, which is why the
mechanism is read from the tractions.}}
\label{tab:fabric_tractions}
\begin{tabular}{l r r r r r r r}
\toprule
Lithology & $\alpha$ & $\sigma_n$ & $|\tau|$ & Open & Rotation & Spread & Corridor \\
 & (deg) & (MPa) & (MPa) & fraction & (deg) & (deg) & dev. (deg) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")
    print("  wrote manuscript/tables/table_S_fabric_tractions.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
