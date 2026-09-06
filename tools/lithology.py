"""Per-lithology configuration and the authoritative specimen pairing table.

Sample identifiers are fixed by the experimental programme and must never be
shared between lithologies:

* augen gneiss     — samples 1–7,  fabric angles 0…90 deg in 15 deg steps
* psammitic schist — samples 8–14, fabric angles 0…90 deg in 15 deg steps

The canonical spelling is **psammitic** (machine token ``psammitic_schist``).
The historical misspelling ``psammatic`` survives in some generated filenames;
:data:`LEGACY_TOKENS` records it so callers can resolve old paths without
propagating the error into new output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from . import output_dirs

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(path):
    """Inverse of :func:`repo_relative`: make a stored path openable.

    Provenance columns are serialized relative to the repository so they are
    portable. Anything that reads one back and opens it must join it onto the
    root rather than trust the working directory, or the read succeeds only
    when the caller happens to be standing in the repository.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def repo_relative(path):
    """Repo-relative POSIX string for a path, for anything written to disk.

    Provenance columns record which input produced each row. Serialized as
    absolute paths they carry the author's home directory into a public
    archive and resolve nowhere on another machine, so the stored form is
    relative to the repository root. Paths outside the repo are returned
    unchanged. Callers that need to open the file join it back onto
    REPO_ROOT.
    """
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)

ANGLES_DEG = (0, 15, 30, 45, 60, 75, 90)

#: Historical misspellings seen in generated filenames, mapped to the canonical token.
LEGACY_TOKENS = {
    "psammatic_schist": "psammitic_schist",
    "psamatic_schist": "psammitic_schist",
    "psamitic_schist": "psammitic_schist",
    "psammetic_schist": "psammitic_schist",
}


@dataclass(frozen=True)
class Lithology:
    """Configuration for one lithology."""

    key: str                      # machine token, e.g. 'augen_gneiss'
    display_name: str             # e.g. 'Augen gneiss'
    sample_ids: tuple             # 7 specimen ids, in angle order
    angles_deg: tuple = ANGLES_DEG
    notebook: str = ""
    legacy_tokens: tuple = field(default_factory=tuple)

    # --- fabric geometry -----------------------------------------------
    # These were literals repeated in every long notebook cell, and they are
    # the only numbers that differed between the two lithology notebooks.
    # They live here so a section module can be written once and driven by
    # the rock it is handed.
    spacing_m: float = 0.010          # mean weak-plane (diametral) spacing
    spacing_warp_amp_m: float = 0.002  # phase-warp amplitude of that spacing

    def figure_stem(self, name: str) -> str:
        """Output-figure basename for this lithology, e.g. 'augen_gneiss_<name>'."""
        return f"{self.key}_{name}"

    def sample_id_spec(self) -> str:
        """Specimen range in the ``"1-7"`` form the command-line cells expect."""
        return f"{self.sample_ids[0]}-{self.sample_ids[-1]}"

    def sample_for_angle(self, angle_deg: int) -> int:
        """Specimen id at a given fabric angle."""
        return self.sample_ids[self.angles_deg.index(int(angle_deg))]

    def angle_for_sample(self, sample_id: int) -> int:
        """Fabric angle of a given specimen."""
        return self.angles_deg[self.sample_ids.index(int(sample_id))]

    def owns(self, sample_id: int) -> bool:
        return int(sample_id) in self.sample_ids


AUGEN_GNEISS = Lithology(
    key="augen_gneiss",
    display_name="Augen gneiss",
    sample_ids=(1, 2, 3, 4, 5, 6, 7),
    notebook="Tensile_augen_gneiss.ipynb",
    spacing_m=0.010,
    spacing_warp_amp_m=0.002,
)

PSAMMITIC_SCHIST = Lithology(
    key="psammitic_schist",
    display_name="Psammitic schist",
    sample_ids=(8, 9, 10, 11, 12, 13, 14),
    notebook="Tensile_psammitic_schist.ipynb",
    legacy_tokens=("psammatic_schist",),
    spacing_m=0.002,
    spacing_warp_amp_m=0.0004,
)

LITHOLOGIES = {lit.key: lit for lit in (AUGEN_GNEISS, PSAMMITIC_SCHIST)}


def get(key: str) -> Lithology:
    """Look up a lithology, accepting the historical misspelling."""
    k = str(key).strip().lower().replace(" ", "_")
    k = LEGACY_TOKENS.get(k, k)
    if k not in LITHOLOGIES:
        raise KeyError(f"unknown lithology {key!r}; expected one of {sorted(LITHOLOGIES)}")
    return LITHOLOGIES[k]


def lithology_of_sample(sample_id: int) -> Lithology:
    """Which lithology a specimen belongs to."""
    for lit in LITHOLOGIES.values():
        if lit.owns(sample_id):
            return lit
    raise KeyError(f"sample {sample_id} belongs to no lithology (expected 1-14)")


def observed_trace_path(sample_id: int, root: Path | None = None) -> Path:
    """Digitized laboratory trace for a specimen."""
    root = Path(root) if root is not None else REPO_ROOT
    lit = lithology_of_sample(sample_id)
    angle = lit.angle_for_sample(sample_id)
    return root / "crack_digitized_data" / f"{lit.key}_{angle}_sample_{sample_id}.csv"


def predicted_trace_path(sample_id: int, root: Path | None = None) -> Path:
    """DDM predicted trace for a specimen."""
    root = Path(root) if root is not None else REPO_ROOT
    return root / output_dirs.FIELD_DIR / f"ddm_crack_sample_{sample_id}.csv"


def field_cache_path(sample_id: int, root: Path | None = None) -> Path:
    """Cached full-field npz for a specimen."""
    root = Path(root) if root is not None else REPO_ROOT
    return (root / output_dirs.FIELDS_NPZ
            / f"sample_{int(sample_id):04d}_full_fields.npz")


def pairing_table():
    """The authoritative 14-row observed/predicted pairing, in specimen order."""
    rows = []
    for lit in (AUGEN_GNEISS, PSAMMITIC_SCHIST):
        for angle in lit.angles_deg:
            sid = lit.sample_for_angle(angle)
            rows.append(dict(sample=sid, lithology=lit.display_name,
                             lithology_key=lit.key, experimental_angle_deg=angle,
                             observed_path=observed_trace_path(sid),
                             predicted_path=predicted_trace_path(sid)))
    return rows


def assert_notebook_lithology(expected_key: str, sample_ids) -> None:
    """Guard against a notebook silently operating on the other lithology's samples.

    Intended to be called near the top of each lithology notebook.
    """
    lit = get(expected_key)
    bad = [int(s) for s in sample_ids if not lit.owns(int(s))]
    if bad:
        raise AssertionError(
            f"notebook configured for {lit.display_name} (samples {lit.sample_ids}) "
            f"but was given sample id(s) {bad}, which belong to another lithology")
