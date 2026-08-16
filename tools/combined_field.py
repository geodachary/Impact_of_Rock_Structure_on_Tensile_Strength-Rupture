"""The scalar rendered by the stress-distribution figure, stated explicitly.

The stress-distribution maps do not show a stress *component*. They show a
composite diagnostic scalar built from two parts of the stress state, which is
why the quantity is hard to read off the figure and why it needs stating in the
caption:

.. math::

    C \\;=\\; \\langle \\sigma_1 \\rangle_+ \\;+\\; w_c \\,
        \\max\\!\\left( \\min(\\sigma_{yy},\\,0),\\; -p_{\\mathrm{peak}} \\right)

with

``⟨σ₁⟩₊ = max(σ₁, 0)``
    the tensile part of the in-plane major principal stress, so compressive
    principal states contribute nothing to the first term;

``min(σ_yy, 0)`` clipped at ``−p_peak``
    the compressive part of the normal stress on the loading axis, floored at
    the peak platen contact pressure so the near-contact singularity cannot
    dominate the map;

``w_c = σ_xx^target / p_peak``, clipped to ``[0, 1]``
    a dimensionless weight that puts the two terms on a comparable scale.

Both terms are in MPa, tension-positive, in the global right-handed ``(x, y)``
frame with ``+y`` the loading axis, and the field is masked outside the disc.

The composite is a *diagnostic* built to show tensile drive and contact
compression in one map. It is not a stress invariant and it is not comparable
with a single component such as ``σ_xx``; the underlying components are cached
separately in ``outputs/fields/fields_npz/`` and remain available for anyone who
wants them.
"""

from __future__ import annotations

import numpy as np

#: Target value of the centre horizontal stress used to scale each solution, MPa.
TARGET_SIGMA_XX_CENTER_MPA = 0.63

DESCRIPTION = (
    "composite diagnostic scalar: tensile part of the major principal stress "
    "plus the weighted, contact-capped compressive part of the loading-axis "
    "normal stress; MPa, tension-positive, global (x, y) frame"
)

COLORBAR_LABEL = r"$\langle\sigma_1\rangle_+ + w_c\,\max(\min(\sigma_{yy},0),-p_{\rm peak})$ (MPa)"


def combined_field(s1, syy, p_peak_mpa, target_sigma_xx_center_mpa=None, mask=None):
    """Reproduce the plotted scalar from the cached stress components.

    Parameters are in MPa, tension-positive. ``p_peak_mpa`` is the peak platen
    contact pressure of that solution. Returns the field with ``NaN`` outside
    ``mask``.
    """
    target = (TARGET_SIGMA_XX_CENTER_MPA if target_sigma_xx_center_mpa is None
              else float(target_sigma_xx_center_mpa))
    s1 = np.asarray(s1, float)
    syy = np.asarray(syy, float)
    p_peak = float(p_peak_mpa)

    w_c = float(np.clip(target / (p_peak + 1e-30), 0.0, 1.0))
    sigma1_plus = np.maximum(s1, 0.0)
    syy_capped = np.maximum(np.minimum(syy, 0.0), -p_peak)
    out = sigma1_plus + w_c * syy_capped
    if mask is not None:
        out = np.where(np.asarray(mask, bool), out, np.nan)
    return out


def weight(p_peak_mpa, target_sigma_xx_center_mpa=None) -> float:
    """The dimensionless weight ``w_c`` applied to the compressive term."""
    target = (TARGET_SIGMA_XX_CENTER_MPA if target_sigma_xx_center_mpa is None
              else float(target_sigma_xx_center_mpa))
    return float(np.clip(target / (float(p_peak_mpa) + 1e-30), 0.0, 1.0))


def term_contributions(s1, syy, p_peak_mpa, mask=None,
                       target_sigma_xx_center_mpa=None) -> dict:
    """How much each term contributes, for stating the balance in the caption."""
    w_c = weight(p_peak_mpa, target_sigma_xx_center_mpa)
    s1 = np.asarray(s1, float)
    syy = np.asarray(syy, float)
    tensile = np.maximum(s1, 0.0)
    compressive = w_c * np.maximum(np.minimum(syy, 0.0), -float(p_peak_mpa))
    if mask is not None:
        m = np.asarray(mask, bool)
        tensile, compressive = tensile[m], compressive[m]
    return dict(
        w_c=w_c,
        tensile_term_mean=float(np.nanmean(tensile)),
        tensile_term_max=float(np.nanmax(tensile)),
        compressive_term_mean=float(np.nanmean(compressive)),
        compressive_term_min=float(np.nanmin(compressive)),
        combined_mean=float(np.nanmean(tensile + compressive)),
        combined_min=float(np.nanmin(tensile + compressive)),
        combined_max=float(np.nanmax(tensile + compressive)),
    )
