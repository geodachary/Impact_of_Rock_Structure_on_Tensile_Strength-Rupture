"""The lithology a section module is currently running for.

Section modules are shared by both rocks, so the numbers that distinguish them
-- specimen ids, weak-plane spacing, phase-warp amplitude, output stem -- cannot
be module-level literals. They are read from the lithology bound here by
``main(rock)`` before any of the section's own code runs.

A module-level binding rather than a threaded argument is deliberate. These
cells define their solvers as deeply nested closures, and several of the
lithology numbers appear as *default arguments*, which Python binds once at
import. Threading a parameter through every frame would mean rewriting hundreds
of signatures; binding once per ``main`` call reproduces exactly what the
notebook did when it executed a cell top to bottom.

The binding is process-wide, so a section runs one rock at a time. That matches
how the notebooks use it -- each notebook drives a single lithology -- and
:func:`current` raises rather than guessing if nothing is bound.
"""
from __future__ import annotations

from contextlib import contextmanager

_BOUND = None


def bind(rock):
    """Make ``rock`` the lithology this section runs for."""
    global _BOUND
    _BOUND = rock
    return rock


def current():
    """The bound lithology, or a clear error if ``main`` was bypassed."""
    if _BOUND is None:
        raise RuntimeError(
            "no lithology bound: call this section's main(rock) rather than "
            "its internals, or use tools.analysis._context.using(rock)")
    return _BOUND


@contextmanager
def using(rock):
    """Bind ``rock`` for the duration of a block, restoring any previous one."""
    global _BOUND
    previous = _BOUND
    _BOUND = rock
    try:
        yield rock
    finally:
        _BOUND = previous
