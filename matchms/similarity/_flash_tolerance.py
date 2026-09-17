"""Matching predicates and conservative candidate intervals for indexed peaks."""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def within(a: float, b: float, tol: float, ppm: bool) -> bool:
    """Test the inclusive Da or symmetric-ppm tolerance on two coordinates."""
    if ppm:
        return abs(a - b) <= tol * 1e-6 * 0.5 * (a + b)
    return abs(a - b) <= tol


@njit(cache=True, inline="always")
def window(coords: np.ndarray, x: float, tol: float, ppm: bool) -> tuple[int, int]:
    """Return conservative candidate bounds in a sorted coordinate array.

    The symmetric-ppm interval is bounded using its larger, upper-side radius.
    Bounds are expanded by one floating-point step to avoid losing candidates
    through endpoint arithmetic. Callers must still apply ``within`` to every
    candidate; the search interval alone does not define a match.
    """
    if ppm and tol > 0:
        if x < 0:
            return 0, 0
        width = (tol * 1e-6 * x) / (1.0 - 0.5 * tol * 1e-6)
    else:
        width = tol if not ppm else 0.0
    low = np.nextafter(x - width, -np.inf)
    high = np.nextafter(x + width, np.inf)
    return np.searchsorted(coords, low, side="left"), np.searchsorted(coords, high, side="right")


@njit(cache=True, inline="always")
def identity_allowed(query_pmz: float, library_pmz: float, tolerance: float, ppm: bool) -> bool:
    """Apply an optional precursor gate; a negative tolerance disables it.

    When enabled, both precursors must be finite and match within tolerance.
    """
    return tolerance < 0 or (
        np.isfinite(query_pmz)
        and np.isfinite(library_pmz)
        and within(query_pmz, library_pmz, tolerance, ppm)
    )
