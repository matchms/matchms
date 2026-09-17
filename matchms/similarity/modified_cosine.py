"""Modified cosine similarity using direct and precursor-shifted peak matches."""
from .cosine import Cosine


class ModifiedCosine(Cosine):
    """Compare spectra using fragment and neutral-loss candidate matches.

    A peak pair is a candidate when its fragment m/z values match, or when its
    ``precursor_mz - fragment_mz`` values match. In Da, the latter is equivalent
    to shifting fragments by the difference between precursor m/z values.

    Both candidate sets participate in one intensity-product greedy assignment.
    Each physical peak is used at most once. When the precursor difference is
    within tolerance, the indexed path uses direct fragment matches only.
    Missing precursors prevent neutral-loss matches in that path. Preprocessing
    can impose additional metadata requirements.

    The parameters and result fields are those of :class:`~matchms.similarity.Cosine`.
    Set ``use_hungarian=True`` to use optimal modified-cosine assignment for
    ``pair`` and ``matrix`` instead of indexed greedy assignment.

    Parameters
    ----------
    tolerance:
        Peaks will be considered a match when <= tolerance apart. Default is 0.01.
    intensity_power:
        The power to raise intensity to in the cosine function. The default is 1.
    use_hungarian:
        Whether to use the Hungarian algorithm to find the best matches. The default is False,
        which means that the greedy algorithm is used to find the best matches.
        The greedy algorithm is typically faster than the Hungarian algorithm, and for most
        applications the results are very similar.
    noise_cutoff:
        Minimum relative intensity for a peak to be considered. Default is 0.01.
        Will only be used if use_hungarian is False.
    remove_precursor:
        Whether to remove peaks with m/z values larger than the precursor-m/z (plus offset).
    offset_to_precursor:
        The offset to add to the precursor-m/z when removing peaks.
    """

    _default_mode = "hybrid"
