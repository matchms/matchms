"""High-level spectral entropy similarity for pairs, matrices, and library search."""
import numpy as np
from .default_parameters import (
    DEFAULT_DTYPE,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_similarity import FlashEntropy


class Entropy(FlashEntropy):
    """Compare mass spectra using entropy-weighted spectral similarity.

    Intensities are entropy-weighted and normalized to sum to 0.5 per spectrum.
    A matched pair contributes ``f(I1 + I2) - f(I1) - f(I2)``, with
    ``f(x) = x * log2(x)`` and ``f(0) = 0``. The sum is returned as a scalar
    ``score``. ``pair``, ``matrix``, and indexed ``search`` share these rules.

    Parameters
    ----------
    matching_mode
        ``"fragment"`` matches direct fragment coordinates one-to-one in
        ascending m/z order. ``"neutral_loss"`` matches only
        ``precursor_mz - fragment_mz`` coordinates, in ascending loss order.
        ``"hybrid"`` accepts fragment matches first, then loss matches between
        still-unused peaks. Missing precursors give zero loss-only scores and
        fragment-only hybrid scores, provided preprocessing retains the spectra.
    tolerance
        Inclusive coordinate tolerance for a match, in Da unless ``use_ppm``
        is True. In neutral-loss matching, this applies to loss coordinates.
    use_ppm
        Interpret tolerance as symmetric ppm rather than Da.
    remove_precursor
        Exclude peaks above ``precursor_mz + offset_to_precursor`` during
        preparation. Missing precursor handling follows the collection cleaner.
    offset_to_precursor
        Signed Da offset for the upper peak cutoff. Peaks at the boundary remain.
    noise_cutoff
        Minimum intensity relative to the maximum after precursor removal.
        Set to zero or None to disable the noise filter.
    merge_within
        Optional within-spectrum merge distance in Da. Zero disables merging.
    dtype
        Float32 or float64 for prepared peaks and the returned score. Calculations
        within the entropy kernel use float64 arithmetic.

    Notes
    -----
    The score formula and entropy weighting follow Li et al., Nature Methods
    18, 1524-1531 (2021), doi:10.1038/s41592-021-01331-z. Indexed accumulation is
    based on the Flash Entropy approach of Li and Fiehn, Nature Methods 20,
    1475-1478 (2023), doi:10.1038/s41592-023-02012-9.

    The index associates globally sorted library peaks with their source spectra.
    Matching entries are accumulated directly, with peak-use tracking for
    overlapping windows. Use ``build_index`` and ``search`` for repeated queries;
    use ``matrix`` for a complete comparison including index construction.
    """

    def __init__(
        self,
        matching_mode: str = "fragment",
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        merge_within: float = 0.0,
        dtype: np.dtype = DEFAULT_DTYPE,
    ):
        super().__init__(
            matching_mode=matching_mode,
            tolerance=tolerance,
            use_ppm=use_ppm,
            intensity_power=1.0,
            remove_precursor=remove_precursor,
            offset_to_precursor=offset_to_precursor,
            noise_cutoff=noise_cutoff,
            merge_within=merge_within,
            dtype=dtype,
            normalize_to_half=True,
        )

    def to_dict(self) -> dict:
        """Return the public entropy constructor parameters."""
        names = (
            "matching_mode", "tolerance", "use_ppm", "remove_precursor",
            "offset_to_precursor", "noise_cutoff", "merge_within",
        )
        return {
            "__Similarity__": type(self).__name__,
            **{name: getattr(self, name) for name in names},
            "dtype": self.dtype.name,
        }
