import numpy as np
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarityWithSparse
from .default_parameters import (
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
)
from .flash_utils import _clean_and_weight


def _xlog2(x: float) -> float:
    """Return ``x * log2(x)`` with the continuous value 0 at x == 0."""
    if x <= 0.0:
        return 0.0
    return x * np.log2(x)


def _within_tolerance(mz_1: float, mz_2: float, tolerance: float, use_ppm: bool) -> bool:
    """Return whether two m/z values match within the configured tolerance."""
    if use_ppm:
        max_difference = tolerance * 1e-6 * 0.5 * (mz_1 + mz_2)
        return abs(mz_1 - mz_2) <= max_difference
    return abs(mz_1 - mz_2) <= tolerance


def _entropy_similarity_greedy(
    peaks_1: np.ndarray,
    peaks_2: np.ndarray,
    tolerance: float,
    use_ppm: bool,
) -> float:
    """Calculate entropy similarity using a one-to-one m/z-ordered sweep.

    Both peak arrays must be sorted by increasing m/z and their intensities must
    already be entropy-weighted and normalized to sum to 0.5 per spectrum.

    The matching rule intentionally mirrors the fragment-mode pairwise path used
    by :class:`~matchms.similarity.FlashEntropy`: when several peaks fall within
    one tolerance neighborhood, peaks are consumed one-to-one in ascending m/z
    order. This keeps this simple pair implementation consistent with the fast
    matrix implementation.
    """
    idx_1 = 0
    idx_2 = 0
    score = 0.0

    while idx_1 < peaks_1.shape[0] and idx_2 < peaks_2.shape[0]:
        intensity_1 = float(peaks_1[idx_1, 1])
        intensity_2 = float(peaks_2[idx_2, 1])

        if intensity_1 <= 0.0:
            idx_1 += 1
            continue
        if intensity_2 <= 0.0:
            idx_2 += 1
            continue

        mz_1 = float(peaks_1[idx_1, 0])
        mz_2 = float(peaks_2[idx_2, 0])

        if _within_tolerance(mz_1, mz_2, tolerance, use_ppm):
            score += (
                _xlog2(intensity_1 + intensity_2)
                - _xlog2(intensity_1)
                - _xlog2(intensity_2)
            )
            idx_1 += 1
            idx_2 += 1
        elif mz_1 < mz_2:
            idx_1 += 1
        else:
            idx_2 += 1

    return score


class EntropyGreedy(BaseSimilarityWithSparse):
    """Calculate entropy similarity between two mass spectra.

    This is a deliberately simple pair-oriented implementation of the spectral
    entropy similarity introduced by Li et al. (Nature Methods, 2021). It is
    intended as the transparent baseline implementation for single or small
    numbers of comparisons. For larger all-vs-all calculations, use
    :class:`~matchms.similarity.Entropy`, whose ``matrix()`` method uses the
    Flash Entropy implementation.

    Intensities are preprocessed using the same entropy weighting as the current
    Matchms Flash implementation. Each spectrum is normalized to a total weighted
    intensity of 0.5. Matching peaks are then consumed one-to-one in ascending
    m/z order and contribute the entropy-similarity increment
    ``xlog2(I1 + I2) - xlog2(I1) - xlog2(I2)``.

    Parameters
    ----------
    tolerance
        Maximum peak m/z difference for a match. Interpreted as Da unless
        ``use_ppm=True``. Default is 0.01.
    use_ppm
        If True, interpret ``tolerance`` as a symmetric ppm tolerance.
    remove_precursor
        If True and ``precursor_mz`` metadata are available, remove peaks above
        ``precursor_mz - precursor_window`` before scoring.
    precursor_window
        Window used when ``remove_precursor=True``. Default is 1.6 Da.
    noise_cutoff
        Remove peaks below this fraction of the spectrum's maximum intensity.
        Set to 0 or None to disable. Default is 0.01.
    merge_within
        If > 0, merge neighboring peaks within this m/z distance during
        preprocessing. Default is 0.
    dtype
        Floating-point dtype used for preprocessing and the returned score.
        Default is ``np.float``.
    """

    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

    def __init__(
        self,
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        remove_precursor: bool = False,
        precursor_window: float = 1.6,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        merge_within: float = 0.0,
        dtype: np.dtype = np.float64,
    ):
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0.")
        if merge_within < 0:
            raise ValueError("merge_within must be >= 0.")

        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.remove_precursor = remove_precursor
        self.precursor_window = precursor_window
        self.noise_cutoff = noise_cutoff
        self.merge_within = merge_within
        self.dtype = np.dtype(dtype)
        self.score_datatype = self.dtype

    def _prepare(self, spectrum: SpectrumType) -> np.ndarray:
        """Return entropy-weighted, normalized peaks for one spectrum."""
        precursor_mz = spectrum.metadata.get("precursor_mz", None)
        return _clean_and_weight(
            spectrum.peaks.to_numpy,
            precursor_mz,
            intensity_power=1.0,
            remove_precursor=self.remove_precursor,
            precursor_window=self.precursor_window,
            noise_cutoff=self.noise_cutoff,
            normalize_to_half=True,
            merge_within_da=self.merge_within,
            weighing_type="entropy",
            dtype=self.dtype,
        )

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> np.ndarray:
        """Calculate entropy similarity for one spectrum pair."""
        peaks_1 = self._prepare(spectrum_1)
        peaks_2 = self._prepare(spectrum_2)

        if peaks_1.size == 0 or peaks_2.size == 0:
            return np.asarray(0.0, dtype=self.score_datatype)

        score = _entropy_similarity_greedy(
            peaks_1,
            peaks_2,
            tolerance=self.tolerance,
            use_ppm=self.use_ppm,
        )
        return np.asarray(score, dtype=self.score_datatype)
