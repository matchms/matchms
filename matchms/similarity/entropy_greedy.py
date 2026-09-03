import numpy as np
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarityWithSparse
from .default_parameters import (
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_utils import _clean_and_weight


def _xlog2(x: float) -> float:
    """Return ``x * log2(x)`` with the continuous value 0 at x == 0."""
    if x <= 0.0:
        return 0.0
    return x * np.log2(x)


def _entropy_increment(intensity_1: float, intensity_2: float) -> float:
    """Return the entropy-similarity contribution of one matched peak pair."""
    return (
        _xlog2(intensity_1 + intensity_2)
        - _xlog2(intensity_1)
        - _xlog2(intensity_2)
    )


def _within_tolerance(
    mz_1: float,
    mz_2: float,
    tolerance: float,
    use_ppm: bool,
) -> bool:
    """Return whether two m/z values match within the configured tolerance."""
    if use_ppm:
        max_difference = tolerance * 1e-6 * 0.5 * (mz_1 + mz_2)
        return abs(mz_1 - mz_2) <= max_difference
    return abs(mz_1 - mz_2) <= tolerance


def _entropy_fragment_similarity_greedy(
    peaks_1: np.ndarray,
    peaks_2: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    *,
    used_1: np.ndarray | None = None,
    used_2: np.ndarray | None = None,
) -> float:
    """Calculate one-to-one entropy similarity from fragment m/z values.

    Peaks are consumed in ascending m/z order. If ``used_1`` and ``used_2`` are
    provided, already-used peaks are skipped and newly matched peaks are marked
    as used. This is used by hybrid matching to give fragment matches priority.
    """
    idx_1 = 0
    idx_2 = 0
    score = 0.0

    track_used = used_1 is not None and used_2 is not None

    while idx_1 < peaks_1.shape[0] and idx_2 < peaks_2.shape[0]:
        if track_used and used_1[idx_1]:
            idx_1 += 1
            continue
        if track_used and used_2[idx_2]:
            idx_2 += 1
            continue

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
            score += _entropy_increment(intensity_1, intensity_2)

            if track_used:
                used_1[idx_1] = True
                used_2[idx_2] = True

            idx_1 += 1
            idx_2 += 1

        elif mz_1 < mz_2:
            idx_1 += 1
        else:
            idx_2 += 1

    return score


def _entropy_neutral_loss_similarity_greedy(
    peaks_1: np.ndarray,
    peaks_2: np.ndarray,
    precursor_mz_1: float,
    precursor_mz_2: float,
    tolerance: float,
    use_ppm: bool,
    *,
    used_1: np.ndarray | None = None,
    used_2: np.ndarray | None = None,
) -> float:
    """Calculate one-to-one entropy similarity from neutral-loss values.

    The input peak arrays are sorted by increasing fragment m/z. Neutral losses
    ``precursor_mz - fragment_mz`` therefore run in increasing order when the
    peak arrays are traversed backwards. This avoids allocating and sorting
    separate neutral-loss arrays.

    If ``used_1`` and ``used_2`` are provided, peaks already consumed by fragment
    matching are skipped.
    """
    idx_1 = peaks_1.shape[0] - 1
    idx_2 = peaks_2.shape[0] - 1
    score = 0.0

    track_used = used_1 is not None and used_2 is not None

    while idx_1 >= 0 and idx_2 >= 0:
        if track_used and used_1[idx_1]:
            idx_1 -= 1
            continue
        if track_used and used_2[idx_2]:
            idx_2 -= 1
            continue

        intensity_1 = float(peaks_1[idx_1, 1])
        intensity_2 = float(peaks_2[idx_2, 1])

        if intensity_1 <= 0.0:
            idx_1 -= 1
            continue
        if intensity_2 <= 0.0:
            idx_2 -= 1
            continue

        neutral_loss_1 = precursor_mz_1 - float(peaks_1[idx_1, 0])
        neutral_loss_2 = precursor_mz_2 - float(peaks_2[idx_2, 0])

        if _within_tolerance(
            neutral_loss_1,
            neutral_loss_2,
            tolerance,
            use_ppm,
        ):
            score += _entropy_increment(intensity_1, intensity_2)

            if track_used:
                used_1[idx_1] = True
                used_2[idx_2] = True

            idx_1 -= 1
            idx_2 -= 1

        elif neutral_loss_1 < neutral_loss_2:
            # Moving backwards in fragment m/z increases neutral-loss m/z.
            idx_1 -= 1
        else:
            idx_2 -= 1

    return score


class EntropyGreedy(BaseSimilarityWithSparse):
    """Calculate entropy similarity between two mass spectra.

    This is a deliberately simple pair-oriented implementation of spectral
    entropy similarity. It is intended as the transparent reference
    implementation for single or small numbers of comparisons. For larger
    all-vs-all calculations, use :class:`~matchms.similarity.Entropy`, whose
    matrix implementation uses :class:`~matchms.similarity.FlashEntropy`.

    Intensities are preprocessed using entropy weighting and normalized to a
    total weighted intensity of 0.5 per spectrum.

    Three matching modes are supported:

    - ``"fragment"``:
      Match fragment m/z values directly. Each peak can be matched at most once.

    - ``"neutral_loss"``:
      Match neutral losses (``precursor_mz - fragment_mz``) only. Each peak can
      be matched at most once. If either spectrum has no precursor m/z, the
      resulting score is zero.

    - ``"hybrid"``:
      First perform one-to-one fragment matching. Peaks consumed by fragment
      matches cannot subsequently participate in neutral-loss matching.
      Remaining peaks are then matched one-to-one by neutral loss. If either
      spectrum has no precursor m/z, hybrid matching falls back to fragment-only
      matching.

    Parameters
    ----------
    matching_mode
        Matching strategy. Must be ``"fragment"``, ``"neutral_loss"``, or
        ``"hybrid"``. Default is ``"fragment"``.
    tolerance
        Maximum difference for a match. Interpreted as Da unless
        ``use_ppm=True``.
    use_ppm
        If True, interpret ``tolerance`` as a symmetric ppm tolerance.
    remove_precursor
        If True and precursor m/z metadata are available, remove peaks above
        ``precursor_mz + offset_to_precursor`` before scoring.
    offset_to_precursor
        Offset used when ``remove_precursor=True``. Peaks with
        ``m/z > precursor_mz + offset_to_precursor`` are removed.
    noise_cutoff
        Remove peaks below this fraction of the spectrum's maximum intensity.
        Set to 0 or None to disable.
    merge_within
        If > 0, merge neighboring peaks within this m/z distance during
        preprocessing.
    dtype
        Floating-point dtype used for preprocessing and returned scores.
    """

    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

    def __init__(
        self,
        matching_mode: str = "fragment",
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        merge_within: float = 0.0,
        dtype: np.dtype = np.float64,
    ):
        if matching_mode not in ("fragment", "neutral_loss", "hybrid"):
            raise ValueError(
                "matching_mode must be 'fragment', 'neutral_loss', or 'hybrid'."
            )
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0.")
        if merge_within < 0:
            raise ValueError("merge_within must be >= 0.")

        self.matching_mode = matching_mode
        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.remove_precursor = remove_precursor
        self.offset_to_precursor = offset_to_precursor
        self.noise_cutoff = noise_cutoff
        self.merge_within = merge_within
        self.dtype = np.dtype(dtype)
        self.score_datatype = self.dtype

    def _prepare(
        self,
        spectrum: SpectrumType,
    ) -> tuple[np.ndarray, float | None]:
        """Return entropy-weighted peaks and precursor m/z for one spectrum."""
        precursor_mz = spectrum.metadata.get("precursor_mz", None)

        peaks = _clean_and_weight(
            spectrum.peaks.to_numpy,
            precursor_mz,
            intensity_power=1.0,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
            normalize_to_half=True,
            merge_within_da=self.merge_within,
            weighing_type="entropy",
            dtype=self.dtype,
        )

        return peaks, (
            None
            if precursor_mz is None
            else float(precursor_mz)
        )

    def pair(
        self,
        spectrum_1: SpectrumType,
        spectrum_2: SpectrumType,
    ) -> np.ndarray:
        """Calculate entropy similarity for one spectrum pair."""
        peaks_1, precursor_mz_1 = self._prepare(spectrum_1)
        peaks_2, precursor_mz_2 = self._prepare(spectrum_2)

        if peaks_1.size == 0 or peaks_2.size == 0:
            return np.asarray(0.0, dtype=self.score_datatype)

        if self.matching_mode == "fragment":
            score = _entropy_fragment_similarity_greedy(
                peaks_1,
                peaks_2,
                tolerance=self.tolerance,
                use_ppm=self.use_ppm,
            )

        elif self.matching_mode == "neutral_loss":
            if precursor_mz_1 is None or precursor_mz_2 is None:
                score = 0.0
            else:
                score = _entropy_neutral_loss_similarity_greedy(
                    peaks_1,
                    peaks_2,
                    precursor_mz_1,
                    precursor_mz_2,
                    tolerance=self.tolerance,
                    use_ppm=self.use_ppm,
                )

        else:  # hybrid
            used_1 = np.zeros(peaks_1.shape[0], dtype=bool)
            used_2 = np.zeros(peaks_2.shape[0], dtype=bool)

            score = _entropy_fragment_similarity_greedy(
                peaks_1,
                peaks_2,
                tolerance=self.tolerance,
                use_ppm=self.use_ppm,
                used_1=used_1,
                used_2=used_2,
            )

            if precursor_mz_1 is not None and precursor_mz_2 is not None:
                score += _entropy_neutral_loss_similarity_greedy(
                    peaks_1,
                    peaks_2,
                    precursor_mz_1,
                    precursor_mz_2,
                    tolerance=self.tolerance,
                    use_ppm=self.use_ppm,
                    used_1=used_1,
                    used_2=used_2,
                )

        return np.asarray(score, dtype=self.score_datatype)