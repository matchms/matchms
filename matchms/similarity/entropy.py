from collections.abc import Sequence
import numpy as np
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarity
from .entropy_greedy import EntropyGreedy
from .flash_similarity import FlashEntropy


class Entropy(BaseSimilarity):
    """Calculate spectral entropy similarity between mass spectra.

    This is the central Matchms entropy-similarity class and should normally be
    the first choice for users interested in spectral entropy similarity.

    The class combines two implementations behind one API:

    - :meth:`pair` uses :class:`~matchms.similarity.EntropyGreedy`, a compact
      pair-oriented baseline implementation.
    - :meth:`matrix` uses :class:`~matchms.similarity.FlashEntropy`, which is the
      efficient implementation for larger all-vs-all comparisons.

    Both paths use the same entropy weighting, noise filtering, normalization,
    tolerance definition, and one-to-one fragment matching semantics.

    Spectral entropy similarity was introduced by Li et al., Nature Methods 18,
    1524-1531 (2021), doi:10.1038/s41592-021-01331-z. The accelerated Flash
    Entropy search strategy was introduced by Li & Fiehn, Nature Methods 20,
    1475-1478 (2023), doi:10.1038/s41592-023-02012-9.

    Parameters
    ----------
    tolerance
        Maximum peak m/z difference for a match. Interpreted as Da unless
        ``use_ppm=True``. Default is 0.02.
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
        Floating-point dtype used for scoring. Default is ``np.float64``.
    """

    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

    def __init__(
        self,
        tolerance: float = 0.02,
        use_ppm: bool = False,
        remove_precursor: bool = False,
        precursor_window: float = 1.6,
        noise_cutoff: float | None = 0.01,
        merge_within: float = 0.0,
        dtype: np.dtype = np.float64,
    ):
        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.remove_precursor = remove_precursor
        self.precursor_window = precursor_window
        self.noise_cutoff = noise_cutoff
        self.merge_within = merge_within
        self.dtype = np.dtype(dtype)
        self.score_datatype = self.dtype

    def _pair_similarity(self) -> EntropyGreedy:
        return EntropyGreedy(
            tolerance=self.tolerance,
            use_ppm=self.use_ppm,
            remove_precursor=self.remove_precursor,
            precursor_window=self.precursor_window,
            noise_cutoff=self.noise_cutoff,
            merge_within=self.merge_within,
            dtype=self.dtype,
        )

    def _matrix_similarity(self) -> FlashEntropy:
        similarity = FlashEntropy(
            matching_mode="fragment",
            tolerance=self.tolerance,
            use_ppm=self.use_ppm,
            remove_precursor=self.remove_precursor,
            precursor_window=self.precursor_window,
            noise_cutoff=self.noise_cutoff,
            normalize_to_half=True,
            merge_within=self.merge_within,
            dtype=self.dtype,
        )

        # FlashEntropy historically declared a float32 score datatype even when
        # its internal dtype was configured differently. For the public Entropy
        # wrapper, keep the returned Scores dtype aligned with the requested dtype.
        similarity.score_datatype = self.dtype
        return similarity

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> np.ndarray:
        """Calculate entropy similarity for one spectrum pair."""
        return self._pair_similarity().pair(spectrum_1, spectrum_2)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ):
        """Calculate a dense matrix of entropy similarity scores."""
        return self._matrix_similarity().matrix(
            spectra_1=spectra_1,
            spectra_2=spectra_2,
            score_fields=score_fields,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
        )
