from collections.abc import Sequence
import numpy as np
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarity
from .default_parameters import (
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .entropy_greedy import EntropyGreedy
from .flash_index import FlashIndex
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

    ``pair`` uses :class:`EntropyGreedy`, ``matrix`` uses the
    SpectraCollection-native :class:`FlashEntropy`, and :meth:`search` performs
    sparse repeated search against a persistent :class:`FlashIndex`.

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
        Maximum peak m/z difference for a match. Interpreted as Da unless
        ``use_ppm=True``. Default is 0.02.
    use_ppm
        If True, interpret ``tolerance`` as a symmetric ppm tolerance.
    remove_precursor
        If True and ``precursor_mz`` metadata are available, remove peaks above
        ``precursor_mz + offset_to_precursor`` before scoring.
    offset_to_precursor
        Offset used when ``remove_precursor=True``. This will only keep 
        mz values <= precursor_mz + offset_to_precursor. Default is -1.6 Da.
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
        matching_mode: str = "fragment",
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        merge_within: float = 0.0,
        dtype: np.dtype = np.float64,
    ):
        self.matching_mode = matching_mode
        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.remove_precursor = remove_precursor
        self.offset_to_precursor = offset_to_precursor
        self.noise_cutoff = noise_cutoff
        self.merge_within = merge_within
        self.dtype = np.dtype(dtype)
        self.score_datatype = self.dtype

    def _pair_similarity(self) -> EntropyGreedy:
        return EntropyGreedy(
            matching_mode=self.matching_mode,
            tolerance=self.tolerance,
            use_ppm=self.use_ppm,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
            merge_within=self.merge_within,
            dtype=self.dtype,
        )

    def _flash_similarity(self) -> FlashEntropy:
        return FlashEntropy(
            matching_mode=self.matching_mode,
            tolerance=self.tolerance,
            use_ppm=self.use_ppm,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
            normalize_to_half=True,
            merge_within=self.merge_within,
            dtype=self.dtype,
        )

    def build_index(self, spectra) -> FlashIndex:
        """Build a reusable library index directly from a SpectraCollection."""
        return self._flash_similarity().build_index(spectra)

    def save_index(self, index: FlashIndex, filename) -> None:
        """Save a compatible persistent Flash index."""
        self._flash_similarity().save_index(index, filename)

    def load_index(self, filename) -> FlashIndex:
        """Load and validate a persistent Flash index."""
        return self._flash_similarity().load_index(filename)

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
        """Calculate a dense matrix of entropy similarity scores.

        Persistent indices are intentionally not part of this API. Use
        :meth:`search` for repeated queries against a pre-built library index.
        """
        return self._flash_similarity().matrix(
            spectra_1=spectra_1,
            spectra_2=spectra_2,
            score_fields=score_fields,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
        )

    def search(
        self,
        query_spectra,
        library_index: FlashIndex,
        *,
        precursor_tolerance: float | None = None,
        precursor_use_ppm: bool = False,
        min_score: float = 0.0,
        top_k: int | None = 20,
        query_batch_size: int = 10,
        n_jobs: int = -1,
        progress_bar: bool = True,
    ):
        """Search query spectra against a persistent Flash library index.

        Results are sparse ``Scores``. Query preprocessing remains
        SpectraCollection-native, and worker tasks reduce each query row to its
        retained hits before sending results back to the parent process.
        """
        return self._flash_similarity().search(
            query_spectra=query_spectra,
            library_index=library_index,
            precursor_tolerance=precursor_tolerance,
            precursor_use_ppm=precursor_use_ppm,
            min_score=min_score,
            top_k=top_k,
            query_batch_size=query_batch_size,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
