import logging
from collections.abc import Sequence
import numpy as np
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarity
from .cosine_greedy import CosineGreedy
from .cosine_hungarian import CosineHungarian
from .default_parameters import (
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_index import FlashIndex
from .flash_similarity import CosineFlash


logger = logging.getLogger("matchms")


class Cosine(BaseSimilarity):
    """Calculate Cosine scores between mass spectra.

    This is matchms central Cosine class. 
    The Cosine score aims at quantifying the similarity between two
    mass spectra. Two peaks are considered a potential match if their m/z ratios
    lie within the given ``tolerance``.

    ``pair`` uses the greedy or Hungarian pair implementation. ``matrix`` uses
    the SpectraCollection-native Flash implementation for the default greedy path.
    For repeated searches against a fixed large library, build a persistent
    :class:`FlashIndex` once with :meth:`build_index` and use :meth:`search`.
    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]
    score_fields = ("score", "matches")

    def __init__(
        self,
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        intensity_power: float = DEFAULT_INTENSITY_POWER,
        use_hungarian: bool = False,
        noise_cutoff: float = DEFAULT_NOISE_CUTOFF,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
    ):
        """Initialize cosine score class.

        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
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
        self.tolerance = tolerance
        self.intensity_power = intensity_power
        self.use_hungarian = use_hungarian
        self.noise_cutoff = noise_cutoff
        self.remove_precursor = remove_precursor
        self.offset_to_precursor = offset_to_precursor

    def _flash_similarity(self) -> CosineFlash:
        if self.use_hungarian:
            raise ValueError(
                "Persistent Flash indices/search are unavailable when "
                "use_hungarian=True."
            )
        return CosineFlash(
            matching_mode="fragment",
            tolerance=self.tolerance,
            intensity_power=self.intensity_power,
            noise_cutoff=self.noise_cutoff,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
        )

    def build_index(self, spectra) -> FlashIndex:
        """Build a reusable library index.

        When ``spectra`` is already a ``SpectraCollection``, index construction is
        fully collection-native and operates directly on its CSR fragment backend.
        """
        return self._flash_similarity().build_index(spectra)

    def save_index(self, index: FlashIndex, filename) -> None:
        """Save a compatible persistent Flash index."""
        self._flash_similarity().save_index(index, filename)

    def load_index(self, filename) -> FlashIndex:
        """Load and validate a persistent Flash index."""
        return self._flash_similarity().load_index(filename)

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> tuple[float, int]:
        """Calculate cosine score between two spectra."""
        if self.use_hungarian:
            cosine = CosineHungarian(
                tolerance=self.tolerance,
                intensity_power=self.intensity_power,
                noise_cutoff=self.noise_cutoff,
                remove_precursor=self.remove_precursor,
                offset_to_precursor=self.offset_to_precursor,
            )
        else:
            cosine = CosineGreedy(
                tolerance=self.tolerance,
                intensity_power=self.intensity_power,
                noise_cutoff=self.noise_cutoff,
                remove_precursor=self.remove_precursor,
                offset_to_precursor=self.offset_to_precursor,
            )
        return cosine.pair(spectrum_1, spectrum_2)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ):
        """Calculate a dense matrix of Cosine scores.

        Persistent indices are intentionally not part of this API. Use
        :meth:`search` for repeated queries against a pre-built library index.

        Parameters
        ----------
        spectra_1
            First collection of input spectra.
        spectra_2
            Second collection of input spectra. If None, compare `spectra_1`
            against itself.
        score_fields
            Requested score fields. Only ``("score",)`` is supported.
        progress_bar
            When True, show a progress bar.
        n_jobs
            Number of parallel jobs to run.
            Default is -1, which means that all available CPUs minus one will be used.

        Returns
        -------
        Scores
            Dense score matrix as a ``Scores`` object.
        """
        if self.use_hungarian:
            cosine = CosineHungarian(
                tolerance=self.tolerance,
                intensity_power=self.intensity_power,
                noise_cutoff=self.noise_cutoff,
                remove_precursor=self.remove_precursor,
                offset_to_precursor=self.offset_to_precursor,
            )
            return cosine.matrix(
                spectra_1=spectra_1,
                spectra_2=spectra_2,
                score_fields=score_fields,
                progress_bar=progress_bar,
            )

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
        min_matches: int | None = None,
        top_k: int | None = 20,
        query_batch_size: int = 10,
        n_jobs: int = -1,
        progress_bar: bool = True,
    ):
        """Search query spectra against a persistent Flash library index.

        Results are returned as sparse ``Scores`` with shape
        ``(n_query_spectra, n_library_spectra)``. Each worker handles a batch of
        query rows and returns only thresholded/top-k hits, avoiding transfer or
        storage of dense query-by-library result matrices.

        Parameters
        ----------
        query_batch_size
            Number of query rows handled per worker task. This reduces scheduling
            and inter-process overhead while retaining row-level Flash scoring.
            A default of 10 is deliberately conservative for very large libraries;
            values around 10-50 are good candidates for benchmarking.
        """
        return self._flash_similarity().search(
            query_spectra=query_spectra,
            library_index=library_index,
            precursor_tolerance=precursor_tolerance,
            precursor_use_ppm=precursor_use_ppm,
            min_score=min_score,
            min_matches=min_matches,
            top_k=top_k,
            query_batch_size=query_batch_size,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
