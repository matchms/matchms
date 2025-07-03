from typing import Tuple
import numpy as np
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .spectrum_similarity_functions import collect_peak_pairs, score_best_matches_entropy


class SpectralEntropy(BaseSimilarity):
    """Calculate the spectral entropy similarity between two spectra.

    This is the unweighted entropy similarity as in Li et al. Nat Methods (2024):
      Similarity = 1 - 2 * (S_AB - (S_A+S_B)/2) / ln(4)
    where S_X = -∑ p_i ln p_i on each normalized spectrum or their 1:1 mixture.

    Example
    -------

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import SpectralEntropy

        ref = Spectrum(mz=np.array([100, 150, 200.]),
                       intensities=np.array([0.7, 0.2, 0.1]))
        qry = Spectrum(mz=np.array([100, 140, 190.]),
                       intensities=np.array([0.4, 0.2, 0.1]))

        entropy = SpectralEntropy(tolerance=0.1)
        result = entropy.pair(ref, qry)
        print(f"Entropy score = {result['score']:.3f}, matches = {result['matches']}")

    """
    is_commutative = True
    score_datatype =  np.float64

    def __init__(
        self,
        tolerance: float = 0.1,
        use_ppm: bool = False,
        total_norm: bool = True
    ):
        """
        Parameters
        ----------
        tolerance:
            Matching tolerance. Interpreted in Daltons if `use_ppm` is False,
            otherwise in ppm. Default is 0.1.
        use_ppm:
            If True, interpret `tolerance` as parts-per-million. Default is False.
        total_norm:
            If True, normalize intensities by the sum; if False, by the maximum.
            Default is True.
        """
        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.total_norm = total_norm

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float]:
        """
        Calculate the entropy similarity between two spectra.

        Parameters
        ----------
        reference:
            Reference spectrum with sorted peaks (mz, intensities).
        query:
            Query spectrum with sorted peaks (mz, intensities).
        """
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=0.0,
            )
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind="mergesort")[::-1], :]
            return matching_pairs

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        if matching_pairs is None:
            return np.asarray(0.0, dtype=self.score_datatype)
        score = 1 - score_best_matches_entropy(
            matching_pairs, spec1, spec2, self.total_norm
        )
        return np.asarray(score, dtype=self.score_datatype)