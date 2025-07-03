from typing import Tuple
import numpy as np
from numba import njit
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


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
    score_datatype = [("score", np.float64)]

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
        mz1 = reference.peaks.mz
        int1 = reference.peaks.intensities
        mz2 = query.peaks.mz
        int2 = query.peaks.intensities

        score = compute_entropy(
            mz1, int1,
            mz2, int2,
            self.tolerance,
            self.use_ppm,
            self.total_norm
        )
        return np.asarray((score,), dtype=self.score_datatype)


@njit
def compute_entropy(
    spec1_mz: np.ndarray,
    spec1_int: np.ndarray,
    spec2_mz: np.ndarray,
    spec2_int: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    total_norm: bool
) -> float:
    """
    Compute Jensen–Shannon entropy similarity between two spectra in a single pass.

    This function merges two sorted spectra under a given tolerance (in Daltons or ppm),
    accumulates the Shannon entropies of each spectrum and their 1:1 mixture,
    and returns a normalized similarity score:

        similarity = 1 - 2 * (S_mixture - 0.5*(S1 + S2)) / ln(4)

    Parameters
    ----------
    spec1_mz:
        Ascending m/z values for spectrum A.
    spec1_int:
        Corresponding intensity values for spectrum A.
    spec2_mz:
        Ascending m/z values for spectrum B.
    spec2_int:
        Corresponding intensity values for spectrum B.
    tolerance:
        Matching tolerance. If `use_ppm` is False, this is in Daltons; otherwise in ppm.
    use_ppm:
        Whether to interpret `tolerance` as parts-per-million (ppm).
    total_norm:
        If True, normalize intensities by sum; if False, by maximum.
    """
    # Determine normalization divisors
    if total_norm:
        div1 = spec1_int.sum()
        div2 = spec2_int.sum()
    else:
        div1 = spec1_int.max()
        div2 = spec2_int.max()

    # Pointers
    id1, id2 = 0, 0
    n1, n2 = spec1_mz.shape[0], spec2_mz.shape[0]

    entropy = 0.0
    while id1 < n1 and id2 < n2:
        mz1 = spec1_mz[id1]
        mz2 = spec2_mz[id2]

        # Dynamic ppm tolerance if needed
        tol = tolerance * 1e-6 * mz1 if use_ppm else tolerance

        # Pre-normalized probabilities
        p1 = spec1_int[id1] / div1
        p2 = spec2_int[id2] / div2

        # Branch on m/z comparison
        if mz1 < mz2 - tol:
            if p1 > 0:
                entropy += p1 * np.log(2)
            id1 += 1
        elif mz2 < mz1 - tol:
            if p2 > 0:
                entropy += p2 * np.log(2)
            id2 += 1
        else:
            # Peaks match within tolerance
            if p1 > 0:
                entropy += p1 * np.log (2*p1 / (p1 + p2))
            if p2 > 0:
                entropy += p2 * np.log (2*p2 / (p1 + p2))
            id1 += 1
            id2 += 1

    # Remaining in spec1
    while id1 < n1:
        p1 = spec1_int[id1] / div1
        if p1 > 0:
            entropy += p1 * np.log(2)
        id1 += 1

    # Remaining in spec2
    while id2 < n2:
        p2 = spec2_int[id2] / div2
        if p2 > 0:
            entropy += p2 * np.log(2)
        id2 += 1

    return 1 - entropy / np.log(4)
