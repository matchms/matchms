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
        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy

        score = compute_entropy_optimal(
            spec1_mz=spec1[:, 0],
            spec1_int=spec1[:, 1],
            spec2_mz=spec2[:, 0],
            spec2_int=spec2[:, 1],
            tolerance=self.tolerance,
            use_ppm=self.use_ppm,
            total_norm=self.total_norm
        )
        return np.asarray(score, dtype=self.score_datatype)


@njit
def compute_entropy_optimal(
    spec1_mz: np.ndarray,
    spec1_int: np.ndarray,
    spec2_mz: np.ndarray,
    spec2_int: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    total_norm: bool
) -> float:
    """
    Compute Jensen–Shannon entropy similarity with optimal 1:1 peak matching
    (via a non-crossing max-weight DP), avoiding sub-optimal "first-peak" matches.
    """
    # Normalize intensities to probabilities p1, p2
    if total_norm:
        div1 = spec1_int.sum()
        div2 = spec2_int.sum()
    else:
        div1 = spec1_int.max()
        div2 = spec2_int.max()

    n1 = spec1_mz.shape[0]
    n2 = spec2_mz.shape[0]
    p1 = spec1_int / div1
    p2 = spec2_int / div2

    # Pre-compute the "unmatched entropy" baseline:
    #  Σ p1*ln2 + Σ p2*ln2 = ln(2) * (1 + 1) = 2*ln2  (since each p-vector sums to 1)
    ln2 = np.log(2.0)
    UE = 2.0 * ln2

    # Rolling‐array DP for maximum total "entropy‐gain" from matching
    # weight(i,j) = reduction in entropy if you match i↔j rather than leave them unmatched:
    #    w = -[ p1*ln(p1/(p1+p2)) + p2*ln(p2/(p1+p2)) ]
    prev = np.zeros(n2+1, dtype=np.float64)
    curr = np.zeros(n2+1, dtype=np.float64)

    for i in range(1, n1+1):
        mz1 = spec1_mz[i-1]
        # ppm or Dalton tolerance for this peak
        tol = (tolerance * 1e-6 * mz1) if use_ppm else tolerance
        low = mz1 - tol
        high = mz1 + tol

        curr[0] = 0.0
        for j in range(1, n2+1):
            # Carry forward the best of skipping one side
            a = prev[j]
            b = curr[j-1]
            best = a if a > b else b

            mz2 = spec2_mz[j-1]
            if mz2 >= low and mz2 <= high:
                # compute the match‐gain weight
                s1 = p1[i-1]
                s2 = p2[j-1]
                denom = s1 + s2

                # Only do log if denom > 0 (it always is unless both intensities are zero)
                w = -(s1 * np.log(s1/denom) + s2 * np.log(s2/denom))
                # Consider matching:
                tmp = prev[j-1] + w
                if tmp > best:
                    best = tmp

            curr[j] = best

        # Swap rows
        prev, curr = curr, prev

    # prev[n2] now holds the maximum total entropy‐reduction from all matched pairs
    total_gain = prev[n2]

    # Reconstruct the actual entropy and similarity ---
    entropy = UE - total_gain
    return 1.0 - entropy / np.log(4.0)
