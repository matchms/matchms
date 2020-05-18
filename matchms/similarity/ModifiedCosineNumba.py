import numba
import numpy as np
from matchms.typing import SpectrumType


class ModifiedCosineNumba:
    """Calculate modified cosine score between two spectra.

    This score is calculated based on matches between peaks of two spectra and
    also includes a shift of peak positions by the difference in precursor mass.

    Args:
    ----
    spectrum1: SpectrumType
        Input spectrum 1.
    spectrum2: SpectrumType
        Input spectrum 2.
    """
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def __call__(self, spectrum1: SpectrumType, spectrum2: SpectrumType) -> float:
        """Calculate cosine score between two spectra."""
        # normalize intensities:
        spec1 = np.vstack((spectrum1.peaks.mz, spectrum1.peaks.intensities)).T
        spec2 = np.vstack((spectrum2.peaks.mz, spectrum2.peaks.intensities)).T
        spec1[:, 1] = spec1[:, 1]/max(spec1[:, 1])
        spec2[:, 1] = spec2[:, 1]/max(spec2[:, 1])

        zero_pairs = find_pairs_numba(spec1, spec2, self.tolerance, shift=0.0)
        if spectrum1.get("precursor_mz") and spectrum2.get("precursor_mz"):
            mass_shift = spectrum1.get("precursor_mz") - spectrum2.get("precursor_mz")
            nonzero_pairs = find_pairs_numba(spec1, spec2, self.tolerance, shift=mass_shift)
            matching_pairs = zero_pairs + nonzero_pairs
        else:
            print("Precursor_mz missing. Calculate cosine score instead.",
                  "Consider applying 'add_precursor_mz' filter first.")
            matching_pairs = zero_pairs
        matching_pairs = sorted(matching_pairs, key=lambda x: x[2], reverse=True)

        used1 = set()
        used2 = set()
        score = 0.0
        used_matches = []
        for match in matching_pairs:
            if not match[0] in used1 and not match[1] in used2:
                score += match[2]
                used1.add(match[0])
                used2.add(match[1])
                used_matches.append(match)

        # Normalize score:
        score = score/max(np.sum(spec1[:, 1]**2), np.sum(spec2[:, 1]**2))
        return score, len(used_matches)


@numba.njit
def find_pairs_numba(spec1, spec2, tol, shift=0):
    """Find matching pairs between two spectra.

    Args
    ----
    spec1: numpy array
        Spectrum peaks and intensities as numpy array.
    spec2: numpy array
        Spectrum peaks and intensities as numpy array.
    tol : float
        Tolerance. Peaks will be considered a match when <= tol appart.
    shift : float, optional
        Shift spectra peaks by shift. The default is 0.

    Returns
    -------
    matching_pairs : list
        List of found matching peaks.
    """
    matching_pairs = []

    for idx in range(len(spec1)):
        intensity = spec1[idx, 1]
        matches = np.where((np.abs(spec2[:, 0] - spec1[idx, 0] + shift) <= tol))[0]
        for match in matches:
            matching_pairs.append((idx, match, intensity*spec2[match][1]))

    return matching_pairs
