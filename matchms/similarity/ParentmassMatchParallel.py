from typing import List
import numba
import numpy
from matchms.typing import SpectrumType


class ParentmassMatchParallel:
    """Return 1 if spectrums match in parent mass (within tolerance), and 0 otherwise."""

    def __init__(self, tolerance: float = 0.1):
        """
        Parameters:
        ----------
        tolerance
            Specify tolerance below which two masses are counted as match.
        """
        self.tolerance = tolerance

    def __call__(self, reference_spectrums: List[SpectrumType],
                 spectrums: List[SpectrumType]) -> numpy.ndarray:
        """Compare parent masses between all reference_spectrums and spectrums."""
        def collect_parentmasses(spectrums):
            """Collect parentmasses."""
            parentmasses = []
            for spectrum in spectrums:
                parentmass = spectrum.get("parent_mass")
                assert parentmass is not None, "Missing parent mass."
                parentmasses.append(parentmass)
            return numpy.asarray(parentmasses)

        parentmasses_ref = collect_parentmasses(reference_spectrums)
        parentmasses_query = collect_parentmasses(spectrums)
        return calculate_parentmass_scores(parentmasses_ref, parentmasses_query, self.tolerance)


@numba.njit
def calculate_parentmass_scores(parentmasses_ref, parentmasses_query, tolerance):
    scores = numpy.zeros((len(parentmasses_ref), len(parentmasses_query)))
    for i, parentmass_ref in enumerate(parentmasses_ref):
        for j, parentmass_query in enumerate(parentmasses_query):
            scores[i, j] = 1 if (abs(parentmass_ref-parentmass_query) <= tolerance) else 0
    return scores
