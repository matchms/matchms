from typing import List
import numpy as np
from sparsestack import StackedSparseArray
from matchms.similarity.spectrum_similarity_functions import (
    number_matching, number_matching_symmetric)
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


class ParentMassMatch(BaseSimilarity):
    """Return True if spectrums match in parent mass (within tolerance), and False otherwise.

    Example to calculate scores between 2 spectrums and iterate over the scores

    .. testcode::

        import numpy as np
        from matchms import calculate_scores
        from matchms import Spectrum
        from matchms.similarity import ParentMassMatch

        spectrum_1 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "1", "parent_mass": 100})
        spectrum_2 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "2", "parent_mass": 110})
        spectrum_3 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "3", "parent_mass": 103})
        spectrum_4 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "4", "parent_mass": 111})
        references = [spectrum_1, spectrum_2]
        queries = [spectrum_3, spectrum_4]

        similarity_score = ParentMassMatch(tolerance=5.0)
        scores = calculate_scores(references, queries, similarity_score)

        for (reference, query, score) in scores:
            print(f"Parentmass match between {reference.get('id')} and {query.get('id')}" +
                  f" is {score}")

    Should output

    .. testoutput::

        Parentmass match between 1 and 3 is [1.0]
        Parentmass match between 2 and 4 is [1.0]

    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g.  "float" or [("score", "float"), ("matches", "int")]
    score_datatype = bool

    def __init__(self, tolerance: float = 0.1):
        """
        Parameters
        ----------
        tolerance
            Specify tolerance below which two masses are counted as match.
        """
        self.tolerance = tolerance

    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
        """Compare parent masses between reference and query spectrum.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.
        """
        parentmass_ref = reference.get("parent_mass")
        parentmass_query = query.get("parent_mass")
        assert parentmass_ref is not None and parentmass_query is not None, "Missing parent mass."

        score = abs(parentmass_ref - parentmass_query) <= self.tolerance
        return np.asarray(score, dtype=self.score_datatype)

    def matrix(self, references: List[SpectrumType], queries: List[SpectrumType],
               array_type: str = "numpy",
               is_symmetric: bool = False) -> np.ndarray:
        """Compare parent masses between all references and queries.

        Parameters
        ----------
        references
            List/array of reference spectrums.
        queries
            List/array of Single query spectrums.
        array_type
            Specify the output array type. Can be "numpy" or "sparse".
            Default is "numpy" and will return a numpy array. "sparse" will return a COO-sparse array.
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        def collect_parentmasses(spectrums):
            """Collect parentmasses."""
            parentmasses = []
            for spectrum in spectrums:
                parentmass = spectrum.get("parent_mass")
                assert parentmass is not None, "Missing parent mass."
                parentmasses.append(parentmass)
            return np.asarray(parentmasses)

        parentmasses_ref = collect_parentmasses(references)
        parentmasses_query = collect_parentmasses(queries)

        if is_symmetric:  # assuming ref and query are identical
            rows, cols, scores = number_matching_symmetric(parentmasses_ref,
                                                           self.tolerance)
        else:
            rows, cols, scores = number_matching(parentmasses_ref, parentmasses_query,
                                                 self.tolerance)

        if array_type == "numpy":
            scores_array = np.zeros((len(parentmasses_ref), len(parentmasses_query)))
            scores_array[rows, cols] = scores.astype(self.score_datatype)
            return scores_array
        if array_type == "sparse":
            scores_array = StackedSparseArray(len(parentmasses_ref), len(parentmasses_query))
            scores_array.add_sparse_data(rows, cols, scores.astype(self.score_datatype), "")
            return scores_array
        return ValueError("`array_type` can only be 'numpy' or 'sparse'.")
