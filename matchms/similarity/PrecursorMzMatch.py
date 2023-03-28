from typing import List
import numpy as np
from sparsestack import StackedSparseArray
from matchms.similarity.spectrum_similarity_functions import (
    number_matching, number_matching_ppm, number_matching_symmetric,
    number_matching_symmetric_ppm)
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


class PrecursorMzMatch(BaseSimilarity):
    """Return True if spectrums match in precursor m/z (within tolerance), and False otherwise.
    The match within tolerance can be calculated based on an absolute m/z difference
    (tolerance_type="Dalton") or based on a relative difference in ppm (tolerance_type="ppm").

    Example to calculate scores between 2 pairs of spectrums and iterate over the scores

    .. testcode::

        import numpy as np
        from matchms import calculate_scores
        from matchms import Spectrum
        from matchms.similarity import PrecursorMzMatch

        spectrum_1 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "1", "precursor_mz": 100})
        spectrum_2 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "2", "precursor_mz": 110})
        spectrum_3 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "3", "precursor_mz": 103})
        spectrum_4 = Spectrum(mz=np.array([]),
                              intensities=np.array([]),
                              metadata={"id": "4", "precursor_mz": 111})
        references = [spectrum_1, spectrum_2]
        queries = [spectrum_3, spectrum_4]

        similarity_score = PrecursorMzMatch(tolerance=5.0, tolerance_type="Dalton")
        scores = calculate_scores(references, queries, similarity_score)

        for (reference, query, score) in scores:
            print(f"Precursor m/z match between {reference.get('id')} and {query.get('id')}" +
                  f" is {score}")

    Should output

    .. testoutput::

        Precursor m/z match between 1 and 3 is [1.0]
        Precursor m/z match between 2 and 4 is [1.0]

    """
    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = bool

    def __init__(self, tolerance: float = 0.1, tolerance_type: str = "Dalton"):
        """
        Parameters
        ----------
        tolerance
            Specify tolerance below which two m/z are counted as match.
        tolerance_type
            Chose between fixed tolerance in Dalton (="Dalton") or a relative difference
            in ppm (="ppm").
        """
        self.tolerance = tolerance
        assert tolerance_type in ["Dalton", "ppm"], "Expected type from ['Dalton', 'ppm']"
        self.type = tolerance_type

    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
        """Compare precursor m/z between reference and query spectrum.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.
        """
        precursormz_ref = reference.get("precursor_mz")
        precursormz_query = query.get("precursor_mz")
        assert precursormz_ref is not None and precursormz_query is not None, "Missing precursor m/z."

        if self.type == "Dalton":
            return abs(precursormz_ref - precursormz_query) <= self.tolerance

        mean_mz = (precursormz_ref + precursormz_query) / 2
        score = abs(precursormz_ref - precursormz_query)/mean_mz <= self.tolerance
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
        def collect_precursormz(spectrums):
            """Collect precursors."""
            precursors = []
            for spectrum in spectrums:
                precursormz = spectrum.get("precursor_mz")
                assert precursormz is not None, "Missing precursor m/z."
                precursors.append(precursormz)
            return np.asarray(precursors)

        precursors_ref = collect_precursormz(references)
        precursors_query = collect_precursormz(queries)
        if is_symmetric and self.type == "Dalton":
            rows, cols, scores =  number_matching_symmetric(precursors_ref,
                                                            self.tolerance)
        elif is_symmetric and self.type == "ppm":
            rows, cols, scores = number_matching_symmetric_ppm(precursors_ref,
                                                               self.tolerance)
        elif self.type == "Dalton":
            rows, cols, scores = number_matching(precursors_ref, precursors_query,
                                                 self.tolerance)
        else:
            rows, cols, scores = number_matching_ppm(precursors_ref, precursors_query,
                                                     self.tolerance)
        if array_type == "numpy":
            scores_array = np.zeros((len(precursors_ref), len(precursors_query)))
            scores_array[rows, cols] = scores.astype(self.score_datatype)
            return scores_array
        if array_type == "sparse":
            scores_array = StackedSparseArray(len(precursors_ref), len(precursors_query))
            scores_array.add_sparse_data(rows, cols, scores.astype(self.score_datatype), "")
            return scores_array
        return ValueError("`array_type` can only be 'numpy' or 'sparse'.")
