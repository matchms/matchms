from typing import List
import numpy as np
from sparsestack import StackedSparseArray  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .linear_cosine_functions import linear_cosine_score, sirius_merge_close_peaks


class LinearCosine(BaseSimilarity):
    """Calculate 'linear cosine similarity score' between two spectra.

    This implements the LinearCosine similarity from SIRIUS (BOECKER lab), which
    achieves O(n+m) time complexity by requiring spectra to be "well-separated"
    (consecutive peaks more than 2x tolerance apart). A preprocessing step
    (sirius_merge_close_peaks) enforces this invariant by greedily merging close
    peaks in descending intensity order.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import LinearCosine

        reference = Spectrum(mz=np.array([100, 150, 200.]),
                             intensities=np.array([0.7, 0.2, 0.1]))
        query = Spectrum(mz=np.array([100, 140, 190.]),
                         intensities=np.array([0.4, 0.2, 0.1]))

        linear_cosine = LinearCosine(tolerance=0.2)
        score = linear_cosine.pair(reference, query)

        print(f"LinearCosine score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        LinearCosine score is 0.83 with 1 matched peaks

    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]  # type: ignore[assignment]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0, intensity_power: float = 1.0):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
            Peaks closer than 2 * tolerance are merged before scoring.
        mz_power:
            The power to raise m/z to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:  # type: ignore[override]
        """Calculate linear cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Score
            Tuple with cosine score and number of matched peaks.
        """
        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        spec1 = sirius_merge_close_peaks(spec1, self.tolerance)
        spec2 = sirius_merge_close_peaks(spec2, self.tolerance)
        score, matches = linear_cosine_score(spec1, spec2, self.tolerance, self.mz_power, self.intensity_power)
        return np.asarray((score, matches), dtype=self.score_datatype)

    def matrix(
        self,
        references: List[SpectrumType],
        queries: List[SpectrumType],
        array_type: str = "numpy",
        is_symmetric: bool = False,
        progress_bar: bool = True,
    ) -> np.ndarray:
        """Optimized matrix computation that precomputes merged spectra.

        Each spectrum is merged once (N+M calls to sirius_merge_close_peaks)
        instead of 2*N*M times in the naive double-loop approach.
        """
        n_rows = len(references)
        n_cols = len(queries)

        if is_symmetric and n_rows != n_cols:
            raise ValueError(f"Found unequal number of spectra {n_rows} and {n_cols} while `is_symmetric` is True.")

        merged_refs = [sirius_merge_close_peaks(r.peaks.to_numpy, self.tolerance) for r in references]
        merged_queries = (
            merged_refs
            if is_symmetric
            else [sirius_merge_close_peaks(q.peaks.to_numpy, self.tolerance) for q in queries]
        )

        idx_row_list = []
        idx_col_list = []
        scores_list = []

        for i_ref in tqdm(range(n_rows), desc="Calculating similarities", disable=not progress_bar):
            j_start = i_ref if (is_symmetric and self.is_commutative) else 0
            for i_query in range(j_start, n_cols):
                score, matches = linear_cosine_score(
                    merged_refs[i_ref], merged_queries[i_query], self.tolerance, self.mz_power, self.intensity_power
                )
                result = np.asarray((score, matches), dtype=self.score_datatype)
                if self.keep_score(result):
                    if is_symmetric and self.is_commutative:
                        idx_row_list += [i_ref, i_query]
                        idx_col_list += [i_query, i_ref]
                        scores_list += [result, result]
                    else:
                        idx_row_list.append(i_ref)
                        idx_col_list.append(i_query)
                        scores_list.append(result)

        idx_row = np.array(idx_row_list, dtype=np.int_)
        idx_col = np.array(idx_col_list, dtype=np.int_)
        scores_data = np.array(scores_list, dtype=self.score_datatype)

        if array_type == "numpy":
            scores_array = np.zeros(shape=(n_rows, n_cols), dtype=self.score_datatype)
            scores_array[idx_row, idx_col] = scores_data.reshape(-1)
            return scores_array
        if array_type == "sparse":
            scores_array = StackedSparseArray(n_rows, n_cols)
            scores_array.add_sparse_data(idx_row, idx_col, scores_data, "")
            return scores_array
        raise ValueError("array_type must be 'numpy' or 'sparse'.")
