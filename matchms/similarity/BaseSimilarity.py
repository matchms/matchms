from abc import abstractmethod
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from matchms import Scores
from matchms.similarity.COOIndex import COOIndex
from matchms.similarity.COOMatrix import COOMatrix
from matchms.similarity.ScoreFilter import FilterScoreByValue
from matchms.typing import SpectrumType


class BaseSimilarity:
    """Similarity function base class.
    When building a custom similarity measure, inherit from this class and implement
    the desired methods.

    Attributes
    ----------
    is_commutative:
       Whether similarity function is commutative, which means that the order of spectra
       does not matter (similarity(A, B) == similarity(B, A)). Default is True.
    score_datatype:
        Data type for the score output, e.g. "float" or [("score", "float"), ("matches", "int")].
        If multiple data types are set, the main score should be set to "score" (used as default for filtering).
    """
    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = np.float64

    @abstractmethod
    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:
        """Method to calculate the similarity for one input pair.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
            The similarity score as numpy array (using self.score_datatype). For instance returning
            np.asarray(score, dtype=self.score_datatype)
        """
        raise NotImplementedError

    def calculate_scores(
            self, scores: Scores,
            filters: Tuple[FilterScoreByValue] = (),
            name: str = None,
            join_type="left"
            ) -> Scores:
        """
        Calculate the similarity between all reference objects vs all query objects using
        the most suitable available implementation of the given similarity_function.
        If Scores object already contains similarity scores, the newly computed measures
        will be added to a new layer (name --> layer name).
        Additional scores will be added as specified with join_type, the default being 'left'.

        Parameters
        ----------
        scores
            A scores object containing the references and queries and potentially previously calculated scores.
        filters
            A tuple of filters to apply to the scores, before storing.
        name
            Label of the new scores layer. If None, the name of the similarity_function class will be used.
        join_type
            Choose from left, right, outer, inner to specify the merge type.
        """
        def is_sparse_advisable():
            return (
                (len(scores.scores.score_names) > 0)  # already scores in Scores
                and (join_type in ["inner", "left"])  # inner/left join
                and (len(scores.scores.row) < (scores.n_rows * scores.n_cols) / 2)
            )
        if name is None:
            name = self.__class__.__name__

        if is_sparse_advisable():
            if filters == ():
                new_scores = self.sparse_array(references=scores.references,
                                           queries=scores.queries,
                                           mask_indices=COOIndex(scores.scores.row, scores.scores.col))
            else:
                new_scores = self.sparse_array_with_filter(references=scores.references,queries=scores.queries,
                                               mask_indices=COOIndex(scores.scores.row, scores.scores.col),
                                                           score_filters=filters)
        else:
            if filters == ():
                new_scores = self.matrix(scores.references,
                                         scores.queries,
                                         is_symmetric=scores.is_symmetric)
            else:
                new_scores = self.matrix_with_filter(scores.references, scores.queries,
                                                     is_symmetric=scores.is_symmetric, score_filters=filters)

        if isinstance(new_scores, COOMatrix):
            scores.scores.add_sparse_data(new_scores.row,
                                          new_scores.column,
                                          new_scores.scores,
                                          name,
                                          join_type=join_type)
            return scores
        if isinstance(new_scores, np.ndarray):
            scores.scores.add_dense_matrix(new_scores, name, join_type=join_type)
            return scores
        raise ValueError("The methods above should always return COOMatrix or np.ndarray")

    def matrix(
            self,
            references: np.ndarray[SpectrumType], queries: np.ndarray[SpectrumType],
            is_symmetric: bool = False
            ) -> np.ndarray:
        """
        Compute a dense similarity matrix for all pairs of reference and query spectra.

        Parameters
        ----------
        references:
            Collection of reference spectra.
        queries:
            Collection of query spectra.
        is_symmetric:
            Indicates if the similarity matrix is symmetric (e.g., for all-vs-all comparisons).
            When True, only the upper triangle of the matrix is computed and then mirrored,
            which can reduce computation time.
        """
        sim_matrix = np.zeros((len(references), len(queries)), dtype=self.score_datatype)
        if is_symmetric:
            if len(references) != len(queries):
                raise ValueError(f"Found unequal number of spectra {len(references)} and {len(queries)} while `is_symmetric` is True.")

            # Compute pairwise similarities
            for i_ref, reference in enumerate(tqdm(references, "Calculating similarities")):
                for i_query, query in enumerate(queries[i_ref:], start=i_ref):  # Compute only upper triangle
                    sim_matrix[i_ref, i_query] = self.pair(reference, query)
                    sim_matrix[i_query, i_ref] = sim_matrix[i_ref, i_query]
        else:
            # Compute pairwise similarities
            for i, reference in enumerate(tqdm(references, "Calculating similarities")):
                for j, query in enumerate(queries):
                    sim_matrix[i, j] = self.pair(reference, query)
        return sim_matrix

    def matrix_with_filter(
            self,
            references: List[SpectrumType], queries: List[SpectrumType],
            score_filters: Tuple[FilterScoreByValue],
            is_symmetric: bool = False
            ) -> COOMatrix:
        """Optional: Provide optimized method to calculate a sparse matrix with filtering applied directly.
        This is helpfull if the filter function is removing most scores. Important note, per score this takes about 12x
        the amount of memory. Therefore, doing filtering during compute is only worth it if you keep less than 1/12th of
        the scores. Otherwise, it is best to just use matrix, followed by a filter.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        score_filters
            Tuple of filters that should be applied to each score before scoring. If you want to run without filtering,
            it is best to run matrix(). Filters can also be run after computing a matrix first, however, for strict
            filtering this implementation can be more memory efficient. Only use matrix_with_filter if you expect to
            filter out more than 90% of your data. Otherwise, it is more memory efficient to first run matrix, followed by
            filtering the matrix.
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        n_rows = len(references)
        n_cols = len(queries)

        if is_symmetric and n_rows != n_cols:
            raise ValueError(f"Found unequal number of spectra {n_rows} and {n_cols} while `is_symmetric` is True.")

        idx_row = []
        idx_col = []
        scores = []
        # Wrap the outer loop with tqdm to track progress
        for i_ref, reference in enumerate(tqdm(references[:n_rows], desc="Calculating similarities")):
            if is_symmetric and self.is_commutative:
                for i_query, query in enumerate(queries[i_ref:n_cols], start=i_ref):
                    score = self.pair(reference, query)
                    # Check if the score passes the filter before storing.
                    if np.all(score_filter.keep_score(score) for score_filter in score_filters):
                        idx_row += [i_ref, i_query]
                        idx_col += [i_query, i_ref]
                        scores += [score, score]
            else:
                for i_query, query in enumerate(queries[:n_cols]):
                    score = self.pair(reference, query)
                    # Check if the score passes the filter before storing.
                    if np.all(score_filter.keep_score(score) for score_filter in score_filters):
                        idx_row.append(i_ref)
                        idx_col.append(i_query)
                        scores.append(score)
        return COOMatrix(row_idx=idx_row, column_idx=idx_col, scores=scores)


    def sparse_array(
            self,
            references: List[SpectrumType], queries: List[SpectrumType],
            mask_indices: COOIndex
            ) -> COOMatrix:
        """Optional: Provide optimized method to calculate a sparse matrix of similarity scores.

        Compute similarity scores for pairs of reference and query spectra as given by the indices
        idx_row (references) and idx_col (queries). If no method is added here, the following naive
        implementation (i.e. a for-loop) is used.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        mask_indices
            The row column index pairs for which a score should be calculated.
        """
        scores = np.zeros((len(mask_indices)), dtype=self.score_datatype)
        for i, (i_row, i_col) in enumerate(tqdm(mask_indices, desc="Calculating sparse similarities")):
            scores[i] = self.pair(references[i_row], queries[i_col])
        return COOMatrix(row_idx=mask_indices.idx_row, column_idx=mask_indices.idx_col, scores=scores)

    def sparse_array_with_filter(
            self,
            references: List[SpectrumType], queries: List[SpectrumType],
            mask_indices: COOIndex,
            score_filters: Tuple[FilterScoreByValue]
            ) -> COOMatrix:
        """Uses a mask to compute only the required scores and filters scores that do not pass the filter.

        This method most of the time does not make sense. It is only worth it if you want to store less than 1/12th
        of the computed scores. This is not the case most of the time, so sparse_array is most of the time the most
        memory efficient.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        mask_indices
            The row column index pairs for which a score should be calculated.
        score_filters
            Tuple of filters that should be applied to each score before scoring. If you want to run without filtering,
            it is best to run sparse_array(). Filters can also be run after computing a sparse array first, however, for strict
            filtering this implementation can be more memory efficient. Only use sparse_array_with_filter if you expect to
            filter out more than 90% of your data. Otherwise, it is more memory efficient to first run matrix, followed by
            filtering the matrix.
        """
        scores = []
        idx_row = []
        idx_col = []
        for row, col in tqdm(mask_indices, desc="Calculating sparse similarities"):
            score = self.pair(references[row], queries[col])
            # Check if the score passes the filter before storing.
            if np.all(score_filter.keep_score(score) for score_filter in score_filters):
                idx_row.append(row)
                idx_col.append(col)
                scores.append(score)
        return COOMatrix(row_idx=idx_row, column_idx=idx_col, scores=scores)


    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {"__Similarity__": self.__class__.__name__,
                **self.__dict__}
