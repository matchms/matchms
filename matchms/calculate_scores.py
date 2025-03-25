from typing import Tuple

import numpy as np

from . import Scores
from .Scores import Scores
from .similarity.BaseSimilarity import BaseSimilarity
from .similarity.COOIndex import COOIndex
from .similarity.COOMatrix import COOMatrix
from .similarity.ScoreFilter import FilterScoreByValue
from .typing import QueriesType, ReferencesType


def create_scores_object_and_calculate_scores(references: ReferencesType, queries: QueriesType,
                     similarity_function: BaseSimilarity,
                     is_symmetric: bool = False) -> Scores:
    """Calculate the similarity between all reference objects versus all query objects.

    Example to calculate scores between 2 spectra and iterate over the scores

    .. testcode::

        import numpy as np
        from matchms import calculate_scores, Spectrum
        from matchms.similarity import CosineGreedy

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={'id': 'spectrum1'})
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={'id': 'spectrum2'})
        spectra = [spectrum_1, spectrum_2]

        scores = calculate_scores(spectra, spectra, CosineGreedy())

        for (reference, query, score) in scores:
            print(f"Cosine score between {reference.get('id')} and {query.get('id')}" +
                  f" is {score[0]:.2f} with {score[1]} matched peaks")

    Should output

    .. testoutput::

        Cosine score between spectrum1 and spectrum1 is 1.00 with 3 matched peaks
        Cosine score between spectrum1 and spectrum2 is 0.83 with 1 matched peaks
        Cosine score between spectrum2 and spectrum1 is 0.83 with 1 matched peaks
        Cosine score between spectrum2 and spectrum2 is 1.00 with 3 matched peaks

    Parameters
    ----------
    references
        List of reference objects
    queries
        List of query objects
    similarity_function
        Function which accepts a reference + query object and returns numpy matrix of scores
    is_symmetric
        Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
        comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
        2x faster. Default is False.

    Returns
    -------

    ~matchms.Scores.Scores
    """
    scores = Scores(references=references, queries=queries,
                  is_symmetric=is_symmetric)
    scores = calculate_scores(similarity_function, scores)
    return scores


def calculate_scores(similarity_metric, scores: Scores,
                     filters: Tuple[FilterScoreByValue] = (),
                     name: str = None,
                     join_type="left") -> Scores:
    """
    Calculate the similarity between all reference objects vs all query objects using
    the most suitable available implementation of the given similarity_function.
    If Scores object already contains similarity scores, the newly computed measures
    will be added to a new layer (name --> layer name).
    Additional scores will be added as specified with join_type, the default being 'left'.

    Parameters
    ----------
    similarity_metric
        The Similarity function to use.
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
        name = similarity_metric.__class__.__name__

    if is_sparse_advisable():
        if filters == ():
            new_scores = similarity_metric.sparse_array(references=scores.references,
                                                        queries=scores.queries,
                                                        mask_indices=COOIndex(scores.scores.row, scores.scores.col))
        else:
            new_scores = similarity_metric.sparse_array_with_filter(references=scores.references, queries=scores.queries,
                                                                    mask_indices=COOIndex(scores.scores.row, scores.scores.col),
                                                                    score_filters=filters)
    else:
        if filters == ():
            new_scores = similarity_metric.matrix(scores.references,
                                                  scores.queries,
                                                  is_symmetric=scores.is_symmetric)
        else:
            new_scores = similarity_metric.matrix_with_filter(scores.references, scores.queries,
                                                              is_symmetric=scores.is_symmetric, score_filters=filters)

    if isinstance(new_scores, COOMatrix):
        scores.scores.add_sparse_data(new_scores.row,
                                      new_scores.column,
                                      new_scores.scores,
                                      name,
                                      join_type="left")
        return scores
    if isinstance(new_scores, np.ndarray):
        scores.scores.add_dense_matrix(new_scores, name, join_type=join_type)
        return scores
    raise ValueError("The methods above should always return COOMatrix or np.ndarray")
