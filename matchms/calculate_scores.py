from .Scores import Scores
from .similarity.BaseSimilarity import BaseSimilarity
from .similarity.COOIndex import COOIndex
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


def calculate_scores(similarity_metric: BaseSimilarity, scores: Scores,
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
    # todo Currently Scores only supports sparse matrixes, so it doesn't make a lot of sense to compute a dense matrix,
    #  we still do this in the case when it is the first run without filters. But it will still be converted to a
    #  sparse matrix (even though that is less memory efficient)
    #  If in the future Scores also supports storing dense matrixes
    #  we could add more complicated logic here, e.g. checking if the mask is <1/3 of dense. Since there are more cases
    #  where a dense matrix is actually more memory efficient than sparse.

    if name is None:
        name = similarity_metric.__class__.__name__

    mask_indices = None
    if len(scores.scores.score_names) > 0:
        mask_indices = COOIndex(scores.scores.row, scores.scores.col)
    elif len(similarity_metric.score_filters) == 0:
        new_scores = similarity_metric.matrix(references=scores.references, queries=scores.queries)
        scores.scores.add_dense_matrix(new_scores,
                                      name,
                                      join_type=join_type)
        return scores

    new_scores = similarity_metric.sparse_array(references=scores.references, queries=scores.queries,
                                                mask_indices=mask_indices)

    scores.scores.add_sparse_data(new_scores.row,
                                  new_scores.column,
                                  new_scores.scores,
                                  name,
                                  join_type=join_type)
    return scores
