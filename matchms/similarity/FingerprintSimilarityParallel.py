import numba
import numpy


class FingerprintSimilarityParallel:
    """Calculate similarity between molecules based on their fingerprints.

    Args:
    ----
    set_empty_scores: "nan" or 0
        Set values to this value if no fingerprint is found.
    """
    def __init__(self, similarity_measure="jaccard", set_empty_scores="nan"):
        self.set_empty_scores = set_empty_scores
        assert similarity_measure in ["cosine", "dice", "jaccard"], "Unknown similarity measure."
        self.similarity_measure = similarity_measure

    def __call__(self, references, queries):
        """Calculate matrix of fingerprint based similarity scores."""
        def get_fingerprints(spectrums):
            for index, spectrum in enumerate(spectrums):
                yield index, spectrum.get("fingerprint")

        def collect_fingerprints(spectrums):
            """Collect fingerprints and indices of spectrum with finterprints."""
            idx_fingerprints = []
            fingerprints = []
            for index, fp in get_fingerprints(spectrums):
                if fp is not None:
                    idx_fingerprints.append(index)
                    fingerprints.append(fp)
            fingerprints = numpy.array(fingerprints)
            return numpy.array(fingerprints), numpy.array(idx_fingerprints)

        def create_full_matrix():
            """Create matrix for all similarities."""
            similarity_matrix = numpy.zeros((len(references), len(queries)))
            if self.set_empty_scores == "nan":
                similarity_matrix[:] = numpy.nan
            elif isinstance(self.set_empty_scores, (float, int)):
                similarity_matrix[:] = self.set_empty_scores
            return similarity_matrix

        fingerprints1, idx_fingerprints1 = collect_fingerprints(references)
        fingerprints2, idx_fingerprints2 = collect_fingerprints(queries)
        assert idx_fingerprints1.size > 0 and idx_fingerprints2.size > 0, ("Not enouth molecular fingerprints.",
                                                                           "Apply 'add_fingerprint'filter first.")
        similarity_matrix = create_full_matrix()
        # Calculate similarity score matrix following specified method
        if self.similarity_measure == "jaccard":
            similarity_matrix[numpy.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = jaccard_similarity_matrix(fingerprints1,
                                                                                        fingerprints2)
        elif self.similarity_measure == "dice":
            similarity_matrix[numpy.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = dice_similarity_matrix(fingerprints1,
                                                                                     fingerprints2)
        elif self.similarity_measure == "cosine":
            similarity_matrix[numpy.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = cosine_score_matrix(fingerprints1,
                                                                                  fingerprints2)
        return similarity_matrix


@numba.njit
def jaccard_similarity_matrix(references, queries):
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries."""
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = numpy.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = jaccard_index(references[i,:], queries[j,:])
    return scores


@numba.njit
def dice_similarity_matrix(references, queries):
    """Returns matrix of dice similarity scores between all-vs-all vectors of references
    and queries."""
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = numpy.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = dice_similarity(references[i,:], queries[j,:])
    return scores


@numba.njit
def cosine_score_matrix(references, queries):
    """Returns matrix of cosine similarity scores between all-vs-all vectors of
    references and queries."""
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = numpy.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = cosine_similarity(references[i,:], queries[j,:])
    return scores


@numba.njit
def jaccard_index(u, v):
    """
    Computes the Jaccard-index (or Jaccard similarity coefficient) of two boolean
    1-D arrays.
    The Jaccard index between 1-D boolean arrays `u` and `v`,
    is defined as
    .. math::
       J(u,v) = \\frac{u \cap v}
                {u \cup v}

    Args:
    ----
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    jaccard_similarity : double
        The Jaccard similarity coefficient between vectors `u` and `v`.
    """
    u_or_v = numpy.bitwise_or(u != 0, v != 0)
    u_and_v = numpy.bitwise_and(u != 0, v != 0)
    jaccard_similarity = numpy.double(u_and_v.sum()) / numpy.double(u_or_v.sum())
    return jaccard_similarity


@numba.njit
def dice_similarity(u, v):
    """
    Computes the Dice similarity coefficient (DSC)between two boolean 1-D arrays.
    The Dice similarity coefficient between `u` and `v`, is
    .. math::
         DSC(u,v) = \\frac{2|u /cap v|}
                    {|x| + |v|}

    Args:
    ----
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.

    Returns
    -------
    dice_similarity : double
        The Dice similarity coefficient between 1-D arrays `u` and `v`.
    """
    u_and_v = numpy.bitwise_and(u != 0, v != 0)
    u_abs_sum = numpy.abs(u).sum()
    v_abs_sum = numpy.abs(v).sum()
    dice_similarity = 2.0 * numpy.double(u_and_v.sum()) / (u_abs_sum + v_abs_sum)
    return dice_similarity


@numba.njit
def cosine_similarity(u, v):
    """Calculate cosine similarity score.

    Args:
    ----
    u : numpy array, float
        Input vector.
    v : numpy array, float
        Input vector.

    Returns
    -------
    cosine_similarity : double
        The Jaccard similarity coefficient between vectors `u` and `v`.
    """
    assert u.shape[0] == v.shape[0], "Input vector must have same shape."
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cosine_similarity = 1
    if uu != 0 and vv != 0:
        cosine_similarity = uv / numpy.sqrt(uu * vv)
    return cosine_similarity
