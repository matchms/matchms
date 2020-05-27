"""Collection of functions for calculating vector-vector similarities."""
import numba
import numpy


@numba.njit
def jaccard_similarity_matrix(references, queries):
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries."""
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = numpy.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = jaccard_index(references[i, :], queries[j, :])
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
            scores[i, j] = dice_similarity(references[i, :], queries[j, :])
    return scores


@numba.njit
def cosine_similarity_matrix(references, queries):
    """Returns matrix of cosine similarity scores between all-vs-all vectors of
    references and queries."""
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = numpy.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = cosine_similarity(references[i, :], queries[j, :])
    return scores


@numba.njit
def jaccard_index(u, v):
    r"""Computes the Jaccard-index (or Jaccard similarity coefficient) of two boolean
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
    jaccard_score = 0
    if u_or_v.sum() != 0:
        jaccard_score = numpy.double(u_and_v.sum()) / numpy.double(u_or_v.sum())
    return jaccard_score


@numba.njit
def dice_similarity(u, v):
    r"""Computes the Dice similarity coefficient (DSC) between two boolean 1-D arrays.

    The Dice similarity coefficient between `u` and `v`, is
    .. math::
         DSC(u,v) = \\frac{2|u \cap v|}
                    {|u| + |v|}

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
    u_abs_and_v_abs = numpy.abs(u).sum() + numpy.abs(v).sum()
    dice_score = 0
    if u_abs_and_v_abs != 0:
        dice_score = 2.0 * numpy.double(u_and_v.sum()) / u_abs_and_v_abs
    return dice_score


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
    cosine_score = 0
    if uu != 0 and vv != 0:
        cosine_score = uv / numpy.sqrt(uu * vv)
    return cosine_score
