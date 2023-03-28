"""Collection of functions for calculating vector-vector similarities."""
import numba
import numpy as np


@numba.njit
def jaccard_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = jaccard_index(references[i, :], queries[j, :])
    return scores


@numba.njit
def dice_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of dice similarity scores between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = dice_similarity(references[i, :], queries[j, :])
    return scores


@numba.njit
def cosine_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of cosine similarity scores between all-vs-all vectors of
    references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = cosine_similarity(references[i, :], queries[j, :])
    return scores


@numba.njit
def jaccard_index(u: np.ndarray, v: np.ndarray) -> np.float64:
    r"""Computes the Jaccard-index (or Jaccard similarity coefficient) of two boolean
    1-D arrays.
    The Jaccard index between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       J(u,v) = \\frac{u \cap v}
                {u \cup v}

    Parameters
    ----------
    u :
        Input array. Expects boolean vector.
    v :
        Input array. Expects boolean vector.

    Returns
    -------
    jaccard_similarity
        The Jaccard similarity coefficient between vectors `u` and `v`.
    """
    u_or_v = np.bitwise_or(u != 0, v != 0)
    u_and_v = np.bitwise_and(u != 0, v != 0)
    jaccard_score = 0
    if u_or_v.sum() != 0:
        jaccard_score = np.float64(u_and_v.sum()) / np.float64(u_or_v.sum())
    return jaccard_score


@numba.njit
def dice_similarity(u: np.ndarray, v: np.ndarray) -> np.float64:
    r"""Computes the Dice similarity coefficient (DSC) between two boolean 1-D arrays.

    The Dice similarity coefficient between `u` and `v`, is

    .. math::

         DSC(u,v) = \\frac{2|u \cap v|}
                    {|u| + |v|}

    Parameters
    ----------
    u
        Input array. Expects boolean vector.
    v
        Input array. Expects boolean vector.

    Returns
    -------
    dice_similarity
        The Dice similarity coefficient between 1-D arrays `u` and `v`.
    """
    u_and_v = np.bitwise_and(u != 0, v != 0)
    u_abs_and_v_abs = np.abs(u).sum() + np.abs(v).sum()
    dice_score = 0
    if u_abs_and_v_abs != 0:
        dice_score = 2.0 * np.float64(u_and_v.sum()) / np.float64(u_abs_and_v_abs)
    return dice_score


@numba.njit
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.float64:
    """Calculate cosine similarity score.

    Parameters
    ----------
    u
        Input vector.
    v
        Input vector.

    Returns
    -------
    cosine_similarity
        The Cosine similarity score between vectors `u` and `v`.
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
        cosine_score = uv / np.sqrt(uu * vv)
    return np.float64(cosine_score)
