from typing import Optional
import numpy as np


class FilterScoreByValue:
    """
    A filter class to determine whether a given similarity score should be retained
    based on a threshold value and a comparison operator.
    """
    def __init__(self, value: float, operator = '>',
                 score_name: Optional[str] = "score"):
        """
        Initialize the filter with a threshold value, an operator, and an optional score name.

        Parameters
        ----------
        value:
            The threshold value to be used in the comparison.
        operator:
            A string representing the comparison operator.
            Supported values are '>', '<', '>=', '<=', '==', '!='.
            Defaults to '>'.
        score_name:
            The name of the score to filter on if the score is provided as a structured array.
            Defaults to "score".
        """
        self.score_name = score_name
        self.value = np.array(value, dtype=np.float64)
        self.operator = _get_operator(operator)

    def keep_score(self, score: np.ndarray) -> bool:
        """
        Determine whether the given score should be kept based on the filtering criteria.

        If the score is a structured numpy array (i.e., it contains multiple fields), the method
        extracts the score corresponding to self.score_name before applying the filter.

        Parameters
        ----------
        score
            The similarity score(s) to be evaluated. May be a simple array or a structured array.
        """
        if len(score.dtype) > 1:  # if structured array
            score = score[self.score_name]
        return self.filter_function(score)

    def filter_matrix(self, scores_matrix: np.ndarray) -> np.ndarray:
        """Filter all the scores in a matrix, the values that don't pass the filter are set to 0"""
        if len(scores_matrix.dtype) > 1:  # if structured array
            mask = self.operator(scores_matrix[self.score_name], self.value)
            for field in scores_matrix.dtype.names:
                scores_matrix[field][~mask] = 0
            return scores_matrix
        return np.where(self.operator(scores_matrix, self.value), scores_matrix, 0)

    def filter_function(self, score: np.ndarray) -> bool:
        """
        Apply the filtering operator to the score.

        Parameters
        ----------
        score : np.ndarray
            The score value (or values) to which the filter operator will be applied.
        """
        return self.operator(score, self.value)


def _get_operator(relation: str):
    """
    Retrieve the numpy comparison function corresponding to the provided operator string.

    Parameters
    ----------
    relation : str
        A string representing the desired comparison operator.
        Expected values are '>', '<', '>=', '<=', '==', or '!='.
    """
    relation = relation.strip()
    ops = {'>': np.greater,
           '<': np.less,
           '>=': np.greater_equal,
           '<=': np.less_equal,
           '==': np.equal,
           '!=': np.not_equal}
    if relation in ops:
        return ops[relation]
    raise ValueError(f"Unknown relation {relation}")
