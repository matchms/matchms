from typing import Optional
import numpy as np


class FilterScoreByValue:
    def __init__(self, value: float, operator = '>',
                 score_name: Optional[str] = "score"):
        self.score_name = score_name
        self.value = np.array(value, dtype=np.float64)
        self.operator = _get_operator(operator)

    def keep_score(self, score: np.ndarray) -> bool:
        """Unpacks the score and runs the filter function.

        Some scores have multiple scores (e.g. cosine and matches) this function makes sure the score is applied
        to the correct score"""
        if len(score.dtype) > 1:  # if structured array
            score = score[self.score_name]
        return self.filter_function(score)

    def filter_function(self, score: np.ndarray) -> bool:
        return self.operator(score, self.value)


def _get_operator(relation: str):
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
