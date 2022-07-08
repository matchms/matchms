import pickle
from matchms.Scores import ScoresBuilder


def scores_from_json(file_path):
    """
    Load :py:class:`~matchms.Score.Score` object from a json file.

    Parameters
    ----------
    file_path : str
        Path to json file containing scores.
    """
    return ScoresBuilder().from_json(file_path).build()


def scores_from_pickle(file_path):
    """
    Load :py:class:`~matchms.Score.Score` object from a pickle file.

    WARNING: Pickle files are not secure and may execute malicious code. Make sure that you are importing a pickle
        file from a trusted source.

    Parameters
    ----------
    file_path : str
        Path to pickle file containing scores.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
