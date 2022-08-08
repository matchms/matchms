import pickle
from matchms.Scores import ScoresBuilder


def scores_from_json(filename):
    """
    Load :py:class:`~matchms.Score.Score` object from a json file.

    Parameters
    ----------
    filename : str
        Path to json file containing scores.
    """
    return ScoresBuilder().from_json(filename).build()


def scores_from_pickle(filename):
    """
    Load :py:class:`~matchms.Score.Score` object from a pickle file.

    WARNING: Pickle files are not secure and may execute malicious code. Make sure that you are importing a pickle
        file from a trusted source.

    Parameters
    ----------
    filename : str
        Path to pickle file containing scores.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
