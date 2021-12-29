from typing import Iterable
from typing import List


def get_first_common_element(first: Iterable[str], second: Iterable[str]) -> str:
    """ Get first common element from two lists.
    Returns 'None' if there are no common elements.
    """
    return next((item for item in first if item in second), None)


def get_common_keys(first: List[str], second: List[str]) -> List[str]:
    """Get common elements of two sets of strings in a case insensitive way.

    Args:
        first (List[str]): First list of strings.
        second (List[str]): List of strings to search for matches.

    Returns:
        List[str]: List of common elements without regarding case of first list.
    """
    return [value for value in first if value in second or value.lower() in second]


def filter_none(iterable: Iterable) -> Iterable:
    """Filter iterable to remove 'None' elements.

    Args:
        iterable (Iterable): Iterable to filter.

    Returns:
        Iterable: Filtered iterable.
    """
    return filter(lambda x: x is not None, iterable)
