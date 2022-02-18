import csv
import os
from functools import lru_cache
from typing import Iterable, List


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


@lru_cache(maxsize=4)
def load_known_key_conversions(key_conversions_file: str = None) -> dict:
    """Load dictionary of known key conversions. Makes sure that file loading is cached.
    """
    if key_conversions_file is None:
        key_conversions_file = os.path.join(os.path.dirname(__file__), "data", "known_key_conversions.csv")
    assert os.path.isfile(key_conversions_file), f"Could not find {key_conversions_file}"

    with open(key_conversions_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        key_conversions = {}
        for row in reader:
            key_conversions[row['known_synonym']] = row['matchms_default']

    return key_conversions
