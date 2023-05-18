import csv
import logging
import os
from functools import lru_cache
from typing import Iterable, List
from .typing import SpectrumType


logger = logging.getLogger("matchms")


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


def load_known_key_conversions(key_conversions_file: str = None) -> dict:
    """Load dictionary of known key conversions. Makes sure that file loading is cached.
    """
    if key_conversions_file is None:
        key_conversions_file = os.path.join(os.path.dirname(__file__), "data", "known_key_conversions.csv")
    assert os.path.isfile(key_conversions_file), f"Could not find {key_conversions_file}"
    return _load_key_conversions(key_conversions_file, 'known_synonym', 'matchms_default')


def load_export_key_conversions(export_key_conversions_file: str = None, export_style: str = None) -> dict:
    """Load dictionary of export key conversions. Makes sure that file loading is cached.
    """
    if export_key_conversions_file is None:
        export_key_conversions_file = os.path.join(os.path.dirname(__file__), "data", "export_key_conversions.csv")
    assert os.path.isfile(export_key_conversions_file), f"Could not find {export_key_conversions_file}"
    return _load_key_conversions(export_key_conversions_file, 'matchms', export_style)


@lru_cache(maxsize=4)
def _load_key_conversions(file: str, source: str, target: str) -> dict:
    with open(file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        key_conversions = {}
        for row in reader:
            key_conversions[row[source]] = row[target]

    return key_conversions


def fingerprint_export_warning(spectrums: List[SpectrumType]):
    if any(x.get("fingerprint") is not None for x in spectrums):
        logger.warning("fingerprint found but will not be written to file.")
