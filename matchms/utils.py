import csv
import logging
import os
from functools import lru_cache
from typing import Callable, Iterable, List
from warnings import warn
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


def fingerprint_export_warning(spectra: List[SpectrumType]):
    if any(x.get("fingerprint") is not None for x in spectra):
        logger.warning("fingerprint found but will not be written to file.")


def filter_empty_spectra(spectra: List[SpectrumType]) -> List[SpectrumType]:
    """Filter None values in spectra list.

    Parameters
    ----------
    spectra
        List of spectra to filter.
    """
    return [x for x in spectra if x is not None]


def rename_deprecated_params(param_mapping: dict, version: str = None) -> Callable:
    """Decorator for renaming old, deprecated parameters.

    Usage example:
    .. testcode::
    @rename_deprecated_params({"spectrums": "spectra"}, version="0.1.0")
    def example_func(spectra: List[Spectrum], another_param: str):
        some function logic using spectra

    example_func(spectrums: [...], another_param: "some string")

    Parameters
    ----------
    param_mapping
        Dict of mapping from old to new parameter names e.g., {"spectrums": "spectra"}
    version
        Version in which the parameters are marked as deprecated.

    Returns:
        Callable function.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # New args
            new_args = list(args)
            new_kwargs = kwargs.copy()

            # Handle positional arguments
            for i, (old_param, new_param) in enumerate(param_mapping.items()):
                if i < len(new_args):
                    new_kwargs[new_param] = new_args[i]
            print(new_kwargs.keys())

            # Handle keyword arguments
            for old_param, new_param in param_mapping.items():
                if old_param in kwargs:
                    new_kwargs[new_param] = kwargs.pop(old_param)

                    warning_msg = f"Parameter '{old_param}' is deprecated and will be removed in the future. Use '{new_param}' instead."
                    if version is not None:
                        warning_msg += f" -- Deprecated since version {version}."

                    warn(warning_msg, DeprecationWarning, stacklevel=2)

            # Remove old params in keyword arguments, if present
            for old_param in param_mapping.keys():
                new_kwargs.pop(old_param, None)

            # Create final args based on new keyword arguments
            final_args = []
            for i, param in enumerate(func.__code__.co_varnames):
                if param in new_kwargs:
                    final_args.append(new_kwargs.pop(param))
                elif i < len(new_args):
                    final_args.append(new_args[i])
                else:
                    break

            return func(*final_args, **new_kwargs)
        return wrapper
    return decorator
