"""Defines matchms Metadata class."""
import logging
import numpy as np
from matchms.filtering.add_precursor_mz import _add_precursor_mz_metadata


logger = logging.getLogger("matchms")


class Metadata:
    """Class to handle spectrum metadata in matchms."""
    def __init__(self, metadata: dict = None, harmonize_defaults: bool = True):
        """

        Parameters
        ----------
        metadata : dict
            Spectrum metadata as a dictionary.
        harmonize_defaults : bool, optional
            Set to False if metadata harmonization to default keys is not desired.
            The default is True.

        """
        if metadata is None:
            self._metadata = {}
        elif isinstance(metadata, dict):
            self._metadata = metadata
        else:
            raise ValueError("Unexpected data type for metadata (should be dictionary, or None).")

        self.harmonize_defaults = harmonize_defaults
        if harmonize_defaults is True:
            self.harmonize_metadata()

    def __eq__(self, other_metadata):
        if self.keys() != other_metadata.keys():
            return False
        for key, value in self.items():
            
            if isinstance(value, np.ndarray):
                if not np.all(value == other_metadata.get(key)):
                    return False
            elif value != other_metadata.get(key):
                return False
        return True

    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict.
        """
        return self._metadata.copy().get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict.
        """
        self._metadata[key] = value
        if self.harmonize_defaults is True:
            self.harmonize_metadata()
        return self

    def keys(self):
        """Retrieve all keys of :attr:`.metadata` dict.
        """
        return self._metadata.keys()

    def values(self):
        """Retrieve all values of :attr:`.metadata` dict.
        """
        return self._metadata.values()

    def items(self):
        """Retrieve all items (key, value pairs) of :attr:`.metadata` dict.
        """
        return self._metadata.items()

    def harmonize_metadata(self):
        if self.get("ionmode") is not None:
            self._metadata["ionmode"] = self.get("ionmode").lower()
        self._metadata = _add_precursor_mz_metadata(self._metadata)

    @property
    def metadata(self):
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
