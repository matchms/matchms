"""Defines matchms Metadata class."""
import logging
from matchms.filtering.add_precursor_mz import _add_precursor_mz_metadata


logger = logging.getLogger("matchms")


class Metadata:
    """Class to handle spectrum metadata in matchms."""
    def __init__(self, metadata: dict = None, harmonize_defaults: bool = True):
        """

        Parameters
        ----------
        metadata : dict, optional
            DESCRIPTION. The default is None.
        harmonize_defaults : bool, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if metadata is None:
            self._metadata = {}
        elif isinstance(metadata, dict):
            self._metadata = metadata
        else:
            raise ValueError("Unexpected data type for metadata (should be dictionary, or None).")

        if harmonize_defaults is True:
            self.harmonize_defaults()

    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict.
        """
        return self._metadata.copy().get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict.
        """
        self._metadata[key] = value
        return self
          
    def harmonize_defaults(self):
        if self.get("ionmode") is not None:
            self.set("ionmode", self.get("ionmode").lower())
        self._metadata = _add_precursor_mz_metadata(self._metadata)

    @property
    def metadata(self):
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        