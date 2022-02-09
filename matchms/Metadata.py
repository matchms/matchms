from collections.abc import Mapping
import numpy as np
from pickydict import PickyDict
from .utils import load_known_key_conversions


_key_regex_replacements = {r"\s": "_",
                           r"[!?.,;:]": ""}
_key_replacements = load_known_key_conversions()


class Metadata:
    """Class to handle spectrum metadata in matchms.

    Metadata entries will be stored as PickyDict dictionary in `metadata.data`.
    Unlike normal Python dictionaries, not all key names will be accepted.
    Key names will be forced to be lower-case to avoid confusions between key such
    as "Precursor_MZ" and "precursor_mz".

    To avoid the default harmonization of the metadata dictionary use the option
    `matchms_key_style=False`.


    Code example:

    .. code-block:: python

        metadata = Metadata({"Precursor_MZ": 201.5, "Compound Name": "SuperStuff"})
        print(metadata["precursor_mz"])  # => 201.5
        print(metadata["compound_name"])  # => SuperStuff

    Or if the matchms default metadata harmonization should not take place:

    .. code-block:: python

        metadata = Metadata({"Precursor_MZ": 201.5, "Compound Name": "SuperStuff"},
                            matchms_key_style=False)
        print(metadata["precursor_mz"])  # => 201.5
        print(metadata["compound_name"])  # => None (now you need to use "compound name")

    """
    def __init__(self, metadata: dict = None,
                 matchms_key_style: bool = True):
        """

        Parameters
        ----------
        metadata:
            Spectrum metadata as a dictionary.
        matchms_key_style:
            Set to False if metadata harmonization to default keys is not desired.
            The default is True.

        """
        if metadata is None:
            self._data = PickyDict({})
        elif isinstance(metadata, Mapping):
            self._data = PickyDict(metadata)
        else:
            raise ValueError("Unexpected data type for metadata (should be dictionary, or None).")

        self.matchms_key_style = matchms_key_style
        if self.matchms_key_style is True:
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

    def harmonize_metadata(self):
        """Runs default harmonization of metadata.

        Method harmonized metadata field names which includes setting them to lower-case
        and runing a series of regex replacements followed by default field name
        replacements (such as precursor_mass --> precursor_mz).

        """
        self._data.key_regex_replacements = _key_regex_replacements
        self._data.key_replacements = _key_replacements

    # ------------------------------
    # Getters and Setters
    # ------------------------------
    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict.
        """
        return self._data.copy().get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict.
        """
        self._data[key] = value
        if self.matchms_key_style is True:
            self.harmonize_metadata()
        return self

    def keys(self):
        """Retrieve all keys of :attr:`.metadata` dict.
        """
        return self._data.keys()

    def values(self):
        """Retrieve all values of :attr:`.metadata` dict.
        """
        return self._data.values()

    def items(self):
        """Retrieve all items (key, value pairs) of :attr:`.metadata` dict.
        """
        return self._data.items()

    def __getitem__(self, key=None):
        return self.get(key)

    def __setitem__(self, key, newvalue):
        self.set(key, newvalue)

    @property
    def data(self):
        return self._data.copy()

    @data.setter
    def data(self, new_dict):
        if isinstance(new_dict, PickyDict):
            self._data = new_dict
        elif isinstance(new_dict, Mapping):
            self._data = PickyDict(new_dict)
            if self.matchms_key_style is True:
                self.harmonize_metadata()
        else:
            raise TypeError("Expected input of type dict or PickyDict.")
