from collections.abc import Mapping
import numpy as np
from pickydict import PickyDict
from .filtering.metadata_processing.add_precursor_mz import _add_precursor_mz_metadata
from .filtering.metadata_processing.add_retention import _add_retention, _retention_index_keys, _retention_time_keys
from .filtering.metadata_processing.interpret_pepmass import _interpret_pepmass_metadata
from .filtering.metadata_processing.make_charge_int import _convert_charge_to_int
from .utils import load_export_key_conversions, load_known_key_conversions


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

        metadata = Metadata({"Precursor_MZ": 201.5, "Compound Name": "SuperStuff"}, matchms_key_style=False)
        print(metadata["precursor_mz"])  # => 201.5
        print(metadata["compound_name"])  # => None (now you need to use "compound name")

    """

    _key_regex_replacements = {r"\s": "_", r"[!?.,;:]": ""}
    _key_replacements = load_known_key_conversions()

    def __init__(self, metadata: dict = None, matchms_key_style: bool = True):
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
            self.harmonize_keys()

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

    def harmonize_keys(self):
        """Runs default harmonization of metadata.

        Method harmonized metadata field names which include setting them to lower-case
        and running a series of regex replacements followed by the default field name
        replacements (such as precursor_mass --> precursor_mz).

        """
        self._data.key_regex_replacements = Metadata._key_regex_replacements
        self._data.key_replacements = Metadata._key_replacements

    def harmonize_values(self):
        """Runs default harmonization of metadata.

        This includes harmonizing entries for ionmode, retention time and index,
        charge, as well as the removal of invalid entries ("", "NA", "N/A", "NaN").
        """
        metadata_filtered = _interpret_pepmass_metadata(self.data)
        metadata_filtered = _add_precursor_mz_metadata(metadata_filtered)

        if metadata_filtered.get("ionmode"):
            metadata_filtered["ionmode"] = self.get("ionmode").lower()

        if metadata_filtered.get("retention_time"):
            metadata_filtered = _add_retention(metadata_filtered, "retention_time", _retention_time_keys)

        if metadata_filtered.get("retention_index"):
            metadata_filtered = _add_retention(metadata_filtered, "retention_index", _retention_index_keys)

        if metadata_filtered.get("parent"):
            metadata_filtered["parent"] = float(metadata_filtered.get("parent"))

        charge = metadata_filtered.get("charge")
        charge_int = _convert_charge_to_int(charge)
        if not isinstance(charge, int) and charge_int is not None:
            metadata_filtered["charge"] = charge_int

        invalid_entries = ["", "NA", "N/A", "NaN"]
        metadata_filtered = {k: v for k, v in metadata_filtered.items() if not (isinstance(v, str) and v in invalid_entries)}

        self.data = metadata_filtered

    # ------------------------------
    # Getters and Setters
    # ------------------------------
    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict."""
        return self._data.copy().get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict."""
        self._data[key] = value
        if self.matchms_key_style is True:
            self.harmonize_keys()
        return self

    def keys(self):
        """Retrieve all keys of :attr:`.metadata` dict."""
        return self._data.keys()

    def values(self):
        """Retrieve all values of :attr:`.metadata` dict."""
        return self._data.values()

    def items(self):
        """Retrieve all items (key, value pairs) of :attr:`.metadata` dict."""
        return self._data.items()

    def to_dict(self, export_style: str = "matchms"):
        """Returns a regular Python dictionary containing the metadata entries.

        Parameters
        ----------
        export_style:
            Specifies the naming style of the metadata fields.
            Default is "matchms".
        """
        if export_style == "matchms":
            return dict(self._data)
        key_conversions = load_export_key_conversions(export_style=export_style)
        keep_list = [x for x in self._data.keys() if x in key_conversions]
        converted_dict = {}
        for key in keep_list:
            value = self._data[key]
            if value != "" and key_conversions[key] != "":
                converted_dict[key_conversions[key]] = self._data[key]
        return converted_dict

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
                self.harmonize_keys()
        else:
            raise TypeError("Expected input of type dict or PickyDict.")

    @staticmethod
    def set_key_replacements(keys: dict):
        """Set key replacements for metadata harmonization.

        Parameters
        ----------
        keys:
            Dictionary with key replacements.
        """
        Metadata._key_replacements = keys
