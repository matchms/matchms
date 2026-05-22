import inspect
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from matchms.exporting import save_spectra
from matchms.filtering.filter_order import ALL_FILTERS
from matchms.filtering.SpectrumProcessor import (
    check_all_parameters_given,
    create_partial_function,
    get_parameter_settings,
    load_matchms_filter_from_string,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.yaml_file_functions import ordered_dump


logger = logging.getLogger("matchms")

FunctionWithParametersType = tuple[Callable | str, dict[str, object]]


class SpectraCollectionProcessor:
    """Process a SpectraCollection using a series of filters.

    This is the SpectraCollection equivalent of SpectrumProcessor, but it applies
    each filter to the full collection instead of processing spectra one by one.

    Parameters
    ----------
    filters
        A list of filter functions. Allowed formats are the same as for
        SpectrumProcessor:

        - str
        - (str, dict)
        - Callable
        - (Callable, dict)

    Examples
    --------
    Create a SpectraCollection and process it with collection-compatible filters:

    .. code-block:: python

        import numpy as np

        from matchms import Spectrum, SpectraCollection
        from matchms.filtering import SpectraCollectionProcessor

        spectra = [
            Spectrum(
                mz=np.array([100.0, 150.0, 200.0]),
                intensities=np.array([5.0, 50.0, 500.0]),
                metadata={"smiles": "n/a", "compound_name": "example"},
            ),
            Spectrum(
                mz=np.array([110.0, 160.0, 210.0]),
                intensities=np.array([10.0, 100.0, 1000.0]),
                metadata={"smiles": "CCCO", "compound_name": "other"},
            ),
        ]

        collection = SpectraCollection(spectra)

        processor = SpectraCollectionProcessor(
            filters=[
                "harmonize_missing_entries",
                (
                    "select_by_relative_intensity",
                    {"intensity_from": 0.01, "intensity_to": 1.0},
                ),
            ]
        )

        processed = processor.process_collection(collection)

        assert isinstance(processed, SpectraCollection)

    The same processor can also create a SpectraCollection from an iterable of
    Spectrum objects:

    .. code-block:: python

        processed = processor.process_spectra(spectra)
    """

    def __init__(self, filters: Iterable[str | Callable | FunctionWithParametersType]):
        self.filters = []
        self.filter_order = [x.__name__ for x in ALL_FILTERS]

        for filter_description in filters:
            self.parse_and_add_filter(filter_description)

    def parse_and_add_filter(
        self,
        filter_description: str | Callable | FunctionWithParametersType,
        filter_position: int | None = None,
    ):
        """Add a filter by parsing the allowed filter description formats."""
        filter_args = None

        if isinstance(filter_description, (tuple, list)):
            if len(filter_description) == 1:
                filter_function = filter_description[0]
            elif len(filter_description) == 2:
                filter_function = filter_description[0]
                filter_args = filter_description[1]
            else:
                raise ValueError(
                    "The filter function description should contain at most two values: "
                    "the first should be a string or callable and the second a dictionary "
                    "with settings."
                )
        else:
            filter_function = filter_description

        if isinstance(filter_function, str):
            filter_function = load_matchms_filter_from_string(filter_function)

        self._add_filter_to_filter_order(
            filter_function.__name__,
            filter_position=filter_position,
        )
        self._store_filter(filter_function, filter_args)

    def _store_filter(self, new_filter_function: Callable, filter_params: dict[str, object] | None):
        """Store filter, replace duplicates, and sort filters."""
        if not callable(new_filter_function):
            raise TypeError("Expected callable filter function.")

        new_filter_function = create_partial_function(new_filter_function, filter_params)
        check_all_parameters_given(new_filter_function)

        self._replace_already_stored_filters(new_filter_function)
        self.filters.sort(key=lambda f: self.filter_order.index(f.__name__))

    def _replace_already_stored_filters(self, new_filter_function: Callable):
        """Replace filters that are already stored.

        If the same filter is added more than once, the last parameter settings
        are used.
        """
        filter_already_added = False

        for i, filter_function in enumerate(self.filters):
            if new_filter_function.__name__ == filter_function.__name__:
                logger.warning(
                    "The filter %s was already in the filter list. "
                    "The last added filter parameters are used.",
                    new_filter_function.__name__,
                )
                self.filters[i] = new_filter_function
                filter_already_added = True

        if not filter_already_added:
            self.filters.append(new_filter_function)

    def _add_filter_to_filter_order(self, filter_function_name: str, filter_position: int | None = None):
        """Add the filter name to the filter order list if it is not yet there."""
        if filter_function_name in self.filter_order:
            if filter_position is None:
                return None
            self.filter_order.remove(filter_function_name)

        if filter_position is None or filter_position >= len(self.filters):
            self.filter_order.append(filter_function_name)
        else:
            current_filter_at_position = self.filters[filter_position].__name__
            order_index = self.filter_order.index(current_filter_at_position)
            self.filter_order.insert(order_index, filter_function_name)

        return None

    def process_collection(self, collection: SpectraCollection) -> SpectraCollection | None:
        """Process a SpectraCollection with all filters in the pipeline.

        Parameters
        ----------
        collection
            SpectraCollection to process.

        Returns
        -------
        SpectraCollection or None
            The processed collection. If a filter removes all spectra and returns
            ``None``, processing stops and ``None`` is returned.
        """
        if not isinstance(collection, SpectraCollection):
            raise TypeError(
                "SpectraCollectionProcessor.process_collection expects a "
                "SpectraCollection."
            )

        if not self.filters:
            logger.warning("No filters have been specified, so the collection was not filtered.")

        processed_collection = collection.copy()

        for filter_func in self.filters:
            method_params = inspect.signature(filter_func).parameters
            kwargs = {"clone": False} if "clone" in method_params else {}

            processed_collection = filter_func(processed_collection, **kwargs)

            if processed_collection is None:
                return None

            if not isinstance(processed_collection, SpectraCollection):
                raise TypeError(
                    f"Filter {filter_func.__name__} returned "
                    f"{type(processed_collection).__name__}, expected "
                    "SpectraCollection or None."
                )

        return processed_collection

    def process_spectra(
        self,
        spectra,
        cleaned_spectra_file=None,
    ) -> SpectraCollection | None:
        """Process spectra as a SpectraCollection.

        Parameters
        ----------
        spectra
            Either a SpectraCollection or an iterable of Spectrum objects.
        cleaned_spectra_file
            Optional output path. The processed collection is materialized as
            Spectrum objects for saving.

        Returns
        -------
        SpectraCollection or None
            Processed collection.
        """
        if cleaned_spectra_file is not None and os.path.exists(cleaned_spectra_file):
            raise FileExistsError("The specified save references file already exists")

        if isinstance(spectra, SpectraCollection):
            collection = spectra
        else:
            collection = SpectraCollection(spectra)

        processed_collection = self.process_collection(collection)

        if cleaned_spectra_file is not None and processed_collection is not None:
            save_spectra(list(processed_collection), cleaned_spectra_file)

        return processed_collection

    @property
    def processing_steps(self):
        filter_list = []

        for filter_step in self.filters:
            parameter_settings = get_parameter_settings(filter_step)
            if parameter_settings is not None:
                filter_list.append((filter_step.__name__, parameter_settings))
            else:
                filter_list.append(filter_step.__name__)

        return filter_list

    def __str__(self):
        workflow = OrderedDict()
        workflow["Processing steps"] = self.processing_steps
        return ordered_dump(workflow)
