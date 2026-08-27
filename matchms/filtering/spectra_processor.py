from __future__ import annotations
import inspect
import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from matchms.filtering.filter_effects import (
    FILTER_EFFECTS,
    FRAGMENTS,
    METADATA,
    REMOVE,
)
from matchms.filtering.filter_order import ALL_FILTERS, FILTER_FUNCTION_NAMES
from matchms.spectra_collection import SpectraCollection
from matchms.spectrum import Spectrum
from matchms.yaml_file_functions import ordered_dump


logger = logging.getLogger("matchms")
FunctionWithParametersType = tuple[Callable | str, dict[str, object]]


@dataclass
class _ProcessingStepReport:
    """Aggregated report values for one filter step."""

    n_input: int = 0
    n_output: int = 0
    removed: int = 0
    metadata_changed: int | None = 0
    fragments_changed: int | None = 0


@dataclass(frozen=True)
class _HashSnapshot:
    """Minimal immutable state needed to report one processing step."""

    n_spectra: int
    metadata_hashes: np.ndarray | None = None
    fragment_hashes: np.ndarray | None = None


class ProcessingReport:
    """Summarize how much each filter changed during data processing.

    For each filter, it records how many spectra entered and left the step,
    how many spectra were removed, and how many retained spectra changed in
    metadata or fragments.

    For built-in matchms filters, :mod:`matchms.filtering.filter_effects`
    determines which hashes need to be compared. Unknown/custom filters are
    compared using both metadata and fragment hashes when row counts stay the
    same. If an unknown filter changes the number of rows, changes among the
    surviving rows cannot be aligned reliably and are reported as missing.
    """

    def __init__(self, filter_functions: Iterable[Callable] | None = None):
        self.filter_names = []
        self._steps: dict[str, _ProcessingStepReport] = {}
        self.counter_number_processed = 0

        if filter_functions is not None:
            for filter_function in filter_functions:
                self._ensure_filter(filter_function.__name__)

    def _ensure_filter(self, filter_name: str) -> _ProcessingStepReport:
        if filter_name not in self._steps:
            self.filter_names.append(filter_name)
            self._steps[filter_name] = _ProcessingStepReport()
        return self._steps[filter_name]

    def add_processed(self, n_spectra: int) -> None:
        """Add spectra entering a new processing pipeline run."""
        self.counter_number_processed += n_spectra

    def add_step(
        self,
        filter_name: str,
        effect: str | None,
        before: _HashSnapshot,
        after: _HashSnapshot,
    ) -> None:
        """Add one before/after filter comparison to the report.
        
        Parameters
        ----------
        filter_name
            Name of the filter step.
        effect
            Filter effect type, or None for unknown/custom filters.
            Can be one of ``REMOVE``, ``METADATA``, or ``FRAGMENTS``.
        before
            Hash snapshot of the spectra before the filter was applied.
        after
            Hash snapshot of the spectra after the filter was applied.
        """
        step = self._ensure_filter(filter_name)
        step.n_input += before.n_spectra
        step.n_output += after.n_spectra

        removed = max(0, before.n_spectra - after.n_spectra)
        step.removed += removed

        metadata_changed, fragments_changed = self._count_changes(
            effect=effect,
            before=before,
            after=after,
        )

        step.metadata_changed = _sum_optional_counts(
            step.metadata_changed,
            metadata_changed,
        )
        step.fragments_changed = _sum_optional_counts(
            step.fragments_changed,
            fragments_changed,
        )

    @staticmethod
    def _count_changes(
        effect: str | None,
        before: _HashSnapshot,
        after: _HashSnapshot,
    ) -> tuple[int | None, int | None]:
        """Return metadata and fragment change counts for one filter step."""
        if effect == REMOVE:
            return 0, 0

        same_length = before.n_spectra == after.n_spectra

        if effect == METADATA:
            if not same_length:
                return None, 0
            return _count_hash_differences(
                before.metadata_hashes,
                after.metadata_hashes,
            ), 0

        if effect == FRAGMENTS:
            if not same_length:
                return 0, None
            return 0, _count_hash_differences(
                before.fragment_hashes,
                after.fragment_hashes,
            )

        # Unknown/custom filter. 
        # If every row disappeared, there are no retained rows whose metadata/fragments need comparison.
        # Otherwise row alignment is only reliable when the number of rows stayed unchanged.
        if after.n_spectra == 0:
            return 0, 0
        if not same_length:
            return None, None

        return (
            _count_hash_differences(
                before.metadata_hashes,
                after.metadata_hashes,
            ),
            _count_hash_differences(
                before.fragment_hashes,
                after.fragment_hashes,
            ),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the processing report as a pandas DataFrame."""
        rows = []
        for filter_name in self.filter_names:
            step = self._steps[filter_name]
            rows.append(
                {
                    "filter": filter_name,
                    "input spectra": step.n_input,
                    "output spectra": step.n_output,
                    "removed spectra": step.removed,
                    "changed metadata": step.metadata_changed,
                    "changed fragments": step.fragments_changed,
                }
            )

        columns = [
            "input spectra",
            "output spectra",
            "removed spectra",
            "changed metadata",
            "changed fragments",
        ]

        if not rows:
            return pd.DataFrame(columns=columns).rename_axis("filter")

        report = pd.DataFrame(rows).set_index("filter")
        for column in columns:
            report[column] = pd.array(report[column], dtype="Int64")
        return report

    def __str__(self) -> str:
        return (
            "----- Spectra Processing Report -----\n"
            f"Number of spectra processed: {self.counter_number_processed}\n"
            f"Number of spectra removed: {sum(step.removed for step in self._steps.values())}\n"
            "Changes during processing:\n"
            f"{self.to_dataframe()}"
        )

    def __repr__(self) -> str:
        return (
            f"ProcessingReport(n_processed={self.counter_number_processed}, "
            f"n_steps={len(self._steps)})"
        )


class SpectraProcessor:
    """Process Spectrum or SpectraCollection objects with one filter pipeline.

    The processor stores one ordered list of filters and exposes two explicit
    execution paths:

    - :meth:`process_spectrum` applies the complete pipeline to one Spectrum.
    - :meth:`process_collection` applies each filter to the complete
      SpectraCollection, allowing collection-native/vectorized implementations.

    Both paths clone/copy the input once before processing. Filters that expose a
    ``clone`` parameter are then called with ``clone=False``.

    Parameters
    ----------
    filters
        Filter descriptions. Each item can be a filter name, a callable, or a
        ``(filter, parameter_dict)`` pair.
    """

    def __init__(self, filters: Iterable[str | Callable | FunctionWithParametersType]):
        self.filters: list[Callable] = []
        self.filter_order = [filter_func.__name__ for filter_func in ALL_FILTERS]

        for filter_description in filters:
            self.parse_and_add_filter(filter_description)

    def parse_and_add_filter(
        self,
        filter_description: str | Callable | FunctionWithParametersType,
        filter_position: int | None = None,
    ) -> None:
        """Parse and add a filter to the processing pipeline."""
        filter_args = None

        if isinstance(filter_description, (tuple, list)):
            if len(filter_description) == 1:
                filter_function = filter_description[0]
            elif len(filter_description) == 2:
                filter_function = filter_description[0]
                filter_args = filter_description[1]
            else:
                raise ValueError(
                    "The filter description should contain at most two values: "
                    "a filter name/callable and a dictionary with settings."
                )
        else:
            filter_function = filter_description

        if isinstance(filter_function, str):
            filter_function = load_matchms_filter_from_string(filter_function)

        if not callable(filter_function):
            raise TypeError("Expected callable filter function.")

        self._add_filter_to_filter_order(
            filter_function.__name__,
            filter_position=filter_position,
        )
        self._store_filter(filter_function, filter_args)

    def _store_filter(
        self,
        new_filter_function: Callable,
        filter_params: dict[str, object] | None,
    ) -> None:
        """Store a filter, replacing duplicates and preserving filter order."""
        new_filter_function = create_partial_function(
            new_filter_function,
            filter_params,
        )
        check_all_parameters_given(new_filter_function)
        self._replace_already_stored_filters(new_filter_function)
        self.filters.sort(key=lambda func: self.filter_order.index(func.__name__))

    def _replace_already_stored_filters(self, new_filter_function: Callable) -> None:
        """Replace an already configured filter with its newest settings."""
        for i, filter_function in enumerate(self.filters):
            if new_filter_function.__name__ == filter_function.__name__:
                logger.warning(
                    "The filter %s was already in the filter list. "
                    "The last added filter parameters are used.",
                    new_filter_function.__name__,
                )
                self.filters[i] = new_filter_function
                return

        self.filters.append(new_filter_function)

    def _add_filter_to_filter_order(
        self,
        filter_function_name: str,
        filter_position: int | None = None,
    ) -> None:
        """Add custom filters to the order or reposition an existing filter."""
        if filter_function_name in self.filter_order:
            if filter_position is None:
                return
            self.filter_order.remove(filter_function_name)

        if filter_position is None or filter_position >= len(self.filters):
            self.filter_order.append(filter_function_name)
        else:
            current_filter_at_position = self.filters[filter_position].__name__
            order_index = self.filter_order.index(current_filter_at_position)
            self.filter_order.insert(order_index, filter_function_name)

    def create_processing_report(self) -> ProcessingReport:
        """Return an empty ProcessingReport configured for this pipeline."""
        return ProcessingReport(self.filters)

    def process_spectrum(
        self,
        spectrum: Spectrum,
        processing_report: ProcessingReport | None = None,
    ) -> Spectrum | None:
        """Process one Spectrum through the complete filter pipeline.

        The input Spectrum is cloned once and is never modified in place.

        Parameters
        ----------
        spectrum
            Spectrum to process.
        processing_report
            Optional report to which this processing run is added.
            When set to None, no report is generated (can save computation time).

        Returns
        -------
        Spectrum or None
            Processed Spectrum, or None if a requirement filter removes it.
        """
        if not isinstance(spectrum, Spectrum):
            raise TypeError("SpectraProcessor.process_spectrum expects a Spectrum.")

        if not self.filters:
            logger.warning("No filters have been specified, so the spectrum was not filtered.")

        working_spectrum = spectrum.clone()

        if processing_report is not None:
            processing_report.add_processed(1)

        for filter_func in self.filters:
            effect = FILTER_EFFECTS.get(filter_func.__name__)
            before = (
                _take_hash_snapshot(working_spectrum, effect)
                if processing_report is not None
                else None
            )

            spectrum_out = _apply_filter(filter_func, working_spectrum)

            if spectrum_out is not None and not isinstance(spectrum_out, Spectrum):
                raise TypeError(
                    f"Filter {filter_func.__name__} returned "
                    f"{type(spectrum_out).__name__}, expected Spectrum or None."
                )

            if processing_report is not None:
                after = _take_hash_snapshot(spectrum_out, effect)
                processing_report.add_step(
                    filter_name=filter_func.__name__,
                    effect=effect,
                    before=before,
                    after=after,
                )

            if spectrum_out is None:
                return None

            working_spectrum = spectrum_out

        return working_spectrum

    def process_spectra(
        self,
        spectra: Iterable[Spectrum],
        processing_report: ProcessingReport | None = None,
        progress_bar: bool = True,
    ) -> list[Spectrum]:
        """Process an iterable of Spectrum objects spectrum by spectrum.

        Each spectrum is passed through the complete filter pipeline using
        :meth:`process_spectrum`. Spectra removed by requirement filters are omitted
        from the returned list.

        Parameters
        ----------
        spectra
            Iterable of Spectrum objects.
        processing_report
            Optional report to which all processing runs are added.
            When set to None, no report is generated (can save computation time).
        progress_bar
            If True, display a progress bar.

        Returns
        -------
        list[Spectrum]
            Processed spectra. Spectra removed by filters are not included.
        """
        processed_spectra = []

        for spectrum in tqdm(
            spectra,
            disable=not progress_bar,
            desc="Processing spectra",
        ):
            if spectrum is None:
                continue

            processed_spectrum = self.process_spectrum(
                spectrum,
                processing_report=processing_report,
            )

            if processed_spectrum is not None:
                processed_spectra.append(processed_spectrum)

        return processed_spectra

    def process_collection(
        self,
        collection: SpectraCollection,
        processing_report: ProcessingReport | None = None,
    ) -> SpectraCollection | None:
        """Process a SpectraCollection using collection-oriented filter execution.

        The input collection is copied once and is never modified in place. Each
        filter is applied to the full working collection so collection-native
        implementations can use vectorized metadata/fragment operations.

        Parameters
        ----------
        collection
            :class:`matchms.spectra_collection.SpectraCollection` to process.
        processing_report
            Optional report to which this processing run is added.
            When set to None, no report is generated (can save computation time).

        Returns
        -------
        SpectraCollection or None
            Processed collection, or None if a filter removes all spectra and
            returns None.
        """
        if not isinstance(collection, SpectraCollection):
            raise TypeError(
                "SpectraProcessor.process_collection expects a SpectraCollection."
            )

        if not self.filters:
            logger.warning("No filters have been specified, so the collection was not filtered.")

        working_collection = collection.copy()

        if processing_report is not None:
            processing_report.add_processed(len(working_collection))

        for filter_func in self.filters:
            effect = FILTER_EFFECTS.get(filter_func.__name__)
            before = (
                _take_hash_snapshot(working_collection, effect)
                if processing_report is not None
                else None
            )

            collection_out = _apply_filter(filter_func, working_collection)

            if collection_out is not None and not isinstance(
                collection_out,
                SpectraCollection,
            ):
                raise TypeError(
                    f"Filter {filter_func.__name__} returned "
                    f"{type(collection_out).__name__}, expected "
                    "SpectraCollection or None."
                )

            if processing_report is not None:
                after = _take_hash_snapshot(collection_out, effect)
                processing_report.add_step(
                    filter_name=filter_func.__name__,
                    effect=effect,
                    before=before,
                    after=after,
                )

            if collection_out is None:
                return None

            working_collection = collection_out

        return working_collection

    @property
    def processing_steps(self):
        """Return filter names and configured parameter settings."""
        filter_list = []

        for filter_step in self.filters:
            parameter_settings = get_parameter_settings(filter_step)
            if parameter_settings is not None:
                filter_list.append((filter_step.__name__, parameter_settings))
            else:
                filter_list.append(filter_step.__name__)

        return filter_list

    def __str__(self) -> str:
        workflow = OrderedDict()
        workflow["Processing steps"] = self.processing_steps
        return ordered_dump(workflow)


def _apply_filter(filter_func: Callable, spectra):
    """Apply one filter without requesting another clone when supported."""
    method_params = inspect.signature(filter_func).parameters
    kwargs = {"clone": False} if "clone" in method_params else {}
    return filter_func(spectra, **kwargs)


def _take_hash_snapshot(
    spectra: Spectrum | SpectraCollection | None,
    effect: str | None,
) -> _HashSnapshot:
    """Capture only hashes needed to report the given filter effect."""
    if spectra is None:
        return _HashSnapshot(n_spectra=0)

    needs_metadata = effect == METADATA or effect is None
    needs_fragments = effect == FRAGMENTS or effect is None

    if isinstance(spectra, Spectrum):
        metadata_hashes = (
            np.asarray([spectra.metadata_hash()], dtype=object)
            if needs_metadata
            else None
        )
        fragment_hashes = (
            np.asarray([spectra.spectrum_hash()], dtype=object)
            if needs_fragments
            else None
        )
        return _HashSnapshot(
            n_spectra=1,
            metadata_hashes=metadata_hashes,
            fragment_hashes=fragment_hashes,
        )

    if isinstance(spectra, SpectraCollection):
        metadata_hashes = (
            np.asarray(spectra.metadata_hashes).copy()
            if needs_metadata
            else None
        )
        fragment_hashes = (
            np.asarray(spectra.fragment_hashes).copy()
            if needs_fragments
            else None
        )
        return _HashSnapshot(
            n_spectra=len(spectra),
            metadata_hashes=metadata_hashes,
            fragment_hashes=fragment_hashes,
        )

    raise TypeError(
        "Hash snapshots require Spectrum, SpectraCollection, or None, "
        f"got {type(spectra).__name__}."
    )


def _count_hash_differences(
    before_hashes: np.ndarray | None,
    after_hashes: np.ndarray | None,
) -> int:
    """Count element-wise hash changes for aligned rows."""
    if before_hashes is None or after_hashes is None:
        raise ValueError("Required hashes were not captured for processing report.")
    if before_hashes.shape != after_hashes.shape:
        raise ValueError(
            "Cannot compare hash arrays with different shapes: "
            f"{before_hashes.shape} and {after_hashes.shape}."
        )
    return int(np.count_nonzero(before_hashes != after_hashes))


def _sum_optional_counts(
    current: int | None,
    addition: int | None,
) -> int | None:
    """Accumulate report counts while preserving unknown values."""
    if current is None or addition is None:
        return None
    return current + addition


def load_matchms_filter_from_string(filter_name: str) -> Callable:
    """Load a matchms filter function from its public name."""
    if not isinstance(filter_name, str):
        raise TypeError("Expected a string.")
    if filter_name not in FILTER_FUNCTION_NAMES:
        raise ValueError(
            f"Unknown filter type: {filter_name}. Expected a known matchms "
            "filter name or a callable."
        )
    return FILTER_FUNCTION_NAMES[filter_name]


def create_partial_function(
    filter_function: Callable,
    filter_params: dict[str, object] | None,
) -> Callable:
    """Apply configured keyword parameters to a filter function.
    
    Parameters
    ----------
    filter_function
        Filter function to which parameters are applied.
    filter_params
        Dictionary of keyword parameters to apply to the filter function.
    """
    if filter_params is None:
        return filter_function

    if not isinstance(filter_params, dict):
        raise TypeError(
            f"Expected a dictionary for filter parameters, got {filter_params}."
        )

    partial_filter_func = partial(filter_function, **filter_params)
    partial_filter_func.__name__ = filter_function.__name__
    return partial_filter_func


def check_all_parameters_given(func: Callable) -> None:
    """Assert that only the spectra input remains as a required parameter."""
    signature = inspect.signature(func)
    parameters_without_value = [
        parameter
        for parameter, value in signature.parameters.items()
        if value.default is inspect.Parameter.empty
    ]

    assert len(parameters_without_value) == 1, (
        f"More than one parameter of the function {func.__name__} is not "
        f"specified, the parameters not specified are {parameters_without_value}."
    )


def get_parameter_settings(func: Callable) -> dict[str, object] | None:
    """Return configured/default parameter values for a filter function."""
    signature = inspect.signature(func)
    parameter_settings = {
        parameter: value.default
        for parameter, value in signature.parameters.items()
        if value.default is not inspect.Parameter.empty
    }

    return parameter_settings or None
