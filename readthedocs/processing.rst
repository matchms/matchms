Filtering and processing
========================

Many matchms filters support both ``Spectrum`` and ``SpectraCollection`` input.
When a filter receives a collection, it operates on all rows while preserving
the alignment between metadata and fragment data.

.. code-block:: python

    from matchms.filtering import (
        harmonize_missing_entries,
        select_by_relative_intensity,
        require_minimum_number_of_peaks,
    )

    collection = harmonize_missing_entries(collection)
    collection = select_by_relative_intensity(collection, intensity_from=0.01)
    collection = require_minimum_number_of_peaks(collection, n_required=5)

For filters that remove spectra, the behavior depends on the input type:

- For ``Spectrum`` input, a failing spectrum returns ``None``.
- For ``SpectraCollection`` input, failing rows are removed from both metadata
  and fragments.

This keeps the collection synchronized throughout the workflow.

SpectraProcessor
----------------

``SpectraProcessor`` provides a single processing pipeline for both individual
``Spectrum`` objects and complete ``SpectraCollection`` objects.

The processor uses the same ordered list of filters for both execution paths,
but provides explicit methods for choosing how the pipeline is applied:

- ``process_spectrum()`` applies all filters sequentially to one ``Spectrum``.
- ``process_collection()`` applies each filter to the complete
  ``SpectraCollection``, allowing collection-native and vectorized filter
  implementations to be used.

For collection-based workflows:

.. code-block:: python

    from matchms import SpectraProcessor
    from matchms.importing import load_ms2_dataset

    collection = load_ms2_dataset("my_spectra.mgf")

    processor = SpectraProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
            ("require_minimum_number_of_peaks", {"n_required": 5}),
        ]
    )

    processed_collection = processor.process_collection(collection)

For individual spectra, the same processor and filter configuration can be
used:

.. code-block:: python

    from matchms import SpectraProcessor
    from matchms.filtering.default_pipelines import BASIC_FILTERS
    from matchms.importing import load_spectra

    spectra = list(load_spectra("my_spectra.mgf"))
    processor = SpectraProcessor(BASIC_FILTERS)

    processed_spectra = [
        processed
        for spectrum in spectra
        if (processed := processor.process_spectrum(spectrum)) is not None
    ]

Both processing methods copy the input once before applying the pipeline, so the
original ``Spectrum`` or ``SpectraCollection`` is not modified.

``SpectraProcessor`` accepts filter descriptions in several forms:

.. code-block:: python

    filters = [
        "harmonize_missing_entries",
        ("select_by_intensity", {"intensity_from": 10.0, "intensity_to": 1000.0}),
        custom_filter_function,
        (custom_filter_with_parameters, {"parameter": "value"}),
    ]

Known matchms filters are ordered according to the matchms filter order. Custom
filters are appended unless a specific position is supplied.

Processing reports
------------------

``SpectraProcessor`` can optionally collect a processing report that summarizes
how much each filter changed the data.

.. code-block:: python

    processor = SpectraProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
            ("require_minimum_number_of_peaks", {"n_required": 5}),
        ]
    )

    report = processor.create_processing_report()

    processed_collection = processor.process_collection(
        collection,
        processing_report=report,
    )

    print(report.to_dataframe())

The report records, for each filter:

- the number of input spectra,
- the number of output spectra,
- the number of removed spectra,
- the number of spectra with changed metadata,
- the number of spectra with changed fragments.

Changes are detected using metadata and fragment hashes, so reporting does not
require keeping full copies of the data before every processing step.

The same reporting mechanism can also be used with ``process_spectrum()``:

.. code-block:: python

    report = processor.create_processing_report()

    processed_spectrum = processor.process_spectrum(
        spectrum,
        processing_report=report,
    )
