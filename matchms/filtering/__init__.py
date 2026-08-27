"""
Processing and filtering mass spectra
######################################

Matchms provides filters for cleaning, harmonizing, and processing mass spectra.
Individual filters usually perform one specific operation, such as updating
metadata, removing spectra that do not meet a requirement, or modifying fragment
peaks.

Most matchms filters support both :class:`~matchms.Spectrum` and
:class:`~matchms.SpectraCollection` input. For processing complete datasets,
working with a ``SpectraCollection`` is generally preferred because filters can
operate directly on collection-level metadata and fragment data.

Loading a dataset
=================

Mass spectra can be loaded directly into a :class:`~matchms.SpectraCollection`
using :func:`~matchms.importing.load_ms2_dataset`.

For example, using the ``pesticides.mgf`` test dataset included with matchms:

.. testcode::

    from pathlib import Path

    from matchms.importing import load_ms2_dataset

    file_path = Path("tests/testdata/pesticides.mgf")
    collection = load_ms2_dataset(file_path)

    print(type(collection).__name__)
    print(collection.n_spectra > 0)

Should output

.. testoutput::

    SpectraCollection
    True


Processing with SpectraProcessor
================================

Because matchms contains many filters, and because some filters should be applied
in a particular order, the recommended way to define a processing workflow is
with :class:`~matchms.filtering.SpectraProcessor`.

The default filter order is defined in :mod:`~matchms.filtering.filter_order`. This
order is based on the recommended order of operations for processing mass spectra and
can be critical for the proper execution of some filters. For example, the ``require_precursor_mz`` filter
should be applied after the ``interpret_pepmass`` filter, because the latter adds precursor m/z values
to spectra that are missing them.

Also normalization of peaks can be an execution-order-sentitive operation. For example, if a filter
removes peaks, the normalization should usually be applied after that filter.

In case you want to apply a filter in a different order, you can either create your own ``SpectraProcessor``
with a custom filter order. Or simply apply every filter individually in the order you want, without using 
``SpectraProcessor``.

A processor is created from a list of filters. Filters can be specified by name,
or together with a dictionary containing non-default parameters.

.. testcode::

    from pathlib import Path

    from matchms import SpectraProcessor
    from matchms.importing import load_ms2_dataset

    file_path = Path("tests/testdata/pesticides.mgf")
    collection = load_ms2_dataset(file_path)

    processor = SpectraProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
            ("require_minimum_number_of_peaks", {"n_required": 5}),
        ]
    )

    processed_collection = processor.process_collection(collection)

    print(type(processed_collection).__name__)
    print(processed_collection.n_spectra <= collection.n_spectra)

Should output

.. testoutput::

    SpectraCollection
    True

``SpectraProcessor`` orders known matchms filters according to the recommended
matchms filter order. Custom filter functions can also be added to the same
pipeline.

The processor provides explicit methods for different processing modes:

- :meth:`~matchms.filtering.SpectraProcessor.process_spectrum` processes one
  :class:`~matchms.Spectrum`.
- :meth:`~matchms.filtering.SpectraProcessor.process_spectra` processes an
  iterable of spectra one spectrum at a time.
- :meth:`~matchms.filtering.SpectraProcessor.process_collection` processes a
  complete :class:`~matchms.SpectraCollection`, allowing collection-native
  implementations to be used.

For dataset-level workflows, ``process_collection`` is generally the preferred
option.


Processing reports
==================

``SpectraProcessor`` can optionally collect a processing report. The report gives
a compact overview of how much each processing step changed the dataset.

A report is created explicitly and passed to the processing method:

.. code-block:: python

    report = processor.create_processing_report()

    processed_collection = processor.process_collection(
        collection,
        processing_report=report,
    )

    report_df = report.to_dataframe()
    print(report_df)

For each filter, the report records:

- the number of input spectra,
- the number of output spectra,
- the number of removed spectra,
- the number of spectra with changed metadata,
- the number of spectra with changed fragments.

Metadata and fragment changes are detected using hashes. This means that
reporting does not require keeping a complete copy of the dataset before every
processing step.

Reporting is optional. If no report is needed, processing can simply be run as:

.. code-block:: python

    processed_collection = processor.process_collection(collection)


Running individual filters
==========================

Filters can also be applied directly without using ``SpectraProcessor``. This is
useful for simple operations or when developing a highly customized processing
workflow.

For example:

.. testcode::

    from pathlib import Path

    from matchms.filtering import select_by_relative_intensity
    from matchms.importing import load_ms2_dataset

    file_path = Path("tests/testdata/pesticides.mgf")
    collection = load_ms2_dataset(file_path)

    processed_collection = select_by_relative_intensity(
        collection,
        intensity_from=0.01,
    )

    print(type(processed_collection).__name__)

Should output

.. testoutput::

    SpectraCollection

For filters that remove spectra, behavior depends on the input type. A
:class:`~matchms.Spectrum` that does not meet the filter requirement returns
``None``. For :class:`~matchms.SpectraCollection` input, failing rows are removed
from both metadata and fragment data so that the collection remains synchronized
throughout processing.


.. figure:: ../_static/filtering_sketch.png
   :width: 700
   :alt: matchms filtering sketch

   Sketch of matchms spectrum processing.
"""


from matchms.filtering.default_filters import default_filters
from matchms.filtering.metadata_processing.add_compound_name import add_compound_name
from matchms.filtering.metadata_processing.add_parent_mass import add_parent_mass
from matchms.filtering.metadata_processing.add_precursor_formula import add_precursor_formula
from matchms.filtering.metadata_processing.add_precursor_mz import add_precursor_mz
from matchms.filtering.metadata_processing.add_retention import add_retention_index, add_retention_time
from matchms.filtering.metadata_processing.clean_adduct import clean_adduct
from matchms.filtering.metadata_processing.clean_compound_name import clean_compound_name
from matchms.filtering.metadata_processing.correct_charge import correct_charge
from matchms.filtering.metadata_processing.derive_adduct_from_name import derive_adduct_from_name
from matchms.filtering.metadata_processing.derive_annotation_from_compound_name import (
    derive_annotation_from_compound_name,
)
from matchms.filtering.metadata_processing.derive_formula_from_name import derive_formula_from_name
from matchms.filtering.metadata_processing.derive_formula_from_smiles import derive_formula_from_smiles
from matchms.filtering.metadata_processing.derive_inchi_from_smiles import derive_inchi_from_smiles
from matchms.filtering.metadata_processing.derive_inchikey_from_inchi import derive_inchikey_from_inchi
from matchms.filtering.metadata_processing.derive_ionmode import derive_ionmode
from matchms.filtering.metadata_processing.derive_smiles_from_inchi import derive_smiles_from_inchi
from matchms.filtering.metadata_processing.harmonize_missing_entries import harmonize_missing_entries
from matchms.filtering.metadata_processing.harmonize_undefined_inchi import harmonize_undefined_inchi
from matchms.filtering.metadata_processing.harmonize_undefined_inchikey import harmonize_undefined_inchikey
from matchms.filtering.metadata_processing.harmonize_undefined_smiles import harmonize_undefined_smiles
from matchms.filtering.metadata_processing.interpret_pepmass import interpret_pepmass
from matchms.filtering.metadata_processing.make_charge_int import make_charge_int
from matchms.filtering.metadata_processing.repair_adduct_and_parent_mass_based_on_smiles import (
    repair_adduct_and_parent_mass_based_on_smiles,
)
from matchms.filtering.metadata_processing.repair_adduct_based_on_parent_mass import repair_adduct_based_on_parent_mass
from matchms.filtering.metadata_processing.repair_inchi_inchikey_smiles import repair_inchi_inchikey_smiles
from matchms.filtering.metadata_processing.repair_not_matching_annotation import repair_not_matching_annotation
from matchms.filtering.metadata_processing.repair_parent_mass_from_smiles import repair_parent_mass_from_smiles
from matchms.filtering.metadata_processing.repair_parent_mass_is_molar_mass import repair_parent_mass_is_molar_mass
from matchms.filtering.metadata_processing.repair_parent_mass_match_smiles_wrapper import (
    repair_parent_mass_match_smiles_wrapper,
)
from matchms.filtering.metadata_processing.repair_smiles_of_salts import repair_smiles_of_salts
from matchms.filtering.metadata_processing.require_compound_name import require_compound_name
from matchms.filtering.metadata_processing.require_correct_ionmode import require_correct_ionmode
from matchms.filtering.metadata_processing.require_correct_ms_level import require_correct_ms_level
from matchms.filtering.metadata_processing.require_formula import require_formula
from matchms.filtering.metadata_processing.require_matching_adduct_and_ionmode import (
    require_matching_adduct_and_ionmode,
)
from matchms.filtering.metadata_processing.require_matching_adduct_precursor_mz_parent_mass import (
    require_matching_adduct_precursor_mz_parent_mass,
)
from matchms.filtering.metadata_processing.require_parent_mass_match_smiles import require_parent_mass_match_smiles
from matchms.filtering.metadata_processing.require_precursor_mz import require_precursor_mz
from matchms.filtering.metadata_processing.require_retention_index import require_retention_index
from matchms.filtering.metadata_processing.require_retention_time import require_retention_time
from matchms.filtering.metadata_processing.require_valid_annotation import require_valid_annotation
from matchms.filtering.peak_processing.normalize_intensities import normalize_intensities
from matchms.filtering.peak_processing.reduce_to_number_of_peaks import reduce_to_number_of_peaks
from matchms.filtering.peak_processing.remove_noise_below_frequent_intensities import (
    remove_noise_below_frequent_intensities,
)
from matchms.filtering.peak_processing.remove_peaks_around_precursor_mz import remove_peaks_around_precursor_mz
from matchms.filtering.peak_processing.remove_peaks_outside_top_k import remove_peaks_outside_top_k
from matchms.filtering.peak_processing.remove_peaks_relative_to_precursor_mz import (
    remove_peaks_relative_to_precursor_mz,
)
from matchms.filtering.peak_processing.remove_profiled_spectra import remove_profiled_spectra
from matchms.filtering.peak_processing.require_maximum_number_of_peaks import require_maximum_number_of_peaks
from matchms.filtering.peak_processing.require_minimum_number_of_high_peaks import require_minimum_number_of_high_peaks
from matchms.filtering.peak_processing.require_minimum_number_of_peaks import require_minimum_number_of_peaks
from matchms.filtering.peak_processing.select_by_intensity import select_by_intensity
from matchms.filtering.peak_processing.select_by_mz import select_by_mz
from matchms.filtering.peak_processing.select_by_relative_intensity import select_by_relative_intensity
from matchms.filtering.species_string import SpeciesString


__all__ = [
    "SpeciesString",
    "add_compound_name",
    "add_parent_mass",
    "add_precursor_formula",
    "add_precursor_mz",
    "add_retention_index",
    "add_retention_time",
    "clean_adduct",
    "clean_compound_name",
    "correct_charge",
    "default_filters",
    "derive_adduct_from_name",
    "derive_annotation_from_compound_name",
    "derive_formula_from_name",
    "derive_formula_from_smiles",
    "derive_inchi_from_smiles",
    "derive_inchikey_from_inchi",
    "derive_ionmode",
    "derive_smiles_from_inchi",
    "harmonize_missing_entries",
    "harmonize_undefined_inchi",
    "harmonize_undefined_inchikey",
    "harmonize_undefined_smiles",
    "interpret_pepmass",
    "make_charge_int",
    "normalize_intensities",
    "reduce_to_number_of_peaks",
    "remove_noise_below_frequent_intensities",
    "remove_peaks_around_precursor_mz",
    "remove_peaks_outside_top_k",
    "remove_peaks_relative_to_precursor_mz",
    "remove_profiled_spectra",
    "repair_adduct_and_parent_mass_based_on_smiles",
    "repair_adduct_based_on_parent_mass",
    "repair_inchi_inchikey_smiles",
    "repair_not_matching_annotation",
    "repair_parent_mass_from_smiles",
    "repair_parent_mass_is_molar_mass",
    "repair_parent_mass_match_smiles_wrapper",
    "repair_smiles_of_salts",
    "require_compound_name",
    "require_correct_ionmode",
    "require_correct_ms_level",
    "require_formula",
    "require_matching_adduct_and_ionmode",
    "require_matching_adduct_precursor_mz_parent_mass",
    "require_maximum_number_of_peaks",
    "require_minimum_number_of_high_peaks",
    "require_minimum_number_of_peaks",
    "require_parent_mass_match_smiles",
    "require_precursor_mz",
    "require_retention_index",
    "require_retention_time",
    "require_valid_annotation",
    "select_by_intensity",
    "select_by_mz",
    "select_by_relative_intensity",
]
