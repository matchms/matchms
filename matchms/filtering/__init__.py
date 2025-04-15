"""
Processing (or: filtering) mass spectra
#######################################

Provided functions will usually only perform a single action to a spectrum.
This can be changes or corrections of metadata, or peak filtering.
More complicated processing pipelines can be build by stacking several of
the provided filters.

Because there are numerous filter functions in matchms and because they often
need to be applied in a specific order, the most feasible workflow for users
is to use the `SpectrumProcessor` class to define a spetrum processing pipeline.
Here is an example:

.. testcode::

    import numpy as np
    from matchms import Spectrum
    from matchms import SpectrumProcessor

    spectrum = Spectrum(mz=np.array([100, 120, 150, 200.]),
                        intensities=np.array([200.0, 300.0, 50.0, 1.0]),
                        metadata={'id': 'spectrum1'})

    # Users can pick a predefined pipeline from default pipelines, or specify a list of filters
    processing = SpectrumProcessor(["normalize_intensities"])

    # Run the processing pipeline:
    spectrum_filtered = processing.process_spectrum(spectrum)
    max_intensity = spectrum_filtered.peaks.intensities.max()
    print(f"Maximum intensity is {max_intensity:.2f}")

Should output

.. testoutput::

    Maximum intensity is 1.00

It is also possible to run each filter function individually. This for instance
makes sense if users want to develop a highly customized spectrum processing
routine.
Example of how to use a single filter function:

.. testcode::

    import numpy as np
    from matchms import Spectrum
    from matchms.filtering import normalize_intensities

    spectrum = Spectrum(mz=np.array([100, 120, 150, 200.]),
                        intensities=np.array([200.0, 300.0, 50.0, 1.0]),
                        metadata={'id': 'spectrum1'})
    spectrum_filtered = normalize_intensities(spectrum)

    max_intensity = spectrum_filtered.peaks.intensities.max()
    print(f"Maximum intensity is {max_intensity:.2f}")

Should output

.. testoutput::

    Maximum intensity is 1.00

.. figure:: ../_static/filtering_sketch.png
   :width: 700
   :alt: matchms filtering sketch

   Sketch of matchms spectrum processing.

"""

from matchms.filtering.default_filters import default_filters
from matchms.filtering.metadata_processing.add_compound_name import add_compound_name
from matchms.filtering.metadata_processing.add_fingerprint import add_fingerprint
from matchms.filtering.metadata_processing.add_parent_mass import add_parent_mass
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
from matchms.filtering.metadata_processing.require_precursor_mz import require_precursor_below_mz, require_precursor_mz
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
from matchms.filtering.peak_processing.remove_profiled_spectra import remove_profiled_spectra
from matchms.filtering.peak_processing.require_maximum_number_of_peaks import require_maximum_number_of_peaks
from matchms.filtering.peak_processing.require_minimum_number_of_high_peaks import require_minimum_number_of_high_peaks
from matchms.filtering.peak_processing.require_minimum_number_of_peaks import require_minimum_number_of_peaks
from matchms.filtering.peak_processing.select_by_intensity import select_by_intensity
from matchms.filtering.peak_processing.select_by_mz import select_by_mz
from matchms.filtering.peak_processing.select_by_relative_intensity import select_by_relative_intensity
from matchms.filtering.SpeciesString import SpeciesString


__all__ = [
    "add_compound_name",
    "add_fingerprint",
    "add_parent_mass",
    "add_precursor_mz",
    "add_retention_index",
    "add_retention_time",
    "clean_compound_name",
    "clean_adduct",
    "correct_charge",
    "default_filters",
    "derive_adduct_from_name",
    "derive_formula_from_name",
    "derive_formula_from_smiles",
    "derive_inchi_from_smiles",
    "derive_inchikey_from_inchi",
    "derive_ionmode",
    "derive_smiles_from_inchi",
    "harmonize_undefined_inchi",
    "harmonize_undefined_inchikey",
    "harmonize_undefined_smiles",
    "interpret_pepmass",
    "make_charge_int",
    "normalize_intensities",
    "reduce_to_number_of_peaks",
    "remove_peaks_around_precursor_mz",
    "remove_peaks_outside_top_k",
    "remove_profiled_spectra",
    "repair_adduct_and_parent_mass_based_on_smiles",
    "repair_adduct_based_on_parent_mass",
    "require_correct_ionmode",
    "remove_noise_below_frequent_intensities",
    "repair_inchi_inchikey_smiles",
    "repair_parent_mass_from_smiles",
    "repair_parent_mass_is_molar_mass",
    "repair_parent_mass_match_smiles_wrapper",
    "repair_smiles_of_salts",
    "derive_annotation_from_compound_name",
    "repair_not_matching_annotation",
    "require_valid_annotation",
    "require_matching_adduct_precursor_mz_parent_mass",
    "require_parent_mass_match_smiles",
    "require_matching_adduct_and_ionmode",
    "require_minimum_number_of_peaks",
    "require_minimum_number_of_high_peaks",
    "require_maximum_number_of_peaks",
    "require_precursor_below_mz",
    "require_precursor_mz",
    "require_compound_name",
    "require_correct_ms_level",
    "require_formula",
    "require_retention_time",
    "require_retention_index",
    "select_by_intensity",
    "select_by_mz",
    "select_by_relative_intensity",
    "SpeciesString",
]
