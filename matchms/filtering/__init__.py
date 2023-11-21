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

    # Users can pick a predefined pipeline ("basic", "default", "fully_annotated")
    # Or set to None if no predefined settings are desired.
    processing = SpectrumProcessor("basic")

    # Additional filters can be added as desired
    processing.add_matchms_filter("normalize_intensities")

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
from .default_filters import default_filters
from .metadata_processing.add_compound_name import add_compound_name
from .metadata_processing.add_fingerprint import add_fingerprint
from .metadata_processing.add_parent_mass import add_parent_mass
from .metadata_processing.add_precursor_mz import add_precursor_mz
from .metadata_processing.add_retention import (add_retention_index,
                                                add_retention_time)
from .metadata_processing.clean_adduct import clean_adduct
from .metadata_processing.clean_compound_name import clean_compound_name
from .metadata_processing.correct_charge import correct_charge
from .metadata_processing.derive_adduct_from_name import \
    derive_adduct_from_name
from .metadata_processing.derive_annotation_from_compound_name import \
    derive_annotation_from_compound_name
from .metadata_processing.derive_formula_from_name import \
    derive_formula_from_name
from .metadata_processing.derive_inchi_from_smiles import \
    derive_inchi_from_smiles
from .metadata_processing.derive_inchikey_from_inchi import \
    derive_inchikey_from_inchi
from .metadata_processing.derive_ionmode import derive_ionmode
from .metadata_processing.derive_smiles_from_inchi import \
    derive_smiles_from_inchi
from .metadata_processing.harmonize_undefined_inchi import \
    harmonize_undefined_inchi
from .metadata_processing.harmonize_undefined_inchikey import \
    harmonize_undefined_inchikey
from .metadata_processing.harmonize_undefined_smiles import \
    harmonize_undefined_smiles
from .metadata_processing.interpret_pepmass import interpret_pepmass
from .metadata_processing.make_charge_int import make_charge_int
from .metadata_processing.repair_adduct_based_on_smiles import \
    repair_adduct_based_on_smiles
from .metadata_processing.repair_inchi_inchikey_smiles import \
    repair_inchi_inchikey_smiles
from .metadata_processing.repair_not_matching_annotation import \
    repair_not_matching_annotation
from .metadata_processing.repair_parent_mass_is_mol_wt import \
    repair_parent_mass_is_mol_wt
from .metadata_processing.repair_parent_mass_match_smiles_wrapper import \
    repair_parent_mass_match_smiles_wrapper
from .metadata_processing.repair_precursor_is_parent_mass import \
    repair_precursor_is_parent_mass
from .metadata_processing.repair_smiles_of_salts import repair_smiles_of_salts
from .metadata_processing.require_correct_ionmode import \
    require_correct_ionmode
from .metadata_processing.require_parent_mass_match_smiles import \
    require_parent_mass_match_smiles
from .metadata_processing.require_precursor_below_mz import \
    require_precursor_below_mz
from .metadata_processing.require_precursor_mz import require_precursor_mz
from .metadata_processing.require_valid_annotation import \
    require_valid_annotation
from .peak_processing.add_losses import add_losses
from .peak_processing.normalize_intensities import normalize_intensities
from .peak_processing.reduce_to_number_of_peaks import \
    reduce_to_number_of_peaks
from .peak_processing.remove_peaks_around_precursor_mz import \
    remove_peaks_around_precursor_mz
from .peak_processing.remove_peaks_outside_top_k import \
    remove_peaks_outside_top_k
from .peak_processing.require_minimum_number_of_high_peaks import \
    require_minimum_number_of_high_peaks
from .peak_processing.require_minimum_number_of_peaks import \
    require_minimum_number_of_peaks
from .peak_processing.select_by_intensity import select_by_intensity
from .peak_processing.select_by_mz import select_by_mz
from .peak_processing.select_by_relative_intensity import \
    select_by_relative_intensity
from .SpeciesString import SpeciesString


__all__ = [
    "add_compound_name",
    "add_fingerprint",
    "add_losses",
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
    "repair_adduct_based_on_smiles",
    "require_correct_ionmode",
    "repair_inchi_inchikey_smiles",
    "repair_parent_mass_is_mol_wt",
    "repair_parent_mass_match_smiles_wrapper",
    "repair_precursor_is_parent_mass",
    "repair_smiles_of_salts",
    "derive_annotation_from_compound_name",
    "repair_not_matching_annotation",
    "require_valid_annotation",
    "require_parent_mass_match_smiles",
    "require_minimum_number_of_peaks",
    "require_minimum_number_of_high_peaks",
    "require_precursor_below_mz",
    "require_precursor_mz",
    "select_by_intensity",
    "select_by_mz",
    "select_by_relative_intensity",
    "SpeciesString",
]
