"""
Functions for processing mass spectra
#####################################

Provided functions will usually only perform a single action to a spectrum.
This can be changes or corrections of metadata, or peak filtering.
More complicated processing pipelines can be build by stacking several of
the provided filters.

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
from .add_compound_name import add_compound_name
from .add_fingerprint import add_fingerprint
from .add_losses import add_losses
from .add_parent_mass import add_parent_mass
from .add_precursor_mz import add_precursor_mz
from .add_retention import add_retention_index, add_retention_time
from .clean_compound_name import clean_compound_name
from .correct_charge import correct_charge
from .default_filters import default_filters
from .derive_adduct_from_name import derive_adduct_from_name
from .derive_formula_from_name import derive_formula_from_name
from .derive_inchi_from_smiles import derive_inchi_from_smiles
from .derive_inchikey_from_inchi import derive_inchikey_from_inchi
from .derive_ionmode import derive_ionmode
from .derive_smiles_from_inchi import derive_smiles_from_inchi
from .harmonize_undefined_inchi import harmonize_undefined_inchi
from .harmonize_undefined_inchikey import harmonize_undefined_inchikey
from .harmonize_undefined_smiles import harmonize_undefined_smiles
from .interpret_pepmass import interpret_pepmass
from .make_charge_int import make_charge_int
from .make_charge_scalar import make_charge_scalar
from .make_ionmode_lowercase import make_ionmode_lowercase
from .normalize_intensities import normalize_intensities
from .reduce_to_number_of_peaks import reduce_to_number_of_peaks
from .remove_peaks_around_precursor_mz import remove_peaks_around_precursor_mz
from .remove_peaks_outside_top_k import remove_peaks_outside_top_k
from .repair_adduct.clean_adduct import clean_adduct
from .repair_adduct.repair_adduct_based_on_smiles import \
    repair_adduct_based_on_smiles
from .repair_inchi_inchikey_smiles import repair_inchi_inchikey_smiles
from .repair_parent_mass_from_smiles.repair_parent_mass_is_mol_wt import \
    repair_parent_mass_is_mol_wt
from .repair_parent_mass_from_smiles.repair_parent_mass_match_smiles_wrapper import \
    repair_parent_mass_match_smiles_wrapper
from .repair_parent_mass_from_smiles.repair_precursor_is_parent_mass import \
    repair_precursor_is_parent_mass
from .repair_parent_mass_from_smiles.repair_smiles_of_salts import \
    repair_smiles_of_salts
from .repair_parent_mass_from_smiles.require_parent_mass_match_smiles import \
    require_parent_mass_match_smiles
from .repair_smiles_from_compound_name import repair_smiles_from_compound_name
from .require_correct_ionmode import require_correct_ionmode
from .require_minimum_number_of_peaks import require_minimum_number_of_peaks
from .require_minimum_of_high_peaks import require_minimum_of_high_peaks
from .require_precursor_below_mz import require_precursor_below_mz
from .require_precursor_mz import require_precursor_mz
from .require_valid_annotation import require_valid_annotation
from .select_by_intensity import select_by_intensity
from .select_by_mz import select_by_mz
from .select_by_relative_intensity import select_by_relative_intensity
from .set_ionmode_na_when_missing import set_ionmode_na_when_missing
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
    "make_charge_scalar",
    "make_ionmode_lowercase",
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
    "repair_smiles_from_compound_name",
    "require_valid_annotation",
    "require_parent_mass_match_smiles",
    "require_minimum_number_of_peaks",
    "require_minimum_of_high_peaks",
    "require_precursor_below_mz",
    "require_precursor_mz",
    "select_by_intensity",
    "select_by_mz",
    "select_by_relative_intensity",
    "set_ionmode_na_when_missing",
    "SpeciesString",
]
