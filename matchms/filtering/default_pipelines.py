"""Stores sets of filters for easy pipeline creation."""
from matchms import filtering as msfilters


HARMONIZE_METADATA_FIELD_NAMES = \
    [msfilters.make_charge_int,
     msfilters.add_compound_name,
     msfilters.interpret_pepmass,
     msfilters.add_precursor_mz,
     msfilters.add_retention_time,
     msfilters.add_retention_index,
     ]
DERIVE_METADATA_IN_WRONG_FIELD = \
    [msfilters.derive_adduct_from_name,
     msfilters.derive_formula_from_name,
     msfilters.clean_compound_name,
     msfilters.derive_ionmode,
     msfilters.repair_inchi_inchikey_smiles,
     ]
HARMONIZE_METADATA_ENTRIES = \
    [msfilters.harmonize_undefined_inchikey,
     msfilters.harmonize_undefined_inchi,
     msfilters.harmonize_undefined_smiles,
     msfilters.clean_adduct,
     ]
DERIVE_MISSING_METADATA = [msfilters.correct_charge,
                           msfilters.add_parent_mass,
                           msfilters.derive_inchi_from_smiles,
                           msfilters.derive_smiles_from_inchi,
                           msfilters.derive_inchikey_from_inchi,
                           ]
REQUIRE_COMPLETE_METADATA = [msfilters.require_precursor_mz,
                             (msfilters.require_correct_ionmode, {"ion_mode_to_keep": "both"}), ]
REPAIR_ANNOTATION = [
    (msfilters.repair_smiles_of_salts, {'mass_tolerance': 0.1}),
    (msfilters.repair_precursor_is_parent_mass, {'mass_tolerance': 0.1}),
    (msfilters.repair_parent_mass_is_mol_wt, {'mass_tolerance': 0.1}),
    (msfilters.repair_adduct_based_on_smiles, {'mass_tolerance': 0.1}),
    msfilters.repair_not_matching_annotation,
    (msfilters.derive_annotation_from_compound_name, {"mass_tolerance": 0.1}),
]
REQUIRE_COMPLETE_ANNOTATION = [(msfilters.require_parent_mass_match_smiles, {'mass_tolerance': 0.1}),
                               msfilters.require_valid_annotation, ]
CLEAN_PEAKS = [(msfilters.select_by_mz, {"mz_from": 0, "mz_to": 1000}),
               (msfilters.select_by_relative_intensity, {"intensity_from": 0.001}),
               (msfilters.reduce_to_number_of_peaks, {"n_max": 1000}),
               (msfilters.require_minimum_number_of_high_peaks, {"no_peaks": 5, "intensity_percent": 2.0}),
               ]
# These filters are in None of the above pipelines
OTHER_FILTERS = [msfilters.require_precursor_below_mz,
                 msfilters.select_by_intensity,
                 msfilters.remove_peaks_around_precursor_mz,
                 msfilters.remove_peaks_outside_top_k,
                 msfilters.require_minimum_number_of_peaks,
                 msfilters.add_fingerprint,
                 msfilters.add_losses,
                 msfilters.repair_parent_mass_match_smiles_wrapper,]

BASIC_FILTERS = HARMONIZE_METADATA_FIELD_NAMES + DERIVE_METADATA_IN_WRONG_FIELD + HARMONIZE_METADATA_ENTRIES
DEFAULT_FILTERS = BASIC_FILTERS + [msfilters.normalize_intensities, ] + REQUIRE_COMPLETE_METADATA + DERIVE_MISSING_METADATA
LIBRARY_CLEANING = DEFAULT_FILTERS + REPAIR_ANNOTATION + REQUIRE_COMPLETE_ANNOTATION
MS2DEEPSCORE_TRAINING = LIBRARY_CLEANING + CLEAN_PEAKS


ALL_FILTER_SETS = [filter_set for filter_name, filter_set in locals().items()
                   if not filter_name.startswith("_") and filter_name != "ALL_FILTER_SETS" and isinstance(filter_set, list)]
