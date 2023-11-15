

HARMONIZE_METADATA_FIELD_NAMES = \
    ["make_charge_int",
     "add_compound_name",
     "interpret_pepmass",
     "add_precursor_mz",
     "add_retention_time",
     "add_retention_index",
     ]
DERIVE_METADATA_IN_WRONG_FIELD = \
    ["derive_adduct_from_name",
     "derive_formula_from_name",
     "clean_compound_name",
     "derive_ionmode",
     "repair_inchi_inchikey_smiles"
     ]
HARMONIZE_METADATA_ENTRIES = \
    ["harmonize_undefined_inchikey",
     "harmonize_undefined_inchi",
     "harmonize_undefined_smiles",
     "clean_adduct",
     ]
DERIVE_MISSING_METADATA = ["correct_charge", "add_parent_mass", ]
REQUIRE_COMPLETE_METADATA = ["require_precursor_mz",
                             ("require_correct_ionmode", {"ion_mode_to_keep": "both"}), ]
REPAIR_ANNOTATION = [
    "derive_inchi_from_smiles",
    "derive_smiles_from_inchi",
    "derive_inchikey_from_inchi",
    ("repair_smiles_of_salts", {'mass_tolerance': 0.1}),
    ("repair_precursor_is_parent_mass", {'mass_tolerance': 0.1}),
    ("repair_parent_mass_is_mol_wt", {'mass_tolerance': 0.1}),
    ("repair_adduct_based_on_smiles", {'mass_tolerance': 0.1}),
    "repair_not_matching_annotation",
    ("derive_annotation_from_compound_name", {"mass_tolerance": 0.1}),
]
REQUIRE_COMPLETE_ANNOTATION = [("require_parent_mass_match_smiles", {'mass_tolerance': 0.1}),
                               "require_valid_annotation", ]
CLEAN_PEAKS = [("select_by_mz", {"mz_from": 0, "mz_to": 1000}),
               ("select_by_relative_intensity", {"intensity_from": 0.001}),
               ("reduce_to_number_of_peaks", {"n_max": 1000}),
               ("require_minimum_of_high_peaks", {"no_peaks": 5, "intensity_percent": 2.0}),
               ]
# These filters are in None of the above pipelines
OTHER_FILTERS = ["require_precursor_below_mz",
                 "select_by_intensity",
                 "remove_peaks_around_precursor_mz",
                 "remove_peaks_outside_top_k",
                 "require_minimum_number_of_peaks",
                 "add_fingerprint",
                 "add_losses",
                 "repair_parent_mass_match_smiles_wrapper"]

BASIC_FILTERS = HARMONIZE_METADATA_FIELD_NAMES + DERIVE_METADATA_IN_WRONG_FIELD + HARMONIZE_METADATA_ENTRIES
DEFAULT_FILTERS = BASIC_FILTERS + ["normalize_intensities", ] + REQUIRE_COMPLETE_METADATA + DERIVE_MISSING_METADATA
FULLY_ANNOTATED_PROCESSING = DEFAULT_FILTERS + REPAIR_ANNOTATION + REQUIRE_COMPLETE_ANNOTATION
MS2DEEPSCORE_TRAINING = FULLY_ANNOTATED_PROCESSING + CLEAN_PEAKS

PREDEFINED_PIPELINES = {
    "basic": BASIC_FILTERS,
    "default": DEFAULT_FILTERS,
    "fully_annotated": FULLY_ANNOTATED_PROCESSING,
    "ms2deepscore": MS2DEEPSCORE_TRAINING,
}
