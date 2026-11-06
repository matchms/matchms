"""Effects of matchms filters used for processing reports."""

METADATA = "metadata"
FRAGMENTS = "fragments"
REMOVE = "remove"


FILTER_EFFECTS = {
    "add_compound_name": METADATA,
    "add_parent_mass": METADATA,
    "add_precursor_formula": METADATA,
    "add_precursor_mz": METADATA,
    "add_retention_index": METADATA,
    "add_retention_time": METADATA,
    "clean_adduct": METADATA,
    "clean_compound_name": METADATA,
    "correct_charge": METADATA,
    "default_filters": METADATA,
    "derive_adduct_from_name": METADATA,
    "derive_annotation_from_compound_name": METADATA,
    "derive_formula_from_name": METADATA,
    "derive_formula_from_smiles": METADATA,
    "derive_inchi_from_smiles": METADATA,
    "derive_inchikey_from_inchi": METADATA,
    "derive_ionmode": METADATA,
    "derive_smiles_from_inchi": METADATA,
    "harmonize_missing_entries": METADATA,
    "harmonize_undefined_inchi": METADATA,
    "harmonize_undefined_inchikey": METADATA,
    "harmonize_undefined_smiles": METADATA,
    "interpret_pepmass": METADATA,
    "make_charge_int": METADATA,

    # FRAGMENTS
    "normalize_intensities": FRAGMENTS,
    "reduce_to_number_of_peaks": FRAGMENTS,
    "remove_noise_below_frequent_intensities": FRAGMENTS,
    "remove_peaks_around_precursor_mz": FRAGMENTS,
    "remove_peaks_outside_top_k": FRAGMENTS,
    "remove_peaks_relative_to_precursor_mz": FRAGMENTS,

    # REMOVE
    "remove_profiled_spectra": REMOVE,

    # REPAIR METADATA
    "repair_adduct_and_parent_mass_based_on_smiles": METADATA,
    "repair_adduct_based_on_parent_mass": METADATA,
    "repair_inchi_inchikey_smiles": METADATA,
    "repair_not_matching_annotation": METADATA,
    "repair_parent_mass_from_smiles": METADATA,
    "repair_parent_mass_is_molar_mass": METADATA,
    "repair_parent_mass_match_smiles_wrapper": METADATA,
    "repair_smiles_of_salts": METADATA,

    # REMOVE
    "require_compound_name": REMOVE,
    "require_correct_ionmode": REMOVE,
    "require_correct_ms_level": REMOVE,
    "require_formula": REMOVE,
    "require_matching_adduct_and_ionmode": REMOVE,
    "require_matching_adduct_precursor_mz_parent_mass": REMOVE,
    "require_maximum_number_of_peaks": REMOVE,
    "require_minimum_number_of_high_peaks": REMOVE,
    "require_minimum_number_of_peaks": REMOVE,
    "require_parent_mass_match_smiles": REMOVE,
    "require_precursor_mz": REMOVE,
    "require_retention_index": REMOVE,
    "require_retention_time": REMOVE,
    "require_valid_annotation": REMOVE,

    # FRAGMENTS
    "select_by_intensity": FRAGMENTS,
    "select_by_mz": FRAGMENTS,
    "select_by_relative_intensity": FRAGMENTS,
}