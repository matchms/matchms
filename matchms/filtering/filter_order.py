from matchms import filtering as msfilters


# List all filters in a functionally working order

# IMPORTANT!! IF YOU CHANGE ANYTHING HERE PLEASE ADD A TEST to test_all_filter_order.py
# to ensure it is not changed back by accident later.
ALL_FILTERS = [msfilters.make_charge_int,
               msfilters.add_compound_name,
               msfilters.derive_adduct_from_name,
               msfilters.derive_formula_from_name,
               msfilters.clean_compound_name,
               msfilters.interpret_pepmass,
               msfilters.add_precursor_mz,
               msfilters.add_retention_index,
               msfilters.add_retention_time,
               msfilters.derive_ionmode,
               msfilters.correct_charge,
               msfilters.require_precursor_mz,
               msfilters.harmonize_undefined_inchikey,
               msfilters.harmonize_undefined_inchi,
               msfilters.harmonize_undefined_smiles,
               msfilters.repair_inchi_inchikey_smiles,
               msfilters.clean_adduct,
               msfilters.add_parent_mass,
               msfilters.derive_annotation_from_compound_name,
               msfilters.derive_smiles_from_inchi,
               msfilters.derive_inchi_from_smiles,
               msfilters.derive_inchikey_from_inchi,
               msfilters.repair_smiles_of_salts,
               msfilters.repair_precursor_is_parent_mass,
               msfilters.repair_parent_mass_is_mol_wt,
               msfilters.repair_adduct_based_on_smiles,
               msfilters.repair_parent_mass_match_smiles_wrapper,
               msfilters.repair_not_matching_annotation,
               msfilters.require_valid_annotation,
               msfilters.require_correct_ionmode,
               msfilters.require_precursor_below_mz,
               msfilters.require_parent_mass_match_smiles,
               msfilters.normalize_intensities,
               msfilters.select_by_intensity,
               msfilters.select_by_mz,
               msfilters.select_by_relative_intensity,
               msfilters.remove_peaks_around_precursor_mz,
               msfilters.remove_peaks_outside_top_k,
               msfilters.reduce_to_number_of_peaks,
               msfilters.require_minimum_number_of_peaks,
               msfilters.require_minimum_number_of_high_peaks,
               msfilters.add_fingerprint,
               msfilters.add_losses,
               ]

FILTER_FUNCTION_NAMES = {x.__name__: x for x in ALL_FILTERS}
