from matchms import filtering as msfilters


# List all filters in a functionally working order
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
               msfilters.derive_smiles_from_inchi,
               msfilters.derive_inchi_from_smiles,
               msfilters.derive_inchikey_from_inchi,
               msfilters.clean_adduct,
               msfilters.add_parent_mass,
               msfilters.derive_smiles_from_pubchem_compound_name_search,
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
               msfilters.require_minimum_of_high_peaks,
               msfilters.add_fingerprint,
               msfilters.add_losses,
               ]
FILTER_FUNCTION_NAMES = {x.__name__: x for x in ALL_FILTERS}
MINIMAL_FILTERS = ["make_charge_int",
                   "interpret_pepmass",
                   "derive_ionmode",
                   "correct_charge",
                   ]
BASIC_FILTERS = MINIMAL_FILTERS \
                + ["add_compound_name",
                   "derive_adduct_from_name",
                   "derive_formula_from_name",
                   "clean_compound_name",
                   "add_precursor_mz",
                   ]
DEFAULT_FILTERS = BASIC_FILTERS \
                  + ["require_precursor_mz",
                     "add_parent_mass",
                     "harmonize_undefined_inchikey",
                     "harmonize_undefined_inchi",
                     "harmonize_undefined_smiles",
                     "repair_inchi_inchikey_smiles",
                     "normalize_intensities",
                     "add_retention_time",
                     ]
FULLY_ANNOTATED_PROCESSING = DEFAULT_FILTERS \
                             + ["clean_adduct",
                                "derive_inchi_from_smiles",
                                "derive_smiles_from_inchi",
                                "derive_inchikey_from_inchi",
                                ("require_correct_ionmode", {"ion_mode_to_keep": "both"}),
                                ("require_parent_mass_match_smiles", {'mass_tolerance': 0.1}),
                                ("repair_smiles_of_salts", {'mass_tolerance': 0.1}),
                                ("repair_precursor_is_parent_mass", {'mass_tolerance': 0.1}),
                                ("repair_parent_mass_is_mol_wt", {'mass_tolerance': 0.1}),
                                ("repair_adduct_based_on_smiles", {'mass_tolerance': 0.1}),
                                "repair_not_matching_annotation",
                                "require_valid_annotation",
                                ("derive_smiles_from_pubchem_compound_name_search", {"mass_tolerance": 0.1}),
                                ]


MS2DEEPSCORE_TRAINING = FULLY_ANNOTATED_PROCESSING + \
                        [("select_by_mz", {"mz_from": 0, "mz_to": 1000}),
                         ("select_by_relative_intensity", {"intensity_from": 0.001}),
                         ("reduce_to_number_of_peaks", {"n_max": 1000}),
                         ("require_minimum_of_high_peaks", {"no_peaks": 5, "intensity_percent": 2.0}),
                         ]

PREDEFINED_PIPELINES = {
    "minimal": MINIMAL_FILTERS,
    "basic": BASIC_FILTERS,
    "default": DEFAULT_FILTERS,
    "fully_annotated": FULLY_ANNOTATED_PROCESSING,
    "ms2deepscore": MS2DEEPSCORE_TRAINING,
}
