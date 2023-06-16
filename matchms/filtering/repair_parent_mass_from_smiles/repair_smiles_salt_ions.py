import itertools
import logging
from matchms.filtering.repair_parent_mass_from_smiles.repair_precursor_is_parent_mass import _get_monoisotopic_neutral_mass

logger = logging.getLogger("matchms")


def repair_smiles_salt_ions(spectrum_in,
                            mass_tolerance):
    """Checks if parent mass matches one of the ions"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()

    smiles = spectrum.get("smiles")
    parent_mass = spectrum.get("parent_mass")
    possible_ion_combinations = create_possible_ions(smiles)

    for ion, not_used_ions in possible_ion_combinations:
        ion_mass = _get_monoisotopic_neutral_mass(ion)
        mass_diff = abs(parent_mass - ion_mass)
        # Check for Repair parent mass is mol wt did only return 1 spectrum. So not added as option for simplicity.
        if mass_diff < mass_tolerance:
            spectrum_with_ions = spectrum.clone()
            spectrum_with_ions.set("smiles", ion)
            spectrum_with_ions.set("salt_ions", not_used_ions)
            logger.info(f"Removed salt ions: {not_used_ions} from {smiles} to match parent mass")
            return spectrum_with_ions
    return spectrum


def create_possible_ions(smiles):
    """Selects all possible ion combinations of a salt"""
    results = []
    if "." in smiles:
        single_ions = smiles.split(".")
        for r in range(1, len(single_ions) + 1):
            combinations = itertools.combinations(single_ions, r)
            for combination in combinations:
                combined_ion = ".".join(combination)
                removed_ions = single_ions.copy()
                for used_ion in combination:
                    removed_ions.remove(used_ion)
                results.append((combined_ion, removed_ions))
    return results

if __name__ == "__main__":
    import os

    from matchms import set_matchms_logger_level
    import os
    import pickle
    from tqdm import tqdm
    from matchms.filtering.repair_parent_mass_from_smiles.require_parent_mass_match_smiles import require_parent_mass_match_smiles

    def load_pickled_file(filename: str):
        with open(filename, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object


    def save_pickled_file(obj, filename: str):
        assert not os.path.exists(filename), "File already exists"
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    set_matchms_logger_level("WARNING")


    def repair_adduct(spectra, mass_tolerance):
        not_repaired = []
        repaired = []
        corrected = 0
        for spectrum in tqdm(spectra):
            spectrum_out = repair_smiles_salt_ions(spectrum, mass_tolerance=mass_tolerance)
            if require_parent_mass_match_smiles(spectrum_out, mass_tolerance) is None:
                not_repaired.append(spectrum)
            else:
                repaired.append(spectrum)
                corrected += 1
        return not_repaired, repaired


    lib_dir = "../../../../../ms2deepscore/data/cleaning_spectra/"
    incorrect_spectra = load_pickled_file( os.path.join(lib_dir,
                                    f"spectra_not_in_0.01.pickle"))
    print(len(incorrect_spectra))

    salt_ions = []
    for spectrum in incorrect_spectra:
        smiles = spectrum.get("smiles")
        if "." in smiles:
            salt_ions.append(spectrum)
    print(len(salt_ions))
    not_repaired, repaired = repair_adduct(salt_ions, mass_tolerance=0.1)
    print(len(repaired))
    print(len(not_repaired))
