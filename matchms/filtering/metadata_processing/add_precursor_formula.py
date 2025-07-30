import logging
import re
from collections import Counter
from typing import Optional
from matchms.filtering.filter_utils.interpret_unknown_adduct import (
    get_ions_from_adduct,
    split_ion,
)


logger = logging.getLogger("matchms")


def add_precursor_formula(spectrum_in, clone: Optional[bool] = True,):
    """Adds the precursor formula based on the smiles and adduct"""
    if spectrum_in is None:
        return None
    spectrum_in = spectrum_in.clone() if clone else spectrum_in

    adduct = spectrum_in.get("adduct")
    formula_str = spectrum_in.get('formula')
    if formula_str is None or adduct is None:
        logger.info("No formula available, so precursor_formula is not set")
        return spectrum_in

    nr_of_parent_masses, ions_split = get_ions_from_adduct(adduct)
    original_precursor_formula = convert_formula_string_to_atom_counter(formula_str)

    new_precursor_formula = Counter()
    for i in range(nr_of_parent_masses):
        new_precursor_formula += original_precursor_formula
    for ion in ions_split:
        sign, number, formula = split_ion(ion)
        for i in range(number):
            if sign == "+":
                new_precursor_formula.update(convert_formula_string_to_atom_counter(formula))
            if sign == "-":
                new_precursor_formula.subtract(convert_formula_string_to_atom_counter(formula))
    has_negative = any(atom_count < 0 for atom_count in new_precursor_formula.values())
    if has_negative:
        logger.warning(f"The adduct: {adduct}, removes atoms not in the formula: {formula}, "
                       f"so no precursor_formula could be set")
        return spectrum_in
    spectrum_in.set("precursor_formula", convert_atom_counter_to_str(new_precursor_formula))
    return spectrum_in

def convert_formula_string_to_atom_counter(formula_str):
    """Converts a molecular formula as str to a counter (kind of dict) with the atom counts"""
    atoms_and_counts = re.findall(r'([A-Z][a-z]?)(\d*)', formula_str)
    return Counter({atom: int(count) if count else 1 for atom, count in atoms_and_counts})

def convert_atom_counter_to_str(atom_counter):
    """Converts a dictionary with atom counts to a str in hill notation (C first, H second rest alphabetically)"""
    elements = list(atom_counter.keys())
    parts = []
    if 'C' in elements:
        parts.append(f"C{atom_counter['C'] if atom_counter['C'] != 1 else ''}")
        elements.remove('C')
    if 'H' in elements:
        parts.append(f"H{atom_counter['H'] if atom_counter['H'] != 1 else ''}")
        elements.remove('H')
    for el in sorted(elements):
        count = atom_counter[el]
        parts.append(f"{el}{count if count != 1 else ''}")
    return ''.join(parts)
