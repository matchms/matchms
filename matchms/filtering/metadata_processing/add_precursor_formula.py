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
    """Derive and set 'precursor_formula' from neutral 'formula' and 'adduct'.

    Requirements:
      - spectrum_in must have metadata keys: 'formula' (neutral) and 'adduct'.
      - 'formula' must be a simple concatenation of element symbols and counts
        (no parentheses/hydrates/isotopes).
    """
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone() if clone else spectrum_in

    adduct = spectrum.get("adduct")
    formula_str = spectrum.get('formula')
    if formula_str is None or adduct is None:
        logger.info(
            f"Missing 'formula' or 'adduct' (formula={formula_str}, adduct={adduct});"\
            "'precursor_formula' not set."
            )
        return spectrum

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
        logger.warning(
            f"Adduct {adduct} leads to negative element count with formula {formula_str}."\
            "'precursor_formula' not set.")
        return spectrum
    spectrum.set("precursor_formula", convert_atom_counter_to_str(new_precursor_formula))
    return spectrum

def convert_formula_string_to_atom_counter(formula_str):
    """Parse a simple elemental formula (no parentheses/hydrates/isotopes) into a Counter."""
    atoms_and_counts = re.findall(r'([A-Z][a-z]?)(\d*)', formula_str)
    return Counter({atom: int(count) if count else 1 for atom, count in atoms_and_counts})

def convert_atom_counter_to_str(atom_counter):
    """Format a mapping of element counts into Hill notation (C, H, then alphabetical)."""
    elements = list(atom_counter.keys())
    parts = []
    # C, then H
    if 'C' in elements:
        parts.append(f"C{atom_counter['C'] if atom_counter['C'] != 1 else ''}")
        elements.remove('C')
    if 'H' in elements:
        parts.append(f"H{atom_counter['H'] if atom_counter['H'] != 1 else ''}")
        elements.remove('H')
    # Then rest alphabetically
    for el in sorted(elements):
        count = atom_counter[el]
        parts.append(f"{el}{count if count != 1 else ''}")
    return ''.join(parts)
