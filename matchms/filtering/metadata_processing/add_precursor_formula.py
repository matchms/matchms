import logging
import re
from collections import Counter
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.interpret_unknown_adduct import (
    get_ions_from_adduct,
    split_ion,
)
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _add_precursor_formula(metadata) -> dict:
    """Derive and set ``precursor_formula`` from neutral ``formula`` and ``adduct``.

    Requirements
    ------------
    - Input metadata must contain ``formula`` and ``adduct``.
    - ``formula`` must be a simple concatenation of element symbols and counts,
      without parentheses, hydrates, or isotope notation.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    clone
        Optionally clone the input before applying the filter. If ``False``, the
        input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with added ``precursor_formula`` metadata, or ``None`` if
        the input was ``None``.
    """
    adduct = as_string_or_none(metadata.get("adduct"))
    formula_str = as_string_or_none(metadata.get("formula"))

    if formula_str is None or adduct is None:
        logger.info(
            "Missing 'formula' or 'adduct' (formula=%s, adduct=%s); "
            "'precursor_formula' not set.",
            formula_str,
            adduct,
        )
        return {}

    nr_of_parent_masses, ions_split = get_ions_from_adduct(adduct)
    original_precursor_formula = _convert_formula_string_to_atom_counter(formula_str)

    new_precursor_formula = Counter()
    for _ in range(nr_of_parent_masses):
        new_precursor_formula += original_precursor_formula

    for ion in ions_split:
        sign, number, formula = split_ion(ion)
        ion_formula = _convert_formula_string_to_atom_counter(formula)

        for _ in range(number):
            if sign == "+":
                new_precursor_formula.update(ion_formula)
            elif sign == "-":
                new_precursor_formula.subtract(ion_formula)

    has_negative = any(atom_count < 0 for atom_count in new_precursor_formula.values())
    if has_negative:
        logger.warning(
            "Adduct %s leads to negative element count with formula %s. "
            "'precursor_formula' not set.",
            adduct,
            formula_str,
        )
        return {}

    return {
        "precursor_formula": _convert_atom_counter_to_str(new_precursor_formula)
    }


def _convert_formula_string_to_atom_counter(formula_str):
    """Parse a simple elemental formula into a Counter."""
    atoms_and_counts = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)
    return Counter(
        {atom: int(count) if count else 1 for atom, count in atoms_and_counts}
    )


def _convert_atom_counter_to_str(atom_counter):
    """Format a mapping of element counts into Hill notation."""
    filtered = {el: int(cnt) for el, cnt in atom_counter.items() if cnt > 0}

    parts: list[str] = []

    if "C" in filtered:
        c = filtered.pop("C")
        parts.append(f"C{'' if c == 1 else c}")

    if "H" in filtered:
        h = filtered.pop("H")
        parts.append(f"H{'' if h == 1 else h}")

    for el in sorted(filtered.keys()):
        cnt = filtered[el]
        parts.append(f"{el}{'' if cnt == 1 else cnt}")

    return "".join(parts)


add_precursor_formula = metadata_update_filter(_add_precursor_formula)