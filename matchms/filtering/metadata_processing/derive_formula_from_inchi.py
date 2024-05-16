import logging
import re
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import \
    is_valid_inchi


logger = logging.getLogger("matchms")


def derive_formula_from_inchi(spectrum_in, overwrite=True):
    spectrum = spectrum_in.clone()
    if spectrum.get("formula") is not None:
        if overwrite is False:
            return spectrum

    formula = _get_formula_from_inchi(spectrum.get("inchi"))
    if formula is not None:
        if spectrum.get("formula") is not None:
            if spectrum.get("formula") != formula:
                print(
                    f"Overwriting formula from inchi. Original formula: {spectrum.get('formula')} New formula: {formula}")
                spectrum.set("formula", formula)
        else:
            print(f"Added formula from inchi. New formula: {formula}")
            spectrum.set("formula", formula)
    return spectrum


def _get_formula_from_inchi(inchi):
    if is_valid_inchi(inchi):
        inchi = inchi.strip('"')
        regexp = r"(InChI=1|1)(S\/|\/)(([A-Z][a-z]?\d*)+)\/"
        match = re.search(regexp, inchi)
        if match:
            return match.group(3)
    return None
