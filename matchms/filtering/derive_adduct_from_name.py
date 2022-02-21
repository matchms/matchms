import logging
from ..metadata_utils import clean_adduct, looks_like_adduct
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def derive_adduct_from_name(spectrum_in: SpectrumType,
                            remove_adduct_from_name: bool = True) -> SpectrumType:
    """Find adduct in compound name and add to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    remove_adduct_from_name:
        Remove found adducts from compound name if set to True. Default is True.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("compound_name", None) is not None:
        name = spectrum.get("compound_name")
    else:
        assert spectrum.get("name", None) in [None, ""], ("Found 'name' but not 'compound_name' in metadata",
                                                          "Apply 'add_compound_name' filter first.")
        return spectrum

    # Detect adduct in compound name
    adduct_from_name = None
    name_split = name.split(" ")
    for name_part in name_split[::-1][:2]:
        if looks_like_adduct(name_part):
            adduct_from_name = name_part
            break

    if adduct_from_name and remove_adduct_from_name:
        name_adduct_removed = " ".join([x for x in name_split if x != adduct_from_name])
        name_adduct_removed = name_adduct_removed.strip("; ")
        spectrum.set("compound_name", name_adduct_removed)
        logger.info("Removed adduct %s from compound name.", adduct_from_name)

    # Add found adduct to metadata (if not present yet)
    if adduct_from_name and not looks_like_adduct(spectrum.get("adduct")):
        adduct_cleaned = clean_adduct(adduct_from_name)
        spectrum.set("adduct", adduct_cleaned)
        logger.info("Added adduct %s to metadata.", adduct_cleaned)

    return spectrum
