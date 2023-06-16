import re

from matchms.filtering import load_adducts_dict, load_known_adduct_conversions


def looks_like_adduct(adduct):
    """Return True if input string has expected format of an adduct."""
    if not isinstance(adduct, str):
        return False
    # Clean adduct
    adduct = clean_adduct(adduct)
    # Load lists of default known adducts
    known_adducts = load_adducts_dict()
    if adduct in known_adducts:
        return True

    # Expect format like: "[2M-H]" or "[2M+Na]+"
    regexp1 = r"^\[(([0-4]M)|(M[0-9])|(M))((Br)|(Br81)|(Cl)|(Cl37)|(S)){0,}[+-][A-Z0-9\+\-\(\)aglire]{1,}[\]0-4+-]{1,4}"
    return re.search(regexp1, adduct) is not None


def clean_adduct(adduct: str) -> str:
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    adduct
        Input adduct string to be cleaned/edited.
    """
    def get_adduct_charge(adduct):
        regex_charges = r"[1-3]{0,1}[+,-]{1,2}$"
        match = re.search(regex_charges, adduct)
        if match:
            return match.group(0)
        return match

    def adduct_conversion(adduct):
        """Convert adduct if conversion rule is known"""
        adduct_conversions = load_known_adduct_conversions()
        if adduct in adduct_conversions:
            return adduct_conversions[adduct]
        return adduct

    if not isinstance(adduct, str):
        return adduct

    adduct = adduct.strip().replace("\n", "").replace("*", "")
    adduct = adduct.replace("++", "2+").replace("--", "2-")
    if adduct.startswith("["):
        return adduct_conversion(adduct)

    if adduct.endswith("]"):
        return adduct_conversion("[" + adduct)

    adduct_core = "[" + adduct
    # Remove parts that can confuse the charge extraction
    for mol_part in ["CH2", "CH3", "NH3", "NH4", "O2"]:
        if mol_part in adduct:
            adduct = adduct.split(mol_part)[-1]
    adduct_charge = get_adduct_charge(adduct)

    if adduct_charge is None:
        return adduct_conversion(adduct_core + "]")

    adduct_cleaned = adduct_core[:-len(adduct_charge)] + "]" + adduct_charge
    return adduct_conversion(adduct_cleaned)
