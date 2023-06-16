from molmass import Formula
import re
from matchms.constants import ELECTRTON_MASS


def get_multiplier_and_mass(adduct):
    charge = get_charge_of_adduct(adduct)

    parent_mass, ions = get_ions_from_adduct(adduct)

    added_mass = get_mass_of_ion(ions) + ELECTRTON_MASS * -charge

    multiplier = 1/abs(charge)*parent_mass
    correction_mass = added_mass/(abs(charge))
    return multiplier, correction_mass


def get_ions_from_adduct(adduct):
    """Returns a list of ions from an adduct.

    e.g. '[M+H-H2O]2+' -> ["M", "+H", "-H2O"]
    """

    # Get adduct from brackets
    if "[" in adduct:
        ions_part = re.findall((r"\[(.*)\]"), adduct)
        assert len(ions_part) == 1, f"Expected to find brackets [] once, not the case in {adduct}"
        adduct = ions_part[0]
    # Finds the pattern M or 2M in adduct it makes sure it is in between
    parent_mass = re.findall(r'(?:^|[+-])([0-9]?M)(?:$|[+-])', adduct)
    assert len(parent_mass)!=0, f"The parent mass (e.g. 2M or M) was not found in {adduct}"
    assert len(parent_mass) ==1, f"The parent mass (e.g. 2M or M) was found multiple times in {adduct}"
    parent_mass = parent_mass[0]
    assert len(parent_mass) < 3, "expected the parent ion of form M, 2M or 3M"
    if parent_mass == "M":
        parent_mass = 1
    else:
        parent_mass = int(parent_mass[0])

    ions_split = re.findall(r'([+-][0-9a-zA-Z]+)', adduct)
    ions_split = replace_abbreviations(ions_split)
    return parent_mass, ions_split


def split_ion(ion):
    sign = ion[0]
    ion = ion[1:]
    assert sign in ["+", "-"], "Expected ion to start with + or -"
    match = re.match(r'^([0-9]+)(.*)', ion)
    if match:
        number = match.group(1)
        ion = match.group(2)
    else:
        number = 1
    return sign, number, ion


def replace_abbreviations(ions_split):
    abbrev_to_formula = {'ACN': 'CH3CN', 'DMSO': 'C2H6OS', 'FA': 'CH2O2',
                         'HAc': 'CH3COOH', 'Hac': 'CH3COOH', 'TFA': 'C2HF3O2',
                         'IsoProp': 'CH3CHOHCH3', 'MeOH': 'CH3OH'}
    corrected_ions = []
    for ion in ions_split:
        sign, number, ion = split_ion(ion)
        if ion in abbrev_to_formula:
            ion = abbrev_to_formula[ion]
        corrected_ions.append(sign + str(number) + ion)
    return corrected_ions


def get_mass_of_ion(ions):
    added_mass = 0
    for ion in ions:
        sign, number, ion = split_ion(ion)
        formula = Formula(ion)
        atom_mass = formula.isotope.mass
        if sign == "-":
            number = -int(number)
        else:
            number = int(number)
        added_mass += number * atom_mass
    return added_mass


def get_charge_of_adduct(adduct):
    charge = re.findall((r"\]([0-9]?[+-])"), adduct)
    if len(charge) == 0:
        return None
    if len(charge) == 1:
        return parse_charge(charge[0])
    print(f'Warning: Charge was found multiple times in adduct {adduct}')


def parse_charge(charge):
    if len(charge) == 1:
        charge_size = "1"
        charge_sign = charge
    elif len(charge) == 2:
        charge_size = charge[0]
        charge_sign = charge[1]
    else:
        assert False, f"Charge is expected of length 1 or 2 {charge} was given"

    return int(charge_sign+charge_size)
