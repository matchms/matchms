import numpy as np


def interpret_charge_mode(spectrum):
    """Derive the charge value and complete ionmode based on metadata.

    Often, MGF files do not provide a correct charge value or ionmode. This
    function corrects for some of the most common errors by extracting the
    adduct and using this to complete missing ionmode fields. Finally, this
    is used to infering the correct charge sign.
    """
    spectrum = spectrum.clone()
    # Start by completing missing ionmode fields
    complete_ionmode(spectrum)
    charge = spectrum.metadata["charge"]
    ionmode = spectrum.metadata["ionmode"]

    # 1) Go through most obviously wrong cases
    if not charge:
        charge = 0
    if charge == 0:
        if ionmode.lower() == 'positive':
            charge = 1
        elif ionmode.lower() == 'negative':
            charge = -1

    # 2) Correct charge when in conflict with ionmode (trust ionmode more!)
    if np.sign(charge) == 1 and ionmode.lower() == 'negative':
        charge *= -1
    elif np.sign(charge) == -1 and ionmode.lower() == 'positive':
        charge *= -1

    # TODO: 3) extend method to deduce charge value based on adduct
    spectrum.metadata["charge"] = charge
    return spectrum


def complete_ionmode(spectrum):
    """Derive missing ionmode based on adduct.

    MGF files do not always provide a correct ionmode. This function reads
    the adduct from the metadata and uses this to fill in the correct ionmode
    where missing.
    """
    # Lists of known adducts (Justin JJ van der Hooft, 2020)
    # TODO: Read those from yaml or json file?
    known_adducts_positive = ['M-2H2O+H', 'M+H', 'M+H-CH3NH2', 'M+Na', 'M-H2O+H+', 'M+H+Na',
                              'M+H+', 'M+K', 'M+H', 'M+K+', '2M+Na', 'M+', 'M+3H', 'M+2H++',
                              'M+NH4', 'M+ACN+H', 'M+H-NH3', 'M-H2O+H', 'M+', 'M+Na+', 'M+',
                              'M+2H', 'M+H-H2O', 'M-2H2O+H+', 'M+', 'M+NH4+', 'Cat', 'M+',
                              'M+Na', '2M+H', 'M+H-2H2O', 'M+2H', 'M+2H']
    known_adducts_negative = ['M+CH3COO-/M-CH3-', 'M-H', 'M+CH3COO-', 'M-', '2M-H', 'M-H-/M-Ser-',
                              'M-', 'M-H-', 'M-2H-', 'M-H2O-H', 'M+FA-H', 'M+Cl', '(M+CH3COOH)-H-',
                              'M-H-H2O', 'M-H-CO2-2HF-']

    spectrum = spectrum.clone()
    ionmode = spectrum.metadata["ionmode"]
    # Try extracting the adduct from given compound name
    add_adducts(spectrum)
    if "adduct" in spectrum.metadata:
        adduct = spectrum.metadata["adduct"]
    else:
        adduct - None

    # Try completing missing or incorrect ionmodes
    if ionmode.lower() not in ['positive', 'negative']:
        if adduct in known_adducts_positive:
            ionmode = 'Positive'
            print("Added ionmode=", ionmode, "based on adduct:", adduct)
        elif adduct in known_adducts_negative:
            ionmode = 'Negative'
            print("Added ionmode=", ionmode, "based on adduct:", adduct)
        else:
            ionmode = 'n/a'
    spectrum.metadata["ionmode"] = ionmode
    return spectrum


def add_adducts(spectrum):
    """Add adduct to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.
    """
    spectrum = spectrum.clone()
    if 'adduct' not in spectrum.metadata:
        try:
            name = spectrum.metadata["name"]
            adduct = name.split(' ')[-1]
            adduct = adduct.replace('\n', '').replace(' ', '').replace(
                '[', '').replace(']', '').replace('*', '')
            if adduct:
                spectrum.metadata["adduct"] = adduct
        except KeyError:
            print("No name and no adduct found.")
    return spectrum
