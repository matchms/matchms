import os
import yaml
import numpy as np


def interpret_charge_mode(spectrum,
                          file_known_adducts='known_adducts.yaml'):
    """Derive the charge value and complete ionmode based on metadata.

    Often, MGF files do not provide a correct charge value or ionmode. This
    function corrects for some of the most common errors by extracting the
    adduct and using this to complete missing ionmode fields. Finally, this
    is used to infering the correct charge sign.

    Args:
    ----
    spectrum: matchms.Spectrum.Spectrum()
        Input spectrum.
    known_adducts_yaml: str
        Filename of yaml listing known adduct strings.
        Default is 'known_adducts.yaml'.
    """
    # Start by completing missing ionmode fields
    complete_ionmode(spectrum, file_known_adducts)
    charge = spectrum.metadata["charge"]
    ionmode = spectrum.metadata["ionmode"]

    # 1) Go through most obviously wrong cases
    if not charge:
        charge = 0
    if charge == 0:
        if ionmode == 'positive':
            charge = 1
        elif ionmode == 'negative':
            charge = -1

    # 2) Correct charge when in conflict with ionmode (trust ionmode more!)
    if np.sign(charge) == 1 and ionmode == 'negative':
        charge *= -1
    elif np.sign(charge) == -1 and ionmode == 'positive':
        charge *= -1

    # TODO: 3) extend method to deduce charge value based on adduct
    spectrum.metadata["charge"] = charge
    return spectrum


def complete_ionmode(spectrum, file_known_adducts):
    """Derive missing ionmode based on adduct.

    MGF files do not always provide a correct ionmode. This function reads
    the adduct from the metadata and uses this to fill in the correct ionmode
    where missing.

    Args:
    ----
    spectrum: matchms.Spectrum.Spectrum()
        Input spectrum.
    known_adducts_yaml: str
        Filename of yaml listing known adduct strings.
    """
    # Load lists of known adducts
    file_known_adducts = os.path.join(os.path.dirname(__file__),
                                      file_known_adducts)
    if os.path.isfile(file_known_adducts):
        with open(file_known_adducts, 'r') as ymlfile:
            known_adducts = yaml.full_load(ymlfile)
    else:
        print("Could not find yaml file with known adducts.")
        known_adducts = {'adducts_positive': [],
                         'adducts_negative': []}
    ionmode = spectrum.metadata["ionmode"]
    # Try extracting the adduct from given compound name
    add_adducts(spectrum)
    if "adduct" in spectrum.metadata:
        adduct = spectrum.metadata["adduct"]
    else:
        adduct = None

    # Try completing missing or incorrect ionmodes
    if ionmode not in ['positive', 'negative']:
        if adduct in known_adducts["adducts_positive"]:
            ionmode = 'positive'
            print("Added ionmode=", ionmode, "based on adduct:", adduct)
        elif adduct in known_adducts["adducts_negative"]:
            ionmode = 'negative'
            print("Added ionmode=", ionmode, "based on adduct:", adduct)
        else:
            ionmode = 'n/a'
    spectrum.metadata["ionmode"] = ionmode
    return spectrum


def add_adducts(spectrum):
    """Add adduct to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.
    """
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
