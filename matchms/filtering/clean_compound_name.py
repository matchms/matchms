import logging
import re
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def clean_compound_name(spectrum_in: SpectrumType) -> SpectrumType:
    """Clean compound name.

    A list of frequently seen name additions that do not belong to the compound
    name will be removed.
    """
    def remove_parts_by_regular_expression(name):
        """Clean name string by removing known parts that don't belong there."""
        name = name.strip()
        # remove type NCGC00180417-03_C31H40O16_
        name = re.split(r"[A-Z]{3,}[0-9]{8,}-[0-9]{2,}_[A-Z,0-9]{4,}_", name)[-1]
        # remove type NCGC00160232-01! or MLS001142816-01!
        name = re.split(r"[A-Z]{3,}[0-9]{8,}-[0-9]{2,3}\!", name)[-1]
        # remove type Massbank:EA008813 option1|option2|option3
        name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2}[0-9]{5,6}.*\|", name)[-1]
        # remove type Massbank:EA008813 or MassbankEU:EA008813
        name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2}[0-9]{5,6}", name)[-1]
        # remove type HMDB:HMDB00943-1336
        name = re.split(r"HMDB:HMDB[0-9]{4,}-[0-9]{1,}", name)[-1]
        # remove type MoNA:662599
        name = re.split(r"MoNA:[0-9]{5,}", name)[-1]
        # ReSpect:PS013405 option1|option2|option3...
        name = re.split(r"ReSpect:[A-Z]{2,}[0-9]{6}.*\|", name)[-1]
        # ReSpect:PS013405 option1
        name = re.split(r"[A-Z]{2,}[0-9]{6}( )", name)[-1]
        # remove type 0072_2-Mercaptobenzothiaz
        name = re.split(r"^[0-9]{4}_", name)[-1]
        # remove type nameofcompound_CID20_170920 or Spiraeoside_HCD30_170919
        name = re.split(r"_((HCD)|(CID))[0-9]{2}_[0-9]{5,6}$", name)[0]
        return name

    def remove_known_non_compound_parts(name):
        """Remove known non compound-name strings from name."""
        parts_remove = ["Spectral Match to",
                        "from NIST14",
                        "Massbank:"]
        for part in parts_remove:
            name = name.replace(part, "")
        return name.strip("; ")

    def remove_misplaced_mass(name):
        """Remove occasionally occurring parent mass addition to name."""
        regex_mass = r"^[0-9]{2,4}\.[0-9]$"
        end_part = name.split(" ")[-1]
        if re.search(regex_mass, end_part) is not None:
            return name.replace(end_part, "").strip()
        return name

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("compound_name", None) is not None:
        name = spectrum.get("compound_name")
    else:
        assert spectrum.get("name", None) in [None, ""], ("Found 'name' but not 'compound_name' in metadata",
                                                          "Apply 'add_compound_name' filter first.")
        return spectrum

    # Clean compound name
    name_cleaned = remove_parts_by_regular_expression(name)
    name_cleaned = remove_known_non_compound_parts(name_cleaned)
    name_cleaned = remove_misplaced_mass(name_cleaned)
    if name_cleaned != name:
        spectrum.set("compound_name", name_cleaned)
        logger.info("Added cleaned compound name: %s", name_cleaned)

    return spectrum
