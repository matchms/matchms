import re
from ..typing import SpectrumType
from ..utils import looks_like_adduct


def clean_compound_name(spectrum_in: SpectrumType) -> SpectrumType:
    """Clean compound name. Includes removing potential adduct.

    This function will look for an adduct string and remove it.
    A list of frequently seen name additions that do not belong to the compound
    name will be removed."""
    def remove_adduct(name):
        """Find and remove adduct string."""
        potential_adduct = name.split(' ')[-1].strip().replace('*', '')
        if looks_like_adduct(potential_adduct):
            name_split = name.split(" ")
            name = " ".join(name_split[:-1])
        return name

    def remove_non_compound_name_parts(name):
        """Clean "name string by removing known parts that don't belong there."""
        name = name.strip()
        # remove type NCGC00180417-03_C31H40O16_
        name = re.split(r"[A-Z]{3,}[0-9]{8,}-[0-9]{2,}_[A-Z,0-9]{4,}_", name)[-1]
        # remove type NCGC00160232-01! or MLS001142816-01!
        name = re.split(r"[A-Z]{3,}[0-9]{8,}-[0-9]{2,}\!", name)[-1]
        # remove type Massbank:EA008813 option1|option2|option3
        name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2,}[0-9]{5,}.*\|", name)[-1]
        # remove type Massbank:EA008813 or MassbankEU:EA008813
        name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2,}[0-9]{5,}", name)[-1]
        # remove type HMDB:HMDB00943-1336
        name = re.split(r"HMDB:HMDB[0-9]{4,}-[0-9]{1,}", name)[-1]
        # remove type MoNA:662599
        name = re.split(r"MoNA:[0-9]{5,}", name)[-1]
        # ReSpect:PS013405 option1|option2|option3...
        name = re.split(r"ReSpect:[A-Z]{2,}[0-9]{5,}.*\|", name)[-1]
        # remove type 0072_2-Mercaptobenzothiaz
        name = re.split(r"^[0-9]{4}_", name)[-1]

        # Remove further non compound-name parts
        parts_remove = ["Spectral Match to",
                        "from NIST14",
                        "Massbank:"]
        for part in parts_remove:
            name = name.replace(part, "")
        return name.strip()

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("compound_name", None):
        name = spectrum.get("compound_name")
    else:
        name = spectrum.get("name", None)

    if name:
        # Clean found name string
        name_cleaned = remove_adduct(name)
        name_cleaned = remove_non_compound_name_parts(name_cleaned)
        if name_cleaned != name:
            spectrum.set("compound_name", name_cleaned)
            print("Added cleaned compound name:", name_cleaned)

    return spectrum
