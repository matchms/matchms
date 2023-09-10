import logging
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts
from ..filters.clean_adduct import CleanAdduct
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class DeriveIonmode(BaseSpectrumFilter):
    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        # Load lists of known adducts
        known_adducts = load_known_adducts()

        adduct = spectrum.get("adduct", None)
        # Harmonize adduct string
        if adduct:
            adduct = CleanAdduct._clean_adduct(adduct)

        ionmode = spectrum.get("ionmode")
        if ionmode:
            assert ionmode == ionmode.lower(), ("Ionmode field not harmonized.",
                                                "Apply 'make_ionmode_lowercase' filter first.")
        if ionmode in ["positive", "negative"]:
            return spectrum
        # Try completing missing or incorrect ionmodes
        if adduct in list(known_adducts["adduct"]):
            ionmode = known_adducts.loc[known_adducts["adduct"] == adduct, "ionmode"].values[0]
        else:
            ionmode = "n/a"

        spectrum.set("ionmode", ionmode)
        logger.info("Set ionmode to %s.", ionmode)

        return spectrum
