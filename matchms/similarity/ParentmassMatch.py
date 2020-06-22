from matchms.typing import SpectrumType


class ParentmassMatch:
    """Return True if spectrums match in parent mass (within tolerance), and False otherwise."""

    def __init__(self, tolerance: float = 0.1):
        """
        Parameters:
        ----------
        tolerance
            Specify tolerance below which two masses are counted as match.
        """
        self.tolerance = tolerance

    def __call__(self, spectrum: SpectrumType, reference_spectrum: SpectrumType) -> bool:
        """Compare parent masses"""
        parentmass = spectrum.get("parent_mass")
        parentmass_ref = reference_spectrum.get("parent_mass")
        assert parentmass is not None and parentmass_ref is not None, "Missing parent mass."
        return abs(parentmass-parentmass_ref) <= self.tolerance
