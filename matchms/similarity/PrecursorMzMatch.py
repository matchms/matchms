from .MetadataMatch import MetadataMatch


class PrecursorMzMatch(MetadataMatch):
    """Return True if spectra match in precursor m/z, and False otherwise.

    The match within tolerance can be calculated based on an absolute m/z
    difference (``tolerance_type="Dalton"``) or based on a relative
    difference in ppm (``tolerance_type="ppm"``).
    """

    def __init__(self, tolerance: float = 0.1, tolerance_type: str = "Dalton"):
        """
        Parameters
        ----------
        tolerance
            Specify tolerance below which two precursor m/z values are counted as match.
        tolerance_type
            Choose between fixed tolerance in Dalton (``"Dalton"``) or a relative
            difference in ppm (``"ppm"``).
        """
        super().__init__(
            field="precursor_mz",
            matching_type="difference",
            tolerance=tolerance,
            tolerance_type=tolerance_type,
        )
