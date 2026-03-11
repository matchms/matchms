from .MetadataMatch import MetadataMatch


class ParentMassMatch(MetadataMatch):
    """Return True if spectra match in parent mass, and False otherwise."""

    def __init__(self, tolerance: float = 0.1):
        """
        Parameters
        ----------
        tolerance
            Specify tolerance below which two parent masses are counted as match.
        """
        super().__init__(
            field="parent_mass",
            matching_type="difference",
            tolerance=tolerance,
            tolerance_type="Dalton",
        )
