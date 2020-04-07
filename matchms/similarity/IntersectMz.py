class IntersectMz:
    """IntersectMz function factory"""

    def __init__(self, label):
        """constructor"""
        self.label = label

    def __call__(self, spectrum, reference_spectrum):
        """call method"""
        mz = set(spectrum.mz)
        mz_ref = set(reference_spectrum.mz)
        intersected = mz.intersection(mz_ref)
        unioned = mz.union(mz_ref)
        return len(intersected) / len(unioned)
