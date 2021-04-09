import numpy


class Spikes:
    """
    Stores arrays of intensities and M/z values, with some checks on their internal consistency.
    """
    def __init__(self, mz=None, intensities=None):
        assert isinstance(mz, numpy.ndarray), "Input argument 'mz' should be a numpy.array."
        assert isinstance(intensities, numpy.ndarray), "Input argument 'intensities' should be a numpy.array."
        assert mz.shape == intensities.shape, "Input arguments 'mz' and 'intensities' should be the same shape."
        assert mz.dtype == "float", "Input argument 'mz' should be an array of type float."
        assert intensities.dtype == "float", "Input argument 'intensities' should be an array of type float."

        self._mz = mz
        self._intensities = intensities

        assert self._is_sorted(), "mz values are out of order."

    def __eq__(self, other):
        return \
            self.mz.shape == other.mz.shape and \
            numpy.allclose(self.mz, other.mz) and \
            self.intensities.shape == other.intensities.shape and \
            numpy.allclose(self.intensities, other.intensities)

    def __len__(self):
        return self._mz.size

    def __getitem__(self, item):
        return [self.mz, self.intensities][item]

    def _is_sorted(self):
        return numpy.all(self.mz[:-1] <= self.mz[1:])

    def clone(self):
        return Spikes(self.mz, self.intensities)

    @property
    def mz(self):
        """getter method for mz private variable"""
        return self._mz.copy()

    @property
    def intensities(self):
        """getter method for intensities private variable"""
        return self._intensities.copy()

    @property
    def to_numpy(self):
        """getter method to return stacked numpy array of both peak mz and
        intensities"""
        return numpy.vstack((self.mz, self.intensities)).T
