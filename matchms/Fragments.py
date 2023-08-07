from typing import Tuple
import numpy as np


class Fragments:
    """
    Stores arrays of intensities and M/z values, with some checks on their internal consistency.

    For example

    .. testcode::

        import numpy as np
        from matchms import Fragments

        mz = np.array([10, 20, 30], dtype="float")
        intensities = np.array([100, 20, 300], dtype="float")

        peaks = Fragments(mz=mz, intensities=intensities)
        print(peaks[2])

    Should output

    .. testoutput::

       [ 30. 300.]

    Attributes
    ----------
    mz:
        Numpy array of m/z values.
    intensities:
        Numpy array of peak intensity values.

    """
    def __init__(self, mz=None, intensities=None):
        assert isinstance(mz, np.ndarray), "Input argument 'mz' should be a np.array."
        assert isinstance(intensities, np.ndarray), "Input argument 'intensities' should be a np.array."
        assert mz.shape == intensities.shape, "Input arguments 'mz' and 'intensities' should be the same shape."
        assert mz.dtype == "float", "Input argument 'mz' should be an array of type float."
        assert intensities.dtype == "float", "Input argument 'intensities' should be an array of type float."

        self._mz = mz
        self._intensities = intensities

        assert self._is_sorted(), "mz values are out of order."

    def __eq__(self, other):
        return \
            self.mz.shape == other.mz.shape and \
            np.allclose(self.mz, other.mz) and \
            self.intensities.shape == other.intensities.shape and \
            np.allclose(self.intensities, other.intensities)

    def __len__(self):
        return self._mz.size

    def __getitem__(self, item):
        return np.asarray([self.mz[item], self.intensities[item]])

    def _is_sorted(self):
        return np.all(self.mz[:-1] <= self.mz[1:])

    def clone(self):
        return Fragments(self.mz, self.intensities)

    def get(self, mz: float, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get peaks at specified mz value with given tolerance

        Args:
            mz (float): mz value at which to look for a peak
            tolerance (float): tolerance to use for mz matching

        Returns:
            Tuple[np.ndarray, np.ndarray]: array of mz values and intensities for this query
        """
        # maybe it is smarter to implement this with np.where?
        indices = np.argwhere(abs(self._mz - mz) <= tolerance)
        return self._mz[indices], self._intensities[indices]

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
        return np.vstack((self.mz, self.intensities)).T
    
