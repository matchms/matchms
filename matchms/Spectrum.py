from matplotlib import pyplot as plt


class Spectrum:
    """An example docstring for a class."""
    def __init__(self, mz, intensities, metadata):
        """An example docstring for a constructor."""
        self.mz = mz
        self.intensities = intensities
        self._metadata = metadata

    def clone(self):
        """An example docstring for a method."""
        return Spectrum(mz=self.mz.copy(),
                        intensities=self.intensities.copy(),
                        metadata=self._metadata.copy())

    def get(self, key, default=None):
        return self._metadata.get(key, default)

    def plot(self):
        """An example docstring for a method."""
        plt.figure(figsize=(10, 10))

        plt.stem(self.mz,
                 self.intensities,
                 linefmt='-',
                 markerfmt='.',
                 basefmt='r-',
                 use_line_collection=True)
        plt.grid(True)
        plt.title('Chromatogram')
        plt.xlabel('M/z')
        plt.ylabel('Intensity')

    def set(self, key, value):
        self._metadata[key] = value
        return self

    @property
    def mz(self):
        """getter method for mz private variable"""
        return self.__mz.copy()

    @mz.setter
    def mz(self, value):
        """setter method for mz private variable"""
        self.__mz = value

    @property
    def intensities(self):
        """getter method for intensities private variable"""
        return self.__intensities.copy()

    @intensities.setter
    def intensities(self, value):
        """setter method for intensities private variable"""
        self.__intensities = value
