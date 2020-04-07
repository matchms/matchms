from matplotlib import pyplot as plt


class Spectrum:
    """An example docstring for a class."""
    def __init__(self, mz, intensities, metadata):
        """An example docstring for a constructor."""
        self.mz = mz
        self.intensities = intensities
        self.metadata = metadata

    def clone(self):
        """An example docstring for a method."""
        return Spectrum(mz=self.mz, intensities=self.intensities, metadata=self.metadata)

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
