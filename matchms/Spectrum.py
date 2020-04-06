from matplotlib import pyplot as plt


class Spectrum:

    def __init__(self, mz, intensities, metadata):
        self.mz = mz
        self.intensities = intensities
        self.metadata = metadata
        if isinstance(self.metadata["charge"], list):  # Avoid pyteomics ChargeList
            self.metadata["charge"] = int(self.metadata["charge"][0])

    def clone(self):
        return Spectrum(mz=self.mz, intensities=self.intensities, metadata=self.metadata)

    def plot(self):
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
