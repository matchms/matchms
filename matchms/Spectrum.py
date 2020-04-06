from matplotlib import pyplot as plt


class Spectrum:

    def __init__(self, mz, intensities, metadata):
        self.mz = mz
        self.intensities = intensities
        self.metadata = metadata

        # Avoid pyteomics ChargeList
        if isinstance(self.metadata["charge"], list):
            self.metadata["charge"] = int(self.metadata["charge"][0])

        # Lowercase ionmode and replace missing ones by'n/a'
        if "ionmode" in self.metadata:
            self.metadata["ionmode"] = self.metadata["ionmode"].lower()
        else:
            self.metadata["ionmode"] = 'n/a'

    def clone(self):
        return Spectrum(mz=self.mz.copy(),
                        intensities=self.intensities.copy(),
                        metadata=self.metadata.copy())

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
