class SimMeas1:

    def __init__(self, factor=0.3):
        self.factor = factor

    def __call__(self, spectrum, reference_spectrum):
        # just some nonsense to demonstrate usage of a parameter
        # when comparing 2 spectrums
        return (max(spectrum.mz) + max(reference_spectrum.mz)) * self.factor
