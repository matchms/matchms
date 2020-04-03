class SimMeas2:

    def __init__(self, label, factor=0.3):
        self.label = label
        self.factor = factor

    def __call__(self, spectrum, reference_spectrum):
        # just some nonsense to demonstrate usage of a parameter
        # when comparing 2 spectrums
        return (min(spectrum.mz) + min(reference_spectrum.mz)) * self.factor
