class Scores:
    def __init__(self, measured_spectrums, reference_spectrums, harmonizations, similarity_functions):
        self.measured_spectrums = measured_spectrums
        self.reference_spectrums = reference_spectrums
        self.harmonizations = harmonizations
        self.similiarity_functions = similiarity_functions
        self.scores = None
        self._calculate()


    def _calculate(self):

        for measured_spectrum in self.measured_spectrums:
            for reference_spectrum in self.reference_spectrums:
                pass
