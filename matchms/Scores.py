import numpy

class Scores:
    def __init__(self, measured_spectrums, reference_spectrums, harmonization_functions, similarity_functions):
        self.measured_spectrums = measured_spectrums
        self.reference_spectrums = reference_spectrums
        self.harmonization_functions = harmonization_functions
        self.similarity_functions = similarity_functions
        self.scores = numpy.empty([len(self.reference_spectrums),
                                   len(self.measured_spectrums),
                                   len(self.similarity_functions)])

    def _calculate(self):
        for i_ref, reference_spectrum in enumerate(self.reference_spectrums):
            for i_meas, measured_spectrum in enumerate(self.measured_spectrums):
                for i_simfun, simfun in enumerate(self.similarity_functions):
                    self.scores[i_ref][i_meas][i_simfun] = simfun(measured_spectrum, reference_spectrum)
        return self
