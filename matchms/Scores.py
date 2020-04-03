import numpy


class Scores:
    def __init__(self, measured_spectrum, reference_spectrums, similarity_functions, scores=None):
        self.measured_spectrum = measured_spectrum
        self.reference_spectrums = reference_spectrums
        self.similarity_functions = similarity_functions
        if scores is None:
            self.scores = numpy.empty([len(self.reference_spectrums),
                                       len(self.similarity_functions)])
        else:
            self.scores = scores

    def __str__(self):
        return self.scores.__str__()

    def calculate(self):
        for i_ref, reference_spectrum in enumerate(self.reference_spectrums):
            for i_meas, measured_spectrum in enumerate(self.measured_spectrums):
                for i_simfun, simfun in enumerate(self.similarity_functions):
                    self.scores[i_ref][i_meas][i_simfun] = simfun(measured_spectrum, reference_spectrum)
        return self
