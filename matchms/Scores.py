import numpy


class Scores:
    """An example docstring for a class definition."""
    def __init__(self, measured_spectrum, reference_spectrums, similarity_functions, scores=None):
        """An example docstring for a constructor."""
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
        """An example docstring for a method."""
        for i_ref, reference_spectrum in enumerate(self.reference_spectrums):
            for i_simfun, simfun in enumerate(self.similarity_functions):
                self.scores[i_ref][i_simfun] = simfun(self.measured_spectrum, reference_spectrum)
        return self

    def sort_by(self, label, kind='quicksort'):
        """An example docstring for a method."""
        found = False
        i_simfun = None
        for i_simfun, simfun in enumerate(self.similarity_functions):
            if simfun.label == label:
                found = True
                break

        assert found, "Label '{0}' not found in similarity functions.".format(label)
        axis = 0
        row_numbers = self.scores[:, i_simfun].argsort(axis=axis, kind=kind)

        reference_spectrums = self.reference_spectrums
        scores = self.scores[row_numbers, :]
        return Scores(measured_spectrum=self.measured_spectrum,
                      reference_spectrums=reference_spectrums,
                      similarity_functions=self.similarity_functions,
                      scores=scores)

    def top(self, n):
        """An example docstring for a method."""
        reference_spectrums = [s.clone() for s in self.reference_spectrums[:n]]
        return Scores(measured_spectrum=self.measured_spectrum,
                      reference_spectrums=reference_spectrums,
                      similarity_functions=self.similarity_functions,
                      scores=self.scores.copy()[:n])

    def reverse(self):
        """An example docstring for a method."""
        scores = self.scores[::-1, :]
        reference_spectrums = self.reference_spectrums[::-1]
        return Scores(measured_spectrum=self.measured_spectrum,
                      reference_spectrums=reference_spectrums,
                      similarity_functions=self.similarity_functions,
                      scores=scores)
