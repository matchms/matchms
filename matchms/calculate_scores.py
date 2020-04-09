from .Scores import Scores


def calculate_scores(measured_spectrum,
                     reference_spectrums,
                     similarity_functions):
    """An example docstring for a unbound function."""

    return Scores(measured_spectrum=measured_spectrum,
                  reference_spectrums=reference_spectrums,
                  similarity_functions=similarity_functions).calculate()
