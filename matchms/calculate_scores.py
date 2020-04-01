from .Scores import Scores

def calculate_scores(measured_spectrums, reference_spectrums,
                     harmonization_functions, similarity_functions):

    return Scores(measured_spectrums=measured_spectrums,
                  reference_spectrums=reference_spectrums,
                  harmonization_functions=harmonization_functions,
                  similarity_functions=similarity_functions)._calculate()
