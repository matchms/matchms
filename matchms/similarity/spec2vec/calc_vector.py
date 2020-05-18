import numpy


def calc_vector(model, document, intensity_weighting_power=0):
    word_vectors = model.wv[document.words]
    weights = numpy.asarray(document.weights).reshape(len(document), 1)
    weights_raised = numpy.power(weights, intensity_weighting_power)
    weights_raised_tiled = numpy.tile(weights_raised, (1, model.wv.vector_size))
    vector = numpy.sum(word_vectors * weights_raised_tiled, 0)
    return vector
