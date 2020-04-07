import numpy


def select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0):

    assert mz_from <= mz_to, "'mz_from' should be smaller than or equal to 'mz_to'."

    condition = numpy.logical_and(mz_from <= spectrum.mz, spectrum.mz <= mz_to)

    spectrum.mz = spectrum.mz[condition]
    spectrum.intensities = spectrum.intensities[condition]
