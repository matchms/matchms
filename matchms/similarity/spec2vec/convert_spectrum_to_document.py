from .Document import Document


def convert_spectrum_to_document(spectrum, n_decimals=1):

    format_string = "peak@{:." + "{}".format(n_decimals) + "f}"
    words = [format_string.format(i) for i in spectrum.peaks.mz]

    return Document(obj=spectrum, words=words)
