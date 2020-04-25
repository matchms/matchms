from .Document import Document


def convert_spectrum_to_document(spectrum, n_decimals=1):

    if spectrum is None:
        return None

    format_string = "{}@{:." + "{}".format(n_decimals) + "f}"
    peak_words = [format_string.format("peak", mz) for mz in spectrum.peaks.mz]
    if spectrum.losses is not None:
        loss_words = [format_string.format("loss", mz) for mz in spectrum.losses.mz]
    else:
        loss_words = []

    return Document(obj=spectrum, words=peak_words + loss_words)
