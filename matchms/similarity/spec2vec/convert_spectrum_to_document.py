from .Document import Document
from matchms.typing import SpectrumType


def convert_spectrum_to_document(spectrum: SpectrumType, n_decimals=1):

    format_string = "peak@{:." + "{}".format(n_decimals) + "f}"
    words = [format_string.format(i) for i in spectrum.mz]

    return Document(obj=spectrum, words=words)
