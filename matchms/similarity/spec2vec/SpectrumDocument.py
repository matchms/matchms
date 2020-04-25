from .Document import Document


class SpectrumDocument(Document):
    def __init__(self, spectrum, n_decimals=1):
        super().__init__(obj=spectrum)
        self._make_words(n_decimals=n_decimals)

    def _make_words(self, n_decimals=1):
        format_string = "{}@{:." + "{}".format(n_decimals) + "f}"
        peak_words = [format_string.format("peak", mz) for mz in self._obj.peaks.mz]
        if self._obj.losses is not None:
            loss_words = [format_string.format("loss", mz) for mz in self._obj.losses.mz]
        else:
            loss_words = []
        self.words = peak_words + loss_words
        return self
