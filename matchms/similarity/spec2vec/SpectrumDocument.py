from .Document import Document


class SpectrumDocument(Document):
    def __init__(self, spectrum, n_decimals=1):
        self.n_decimals = n_decimals
        self.weights = None
        super().__init__(obj=spectrum)
        self._add_weights()

    def _make_words(self):
        """Create word from peaks (and losses)."""
        format_string = "{}@{:." + "{}".format(self.n_decimals) + "f}"
        peak_words = [format_string.format("peak", mz) for mz in self._obj.peaks.mz]
        if self._obj.losses is not None:
            loss_words = [format_string.format("loss", mz) for mz in self._obj.losses.mz]
        else:
            loss_words = []
        self.words = peak_words + loss_words
        return self

    def _add_weights(self):
        """Add peaks (and loss) intensities as weights."""
        assert self._obj.peaks.intensities.max() <= 1, "peak intensities not normalized"

        peak_intensities = self._obj.peaks.intensities.tolist()
        if self._obj.losses is not None:
            loss_intensities = self._obj.losses.intensities.tolist()
        else:
            loss_intensities = []
        self.weights = peak_intensities + loss_intensities
        return self
