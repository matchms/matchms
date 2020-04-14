class Document:
    def __init__(self, spectrum, words):
        self._spectrum = spectrum
        self.words = words
        self._index = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.words)

    def __next__(self):
        """gensim.models.Word2Vec() wants its corpus elements to be iterable"""

        if self._index < len(self.words) - 1:
            self._index += 1
            return self.words[self._index]
        raise StopIteration

    def __str__(self):
        return self.words.__str__()

    @property
    def weights(self):
        """getter method for _spectrum.intensities private variable"""
        return self._spectrum.intensities.copy()
