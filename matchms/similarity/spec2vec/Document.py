class Document:
    def __init__(self, obj):
        self._obj = obj
        self._index = 0
        self._make_words()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.words)

    def __next__(self):
        """gensim.models.Word2Vec() wants its corpus elements to be iterable"""
        if self._index < len(self.words):
            word = self.words[self._index]
            self._index += 1
            return word
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self.words.__str__()

    def _make_words(self):
        print("You should override this method in your own subclass.")
        self.words = []
        return self
