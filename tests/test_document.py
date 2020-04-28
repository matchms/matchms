import pytest
from matchms.similarity.spec2vec.Document import Document


def test_document_init():
    obj = "asdasd"
    document = Document(obj=obj)
    assert len(document) == 0


def test_document_raises_stop_iteration():
    obj = "asdasd"
    document = Document(obj=obj)
    with pytest.raises(StopIteration):
        next(document)
