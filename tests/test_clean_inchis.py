import os

from matchms.importing import load_from_mgf
from matchms.filtering import clean_inchis


def test_clean_inchis():
    """Draft for test.
    """
    module_root = os.path.join(os.path.dirname(__file__), '..')

    def apply_filters(s):
        clean_inchis(s)

    # Loading
    references_file = os.path.join(module_root, 'tests', 'testdata01.mgf')

    reference_spectrums_raw = load_from_mgf(references_file)
    reference_spectrums = [s.clone() for s in reference_spectrums_raw]

    query_spectrum_raw = reference_spectrums_raw[0]
    query_spectrum = query_spectrum_raw.clone()

    # Filtering
    for s in reference_spectrums:
        apply_filters(s)

    apply_filters(query_spectrum)
    assert query_spectrum_raw.metadata["inchi"].startswith('InChI='), 'expected different InChI'
    assert query_spectrum.metadata["inchi"].startswith('"InChI='), 'InChI style not as expected.'
    original_inchi = reference_spectrums_raw[2].metadata["inchi"]
    assert original_inchi.startswith('"InChI=CCCCCCCCCCCCCCCC(=O)'), "expected misplaced smiles"
    modified_inchi = reference_spectrums[2].metadata["inchi"]
    assert modified_inchi.startswith('"InChI=1S/C24H50NO7P/'), "inchi was not converted correctly"


if __name__ == '__main__':
    test_clean_inchis()
