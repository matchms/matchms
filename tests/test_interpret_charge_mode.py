import os


from matchms.importing import load_from_mgf
from matchms.filtering import interpret_charge_mode


def test_interpret_charge_mode():
    """Draft for test.
    """
    module_root = os.path.join(os.path.dirname(__file__), '..')
    yaml_file = os.path.join(module_root, 'matchms',
                             'filtering', 'known_adducts.yaml')

    def apply_filters(s):
        interpret_charge_mode(s, yaml_file)

    # Loading
    references_file = os.path.join(module_root, 'tests', 'testdata.mgf')

    reference_spectrums_raw = load_from_mgf(references_file)
    reference_spectrums = [s.clone() for s in reference_spectrums_raw]

    query_spectrum_raw = reference_spectrums_raw[0]
    query_spectrum = query_spectrum_raw.clone()

    # Filtering
    for s in reference_spectrums:
        apply_filters(s)

    apply_filters(query_spectrum)
    assert query_spectrum_raw.metadata["ionmode"] == 'n/a', 'expected ionmode: "n/a"'
    assert query_spectrum.metadata["ionmode"] == 'positive', 'expected other ionmode'


if __name__ == '__main__':
    test_interpret_charge_mode()
