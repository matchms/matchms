from pyteomics.mgf import MGF


def load_from_mgf(filename):
    return list(MGF(filename, convert_arrays=1))
