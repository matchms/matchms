import os
import pytest
from matchms.filtering import add_losses
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.importing import load_from_msp
from matchms.importing import load_from_mgf

def test_load_from_msp():

    def apply_my_filters(s):
        """This is how one would typically design a desired pre- and post-
        processing pipeline."""
        s = default_filters(s)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
        s = require_minimum_number_of_peaks(s, n_required=10)
        return s

    module_root = os.path.join(os.path.dirname(__file__), "..")
    # spectrums_file = os.path.join(module_root, "tests", "pesticides.mgf")
    spectrums_file = os.path.join(module_root, "tests", "MoNA-export-GC-MS.msp")

    # apply my filters to the data
    # spectrums = load_from_mgf(spectrums_file)
    spectrums = load_from_msp(spectrums_file)
    # spectrums = [apply_my_filters(s) for s in load_from_msp(spectrums_file)]
    # print(spectrums)s


    # for x in range(10):
    #     print(spectrums[x]['peaks'])


test_load_from_msp()