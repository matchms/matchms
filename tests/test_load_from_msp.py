import os
import pytest
from matchms.importing import load_from_msp

def test_load_from_msp():

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "MoNA-export-GC-MS.msp")

    spectrums = load_from_msp(spectrums_file)
    print(spectrums)


test_load_from_msp()