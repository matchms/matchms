import os
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from matchms.filtering.list_of_spectra_filters.get_comound_from_pubchem import (
    annotate_with_pubchem_wrapper, write_compound_names_to_file)


@pytest.fixture()
def csv_file_with_compound_names(tmp_path):
    csv_file_name = os.path.join(tmp_path, "expected_compound_annotation.csv")
    expected_result = r""",compound_name,smiles,inchi,inchikey,monoisotopic_mass
    0,PC(18:0/20:4),CCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC)COP(=O)([O-])OCC[N+](C)(C)C,"InChI=1S/C46H84NO8P/c1-6-8-10-12-14-16-18-20-22-23-25-26-28-30-32-34-36-38-45(48)52-42-44(43-54-56(50,51)53-41-40-47(3,4)5)55-46(49)39-37-35-33-31-29-27-24-21-19-17-15-13-11-9-7-2/h14,16,20,22,25-26,30,32,44H,6-13,15,17-19,21,23-24,27-29,31,33-43H2,1-5H3/b16-14-,22-20-,26-25-,32-30-/t44-/m1/s1",DNYKSJQVBCVGOF-LCKGXUDJSA-N,809.59345564
    1,fructose,C1[C@H]([C@H]([C@@H](C(O1)(CO)O)O)O)O,"InChI=1S/C6H12O6/c7-2-6(11)5(10)4(9)3(8)1-12-6/h3-5,7-11H,1-2H2/t3-,4-,5+,6?/m1/s1",LKDRXBCSQODPBY-VRPWFDPXSA-N,180.0633881
    2,glucose,C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O,"InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1",WQZGKKKJIJFFOK-GASJEMHNSA-N,180.0633881
    3,this compound does not exist,,,,
    4,galactose,C([C@@H]1[C@@H]([C@@H]([C@H](C(O1)O)O)O)O)O,"InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3+,4+,5-,6?/m1/s1",WQZGKKKJIJFFOK-SVZMEOIVSA-N,180.0633881
    """
    with open(csv_file_name, "w", encoding="utf8") as f:
        f.write(expected_result)
    return csv_file_name


def test_write_compound_names_to_file(tmp_path, csv_file_with_compound_names):
    csv_file_name = os.path.join(tmp_path, "compound_annotation.csv")
    result = write_compound_names_to_file(["glucose", "fructose", 1234, "PC(18:0/20:4)", "this compound does not exist"],
                                          csv_file_name)
    expected_results = pd.read_csv(csv_file_with_compound_names, index_col=0)
    pd.testing.assert_frame_equal(expected_results.iloc[:4], result)
    # Run a second time to make sure an alrady existing file can be (partly) reused
    result_2nd_time = write_compound_names_to_file(["glucose", "fructose", "PC(18:0/20:4)", "this compound does not exist", "galactose"],
                                                   csv_file_name)
    pd.testing.assert_frame_equal(expected_results, result_2nd_time)


def two_test_spectra():
    """Returns a list with two spectra

    The spectra are created by using peaks from the first two spectra in
    100_test_spectra.pickle, to make sure that the peaks occur in the s2v
    model. The other values are random.
    """
    spectrum1 = Spectrum(mz=np.array([100.0], dtype="float"),
                         intensities=np.array([1., ], dtype="float"),
                         metadata={'parent_mass': 180,
                                   'compound_name': "fructose"})
    spectrum2 = Spectrum(mz=np.array([100.0], dtype="float"),
                         intensities=np.array([1., ], dtype="float"),
                         metadata={'parent_mass': 100,
                                   'compound_name': "does not exist"})

    spectra = [spectrum1, spectrum2]
    return spectra


def test_annotate_with_pubchem_wrapper(csv_file_with_compound_names):
    pytest.importorskip("rdkit")
    test_spectra = two_test_spectra()
    annotated_spectra, newly_annotated_spectra, not_annotated_spectra = annotate_with_pubchem_wrapper(test_spectra,
                                                                                                      csv_file_with_compound_names)
    assert len(annotated_spectra) == 0
    assert len(newly_annotated_spectra) == 1
    assert len(not_annotated_spectra) == 1
    assert test_spectra[1] == not_annotated_spectra[0]
    assert newly_annotated_spectra[0].get("inchikey") is not None
    assert newly_annotated_spectra[0].get("inchi") is not None
    assert newly_annotated_spectra[0].get("smiles") is not None
