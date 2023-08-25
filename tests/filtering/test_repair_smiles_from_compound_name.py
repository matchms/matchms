import csv
import os
import pytest
from matchms.filtering import repair_smiles_from_compound_name
from matchms.filtering.metadata_processing.repair_smiles_from_compound_name import (
    _get_compound_name_annotation, _load_compound_name_annotations,
    _pubchem_name_search, _write_compound_name_annotations)
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture()
def csv_file_annotated_compound_names(tmp_path):
    data = [
        ["compound_name", "smiles", "inchi", "inchikey", "monoisotopic_mass"],
        ["compound_1", "smile_1", "inchi_1", 'inchikey_1', 100],
        ["compound_2", None, None, None, None]]
    csv_file_name = os.path.join(tmp_path, "annotated_compounds.csv")
    with open(csv_file_name, "w", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerows(data)
    return csv_file_name


@pytest.fixture()
def csv_file_with_real_compound_names(tmp_path):
    csv_file_name = os.path.join(tmp_path, "expected_compound_annotation.csv")
    expected_result = r"""compound_name,smiles,inchi,inchikey,monoisotopic_mass
    PC(18:0/20:4),CCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC)COP(=O)([O-])OCC[N+](C)(C)C,"InChI=1S/C46H84NO8P/c1-6-8-10-12-14-16-18-20-22-23-25-26-28-30-32-34-36-38-45(48)52-42-44(43-54-56(50,51)53-41-40-47(3,4)5)55-46(49)39-37-35-33-31-29-27-24-21-19-17-15-13-11-9-7-2/h14,16,20,22,25-26,30,32,44H,6-13,15,17-19,21,23-24,27-29,31,33-43H2,1-5H3/b16-14-,22-20-,26-25-,32-30-/t44-/m1/s1",DNYKSJQVBCVGOF-LCKGXUDJSA-N,809.59345564
    fructose,C,"InChI=1S/C6H12O6/c7-2-6(11)5(10)4(9)3(8)1-12-6/h3-5,7-11H,1-2H2/t3-,4-,5+,6?/m1/s1",LKDRXBCSQODPBY-VRPWFDPXSA-N,180.0633881
    glucose,C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O,"InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1",WQZGKKKJIJFFOK-GASJEMHNSA-N,180.0633881
    this compound does not exist,,,,
    galactose,C([C@@H]1[C@@H]([C@@H]([C@H](C(O1)O)O)O)O)O,"InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3+,4+,5-,6?/m1/s1",WQZGKKKJIJFFOK-SVZMEOIVSA-N,180.0633881
    """
    with open(csv_file_name, "w", encoding="utf8") as f:
        f.write(expected_result)
    return csv_file_name


def test_repair_smiles_from_compound_name_skip_already_correct():
    """Tests if already correct annotations are not repaired (even if the compound name does not match)"""
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"compound_name": "compound_1",
                                         "parent_mass": 100.01,
                                         "smiles": "C1CSSC1CCCCC(=O)O",
                                         "inchi": "InChI=1S/C8H14O2S2/c9-8(10)4-2-1-3-7-5-6-11-12-7/h7H,1-6H2,(H,9,10)",
                                         "inchikey": "AGBQKNBQESQNJD-UHFFFAOYSA-N"}).build()
    spectrum = repair_smiles_from_compound_name(spectrum_in, mass_tolerance=0.1)
    assert spectrum_in == spectrum


@pytest.mark.parametrize("compound_name, smiles, parent_mass, expected_smiles", [
    ("PC(18:0/20:4)", "wrong_smiles",
     809.593, "CCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC)COP(=O)([O-])OCC[N+](C)(C)C"),
    ("glucose", "input_smile_1", 180.0633881, 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O'),
    ("this compound does not exist", None, 200.01, None),
    ("also_does_not_exist_and_not_in_csv", None, 100.01, None),
])
def test_repair_smiles_from_compound_name(compound_name, parent_mass, smiles,
                                          expected_smiles,
                                          csv_file_with_real_compound_names, tmp_path):
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"compound_name": compound_name,
                                         "parent_mass": parent_mass,
                                         "smiles": smiles}).build()
    spectrum = repair_smiles_from_compound_name(spectrum_in, csv_file_with_real_compound_names, mass_tolerance=0.1)
    assert spectrum.get("smiles") == expected_smiles, "Expected different smiles."
    # Run without csv file
    spectrum = repair_smiles_from_compound_name(spectrum_in, mass_tolerance=0.1)
    assert spectrum.get("smiles") == expected_smiles, "Expected different smiles."
    # Run with empty csv file
    empty_csv_file = os.path.join(tmp_path, "test.csv")
    spectrum = repair_smiles_from_compound_name(spectrum_in, empty_csv_file, mass_tolerance=0.1)
    assert spectrum.get("smiles") == expected_smiles, "Expected different smiles."
    stored_in_csv_file = _load_compound_name_annotations(empty_csv_file, compound_name)
    assert len(stored_in_csv_file) == 1
    assert stored_in_csv_file[0]["smiles"] == expected_smiles
    assert stored_in_csv_file[0]["compound_name"] == compound_name


def test_write_compound_names_to_file(tmp_path):
    csv_file_name = os.path.join(tmp_path, "compound_annotation.csv")
    annotation_1 = [{'compound_name': 'glucose', 'smiles': 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
                  'inchi': 'InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1',
                  'inchikey': 'WQZGKKKJIJFFOK-GASJEMHNSA-N', 'monoisotopic_mass': 180.06338810},
                    {'compound_name': 'glucose', 'smiles': 'test_smiles',
                     'inchi': 'test_inchi',
                     'inchikey': 'test_inchikey', 'monoisotopic_mass': 10}
                    ]
    _write_compound_name_annotations(csv_file_name, annotation_1)
    # Run a second time to make sure an alrady existing file can be reused
    annotation_2 = [{'compound_name': 'compound_2', 'smiles': None,
                     'inchi': None, 'inchikey': None, 'monoisotopic_mass': None}]
    _write_compound_name_annotations(csv_file_name,
                                     annotation_2)
    assert _load_compound_name_annotations(csv_file_name, "glucose") == annotation_1
    assert _load_compound_name_annotations(csv_file_name, "compound_2") == annotation_2


@pytest.mark.parametrize("compound_name, expected_output", [
    ("compound_1", [{'compound_name': 'compound_1', 'smiles': 'smile_1',
                     'inchi': 'inchi_1',
                     'inchikey': 'inchikey_1', 'monoisotopic_mass': 100}]),
    ("compound_2", [{'compound_name': 'compound_2', 'smiles': None,
                     'inchi': None,
                     'inchikey': None, 'monoisotopic_mass': None}]), ])
def test_load_compound_name_annotation(compound_name, expected_output, csv_file_annotated_compound_names):
    result = _load_compound_name_annotations(csv_file_annotated_compound_names, compound_name)
    assert result == expected_output


@pytest.mark.parametrize("compound_name, expected_output", [
    ("glucose", [{'compound_name': 'glucose', 'smiles': 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
                  'inchi': 'InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1',
                  'inchikey': 'WQZGKKKJIJFFOK-GASJEMHNSA-N', 'monoisotopic_mass': 180.06338810}]),
    ("does_not_exist", [])])
def test_pubchem_name_search(compound_name, expected_output):
    result = _pubchem_name_search(compound_name)
    assert result == expected_output


@pytest.mark.parametrize("compound_name, expected_output", [
    ("compound_1", [{'compound_name': 'compound_1', 'smiles': 'smile_1',
                     'inchi': 'inchi_1',
                     'inchikey': 'inchikey_1', 'monoisotopic_mass': 100}]),
    ("compound_2", [{'compound_name': 'compound_2', 'smiles': None,
                     'inchi': None,
                     'inchikey': None, 'monoisotopic_mass': None}]),
    ("glucose", [{'compound_name': 'glucose', 'smiles': 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
                  'inchi': 'InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1',
                  'inchikey': 'WQZGKKKJIJFFOK-GASJEMHNSA-N', 'monoisotopic_mass': 180.06338810}]),
    ("does_not_exist", [])])
def test_get_compound_name_annotation(compound_name, expected_output, csv_file_annotated_compound_names):
    result = _get_compound_name_annotation(compound_name, csv_file_annotated_compound_names)
    assert result == expected_output
