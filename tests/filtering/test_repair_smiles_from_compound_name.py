import os
import csv
import pytest
from matchms.filtering import repair_smiles_from_compound_name
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture()
def csv_file_annotated_compound_names(tmp_path):
    data = [
        ["compound_name", "smiles", "inchi", "inchikey", "monoisotopic_mass"],
        ["compound_1", "smile_1", "inchi_1", 'inchikey_1', 100],
        ["compound_2", None, None, None, None]]
    csv_file_name = os.path.join(tmp_path, "annotated_compounds.csv")
    with open(csv_file_name, "w") as csv_file:
        csv.writer(csv_file).writerows(data)
    return csv_file_name


@pytest.mark.parametrize("compound_name, parent_mass, expected_smiles", [
    ("compound_1", 100.01, 'smile_1'),
    ("compound_2", 100.01, 'test_smile'),
    ("compound_1", 100.5, 'test_smile'),
])
def test_clean_compound_name(compound_name, parent_mass, expected_smiles, csv_file_annotated_compound_names):
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"compound_name": compound_name,
                                         "parent_mass": parent_mass,
                                         "smiles": "test_smile"}).build()
    spectrum = repair_smiles_from_compound_name(spectrum_in, csv_file_annotated_compound_names, mass_tolerance=0.1)

    assert spectrum.get("smiles") == expected_smiles, "Expected different smiles."
