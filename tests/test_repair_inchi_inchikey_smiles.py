from matchms.filtering import repair_inchi_inchikey_smiles
from matchms import Spectrum
import numpy


def test_repair_inchi_inchikey_smiles_clean_inchi_entered_as_inchi():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"
    assert spectrum.get("inchikey") == ""
    assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_clean_inchi_entered_as_inchikey():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"
    assert spectrum.get("inchikey") == ""
    assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_clean_inchi_entered_as_smiles():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"
    assert spectrum.get("inchikey") == ""
    assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_clean_inchikey_entered_as_inchi():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)

    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == ""
    assert spectrum.get("inchikey") == "ABTNALLHJFCFRZ-UHFFFAOYSA-N"
    assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_clean_inchikey_entered_as_inchikey():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == ""
    assert spectrum.get("inchikey") == "ABTNALLHJFCFRZ-UHFFFAOYSA-N"
    assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_clean_inchikey_entered_as_smiles():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)

    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == ""
    assert spectrum.get("inchikey") == "ABTNALLHJFCFRZ-UHFFFAOYSA-N"
    assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_clean_smiles_entered_as_inchi():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "C[C@H](Cc1ccccc1)N(C)CC#C"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == ""
    assert spectrum.get("inchikey") == ""
    assert spectrum.get("smiles") == "C[C@H](Cc1ccccc1)N(C)CC#C"


def test_repair_inchi_inchikey_smiles_clean_smiles_entered_as_inchikey():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "C[C@H](Cc1ccccc1)N(C)CC#C"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == ""
    assert spectrum.get("inchikey") == ""
    assert spectrum.get("smiles") == "C[C@H](Cc1ccccc1)N(C)CC#C"


def test_repair_inchi_inchikey_smiles_clean_smiles_entered_as_smiles():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "C[C@H](Cc1ccccc1)N(C)CC#C"})

    spectrum = repair_inchi_inchikey_smiles(spectrum_in)
    assert spectrum is not spectrum_in
    assert spectrum.get("inchi") == ""
    assert spectrum.get("inchikey") == ""
    assert spectrum.get("smiles") == "C[C@H](Cc1ccccc1)N(C)CC#C"
