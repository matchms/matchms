import numpy
from matchms import Spectrum
from matchms.filtering import repair_inchi_inchikey_smiles


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


def test_repair_inchi_inchikey_smiles_various_inchi_entered_as_smiles():
    """Test a wider variety of different inchis."""
    test_inchis = [
        "1S/C4H11N5.ClH/c1-7-3(5)9-4(6)8-2;/h1-2H3,(H5,5,6,7,8,9);1H",
        "InChI=1S/C11H15N3O2.ClH/c1-12-11(15)16-10-6-4-5-9(7-10)13-8-14(2)3;/h4-8H,1-3H3,(H,12,15);1H/b13-8+;",
        '"InChI=1S/C17O8/c1-9-7-12(19)14(16(20)21)13(8-9)25-15(10(2)23-3)11(5-6-18)17(22)24-4"',
        "InChI=1S/CH3/h1H3",
        "1/2C17H18N3O3S.Mg/c2*1-10-8-18-15(11(2)16(10)23-4)9-24(21)17-19-13-6-5-12(22-3)7-14(13)20-17;/h2*5-8H,9H2,1-4H3;/q2*-1;+2"
    ]
    for inchi in test_inchis:
        spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                               intensities=numpy.array([], dtype="float"),
                               metadata={"smiles": inchi})

        spectrum = repair_inchi_inchikey_smiles(spectrum_in)
        assert spectrum is not spectrum_in
        assert spectrum.get("inchi") == "InChI=" + inchi.replace("InChI=", "").replace('"', "")
        assert spectrum.get("inchikey") == ""
        assert spectrum.get("smiles") == ""
