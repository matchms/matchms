import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering import interpret_pepmass
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "input_pepmass, expected_results",
    [
        (None, (None, None, None)),
        (896.05, (896.05, None, None)),
        ((896.05, None), (896.05, None, None)),
        ((896.05, 1111.2, "2-"), (896.05, 1111.2, -2)),
        ((896.05, 1111.2, "2+"), (896.05, 1111.2, 2)),
        ((896.05, 1111.2, -1), (896.05, 1111.2, -1)),
    ],
)
def test_interpret_pepmass(input_pepmass, expected_results, as_collection):
    """Test if example inputs are correctly converted."""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": input_pepmass}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        interpret_pepmass,
        spectrum_in,
        as_collection,
    )

    actual_results = (
        spectrum.get("precursor_mz"),
        spectrum.get("precursor_intensity"),
        spectrum.get("charge"),
    )

    assert actual_results == expected_results, "Expected different 3 values."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_interpret_pepmass_removes_pepmass(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"pepmass": (896.05, 1111.2, "2-")}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        interpret_pepmass,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("pepmass") is None
    assert spectrum.get("precursor_mz") == 896.05
    assert spectrum.get("precursor_intensity") == 1111.2
    assert spectrum.get("charge") == -2


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_interpret_pepmass_charge_present(caplog, as_collection):
    """Test if example inputs are correctly converted when charge already exists."""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": (896.05, 1111.2, "2-"), "charge": -1}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        interpret_pepmass,
        spectrum_in,
        as_collection,
    )

    actual_results = (
        spectrum.get("precursor_mz"),
        spectrum.get("precursor_intensity"),
        spectrum.get("charge"),
    )

    assert actual_results == (896.05, 1111.2, -2), "Expected different 3 values."
    assert "Overwriting existing charge -1 with new one: -2" in caplog.text


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_interpret_pepmass_mz_present(caplog, as_collection):
    """Test if example inputs are correctly converted when precursor_mz already exists."""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": (203, 44, "2-"), "precursor_mz": 202}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        interpret_pepmass,
        spectrum_in,
        as_collection,
    )

    actual_results = (
        spectrum.get("precursor_mz"),
        spectrum.get("precursor_intensity"),
        spectrum.get("charge"),
    )

    assert actual_results == (203, 44, -2), "Expected different 3 values."
    assert "Overwriting existing precursor_mz 202 with new one: 203" in caplog.text


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_interpret_pepmass_intensity_present(caplog, as_collection):
    """Test if example inputs are correctly converted when precursor_intensity already exists."""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": (203, 44, "2-"), "precursor_intensity": 100}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        interpret_pepmass,
        spectrum_in,
        as_collection,
    )

    actual_results = (
        spectrum.get("precursor_mz"),
        spectrum.get("precursor_intensity"),
        spectrum.get("charge"),
    )

    assert actual_results == (203, 44, -2), "Expected different 3 values."
    assert "Overwriting existing precursor_intensity 100 with new one: 44" in caplog.text


def test_interpret_pepmass_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"pepmass": (896.05, 1111.2, "2-")}).build(),
            SpectrumBuilder().with_metadata({"pepmass": "100.2"}).build(),
            SpectrumBuilder().with_metadata({"compound_name": "no pepmass"}).build(),
        ]
    )

    processed = interpret_pepmass(collection)

    assert processed is not collection

    assert processed.metadata.loc[0, "precursor_mz"] == 896.05
    assert processed.metadata.loc[0, "precursor_intensity"] == 1111.2
    assert processed.metadata.loc[0, "charge"] == -2

    assert processed.metadata.loc[1, "precursor_mz"] == 100.2
    assert pd.isna(processed.metadata.loc[1, "precursor_intensity"])
    assert pd.isna(processed.metadata.loc[1, "charge"])

    assert "pepmass" not in processed.metadata.columns


def test_interpret_pepmass_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"pepmass": (896.05, 1111.2, "2-")}).build(),
        ]
    )

    processed = interpret_pepmass(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "precursor_mz"] == 896.05
    assert collection.metadata.loc[0, "precursor_intensity"] == 1111.2
    assert collection.metadata.loc[0, "charge"] == -2
    assert "pepmass" not in collection.metadata.columns


def test_interpret_pepmass_empty_spectrum():
    assert interpret_pepmass(None) is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "input_pepmass, expected_results",
    [
        ["(981.54, None)", (981.54, None, None)],
        ["(981.54, 44, -2)", (981.54, 44, -2)],
        ["100.2", (100.2, None, None)],
        ["something_random", (None, None, None)],
    ],
)
def test_interpret_pepmass_error_v0_22_0(input_pepmass, expected_results, as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"PEPMASS": input_pepmass}, metadata_harmonization=True)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        interpret_pepmass,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_mz") == expected_results[0]
    assert spectrum.get("precursor_intensity") == expected_results[1]
    assert spectrum.get("charge") == expected_results[2]


def test_load_pepmass_error_issue_452():
    comment = """
    \"computed [M+Cl]-=539.622034944\"
    \"InChI=InChI=1S/C18H32O16/c19-1-6-9(23)12(26)13(27)16(31-6)34-18(15(29)11(25)8(3-21)33-18)5-30-17(4-22)14(28)10(24)7(2-20)32-17/h6-16,19-29H,1-5H2/t6-,7-,8-,9-,10-,11-,12+,13-,14+,15+,16-,17-,18+/m1/s1\"
    \"computed SMILES=OCC1OC(OC2(OC(CO)C(O)C2O)COC3(OC(CO)C(O)C3O)CO)C(O)C(O)C1O\"
    \"computed [2M+H-H2O]+=991.3306998879999\" \"computed [2M+K]+=1047.436369888\"
    \"computed [M+H]+=505.177004944\" \"computed [2M+Na]+=1031.327839888\"
    \"computed [2M+Cl]-=1043.791069888\" \"computed [2M+HAc-H]-=1067.3820998879999\"
    \"computed [M-H20-H]-=485.15064494399996\" \"computed [M-H]-=503.161758944\"
    \"computed [2M-H]-=1007.3307938879999\" \"computed [M+Na]+=527.1588049439999\"
    \"computed [M+H-H2O]+=487.161664944\" \"computed [2M+NH4]+=1026.376649888\"
    \"computed [M+NH4]+=522.2076149439999\" \"computed [M+K]+=543.267334944\"
    \"computed [2M+H]+=1009.346039888\" \"computed [M+HAc-H]-=563.2130649439999\"
    \"rtinseconds=0.005\" \"pepmass=505.0\" \"sample introduction=Direct Infusion (DI)\"
    \"ionization=Electrospray Ionization (ESI)\" \"author=Biswapriya B. Misra\"
    \"computed spectral entropy=1.0397207708399179\" \"computed normalized entropy=0.946394630357186\"
    \"SPLASH=splash10-0a4i-0003090000-d95999e0141df8aa4d9e\"
    \"submitter=submitter = Biswapriya Misra (Wake Forest School of Medicine)\"
    \"MoNA Rating=2.5\"
    """
    metadata = {
        "Name": "1-Kestose",
        "Synon": "$:00in-source",
        "DB#": "MoNA001675",
        "InChIKey": "VAWYEUIPHLMNNF-OESPXIITSA-N",
        "Spectrum_type": "MS2",
        "Ion_mode": "P",
        "Formula": "C18H32O16",
        "MW": "504",
        "ExactMass": "504.169034944",
        "Comments": comment,
    }

    mz = np.array([387.3, 387.4, 505, 505.1, 505.2], dtype=np.float64)
    ints = np.array([50, 0, 0, 50, 100], dtype=np.float64)

    spectrum = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(ints)
        .with_metadata(metadata, metadata_harmonization=True)
        .build()
    )
    assert spectrum is not None