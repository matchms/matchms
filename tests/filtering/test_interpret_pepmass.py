import numpy as np
import pytest
from matchms.filtering import interpret_pepmass
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "input_pepmass, expected_results",
    [
        ((None), (None, None, None)),
        ((896.05), (896.05, None, None)),
        ((896.05, None), (896.05, None, None)),
        ((896.05, 1111.2, "2-"), (896.05, 1111.2, -2)),
        ((896.05, 1111.2, "2+"), (896.05, 1111.2, 2)),
        ((896.05, 1111.2, -1), (896.05, 1111.2, -1)),
    ],
)
def test_interpret_pepmass(input_pepmass, expected_results):
    """Test if example inputs are correctly converted"""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": input_pepmass}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == expected_results, "Expected different 3 values."


def test_interpret_pepmass_charge_present(caplog):
    """Test if example inputs are correctly converted when entries already exist"""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": (896.05, 1111.2, "2-"), "charge": -1}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == (896.05, 1111.2, -2), "Expected different 3 values."
    assert "Overwriting existing charge -1 with new one: -2" in caplog.text, "Expected different log message"


def test_interpret_pepmass_mz_present(caplog):
    """Test if example inputs are correctly converted when entries already exist"""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": (203, 44, "2-"), "precursor_mz": 202}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == (203, 44, -2), "Expected different 3 values."
    assert "Overwriting existing precursor_mz 202 with new one: 203" in caplog.text, "Expected different log message"


def test_interpret_pepmass_intensity_present(caplog):
    """Test if example inputs are correctly converted when entries already exist"""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"pepmass": (203, 44, "2-"), "precursor_intensity": 100}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == (203, 44, -2), "Expected different 3 values."
    assert "Overwriting existing precursor_intensity 100 with new one: 44" in caplog.text, (
        "Expected different log message"
    )


def test_empty_spectrum():
    spectrum_in = None
    spectrum = interpret_pepmass(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."


@pytest.mark.parametrize(
    "input_pepmass, expected_results",
    [
        ["(981.54, None)", (981.54, None, None)],
        ["(981.54, 44, -2)", (981.54, 44, -2)],
        ["100.2", (100.2, None, None)],
        ["something_random", (None, None, None)],
    ],
)
def test_interpret_pepmass_error_v0_22_0(input_pepmass, expected_results):
    spectrum = SpectrumBuilder().with_metadata({"PEPMASS": input_pepmass}, metadata_harmonization=True).build()

    assert spectrum.get("precursor_mz") == expected_results[0], "Expected different precursor_mz."
    assert spectrum.get("precursor_intensity") == expected_results[1], "Expected different precursor_intensity."
    assert spectrum.get("charge") == expected_results[2], "Expected different charge."


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
