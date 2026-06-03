import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.filtering import make_charge_int
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "input_charge, corrected_charge",
    [
        ("+1", 1),
        ("1", 1),
        (" 1 ", 1),
        ("-2", -2),
        ([-1, "stuff"], -1),
        (["-3"], -3),
        ("0", 0),
        ("n/a", "n/a"),
        ("2+", 2),
        ("2-", -2),
    ],
)
def test_make_charge_int(input_charge, corrected_charge, as_collection):
    """Test if example inputs are correctly converted."""
    mz = np.array([100, 200.0])
    intensities = np.array([0.7, 0.1])
    metadata = {"charge": input_charge}
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(mz)
        .with_intensities(intensities)
        .with_metadata(metadata)
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        make_charge_int,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("charge") == corrected_charge, "Expected different charge integer"


def test_make_charge_int_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"charge": "+1"}).build(),
            SpectrumBuilder().with_metadata({"charge": "2-"}).build(),
            SpectrumBuilder().with_metadata({"charge": "n/a"}).build(),
        ]
    )

    processed = make_charge_int(collection)

    assert processed is not collection
    assert processed.metadata.loc[0, "charge"] == 1
    assert processed.metadata.loc[1, "charge"] == -2
    assert processed.metadata.loc[2, "charge"] == "n/a"


def test_make_charge_int_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"charge": "+1"}).build(),
        ]
    )

    processed = make_charge_int(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "charge"] == 1


def test_make_charge_int_empty_spectrum():
    assert make_charge_int(None) is None