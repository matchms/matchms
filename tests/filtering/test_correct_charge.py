import pytest
from matchms.filtering import correct_charge
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{}, 0],
        [{"ionmode": "positive"}, 1],
        [{"ionmode": "negative"}, -1],
        [{"ionmode": "positive", "charge": 0}, 1],
        [{"ionmode": "negative", "charge": 0}, -1],
        [{"ionmode": "positive", "charge": 2}, 2],
        [{"ionmode": "negative", "charge": -2}, -2],
        [{"ionmode": "positive", "charge": -2}, 2],
        [{"ionmode": "negative", "charge": 2}, -2],
        [{"charge": 3}, 3],
        [{"charge": -3}, -3],
        [{"charge": 0}, 0],
    ],
)
def test_correct_charge(metadata, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        correct_charge,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("charge") == expected


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_correct_charge_raises_for_non_lowercase_ionmode(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"ionmode": "Positive"}).build()

    with pytest.raises(
        ValueError,
        match="Ionmode field not harmonized",
    ):
        run_filter_as_spectrum_or_collection(
            correct_charge,
            spectrum_in,
            as_collection,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_correct_charge_raises_for_string_charge(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"charge": "+1"}).build()

    with pytest.raises(
        ValueError,
        match="Charge is given as string",
    ):
        run_filter_as_spectrum_or_collection(
            correct_charge,
            spectrum_in,
            as_collection,
        )


def test_correct_charge_clone_true_does_not_modify_input_spectrum():
    spectrum_in = SpectrumBuilder().with_metadata({"ionmode": "positive"}).build()

    spectrum = correct_charge(spectrum_in, clone=True)

    assert spectrum is not spectrum_in
    assert spectrum.get("charge") == 1
    assert spectrum_in.get("charge") is None


def test_correct_charge_clone_false_modifies_input_spectrum():
    spectrum_in = SpectrumBuilder().with_metadata({"ionmode": "positive"}).build()

    spectrum = correct_charge(spectrum_in, clone=False)

    assert spectrum is spectrum_in
    assert spectrum_in.get("charge") == 1


def test_correct_charge_empty_spectrum():
    spectrum = correct_charge(None)

    assert spectrum is None
