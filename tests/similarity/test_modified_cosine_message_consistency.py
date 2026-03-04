import numpy as np
import pytest
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from matchms.similarity import ModifiedCosineGreedy, ModifiedCosineHungarian
from ..builder_Spectrum import SpectrumBuilder


EXPECTED_MISSING_PRECURSOR = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
EXPECTED_PRECURSOR_TYPE_WARNING = "Precursor_mz must be int or float. Apply 'add_precursor_mz' filter first."


@pytest.mark.parametrize("scenario", ["missing_precursor", "string_precursor"])
def test_modified_cosine_precursor_messages_consistent(scenario, caplog):
    """Modified cosine variants should emit identical precursor messages."""
    classes = [ModifiedCosineGreedy, ModifiedCosineHungarian]
    collected_messages = []

    if scenario == "missing_precursor":
        mz = np.array([100, 150, 200], dtype="float")
        intensities = np.array([700, 200, 100], dtype="float")
        builder = SpectrumBuilder()
        spectrum_1 = builder.with_mz(mz).with_intensities(intensities).build()
        spectrum_2 = builder.with_mz(mz).with_intensities(intensities).build()
        norm_spectrum_1 = normalize_intensities(spectrum_1)
        norm_spectrum_2 = normalize_intensities(spectrum_2)

        for similarity_class in classes:
            similarity = similarity_class()
            with pytest.raises(AssertionError) as msg:
                similarity.pair(norm_spectrum_1, norm_spectrum_2)
            collected_messages.append(str(msg.value))

        assert len(set(collected_messages)) == 1
        assert collected_messages[0] == EXPECTED_MISSING_PRECURSOR
        return

    spectrum_1 = Spectrum(
        mz=np.array([100, 200, 300], dtype="float"),
        intensities=np.array([10, 10, 500], dtype="float"),
        metadata={"precursor_mz": 1000.0},
        metadata_harmonization=False,
    )
    spectrum_2 = Spectrum(
        mz=np.array([120, 220, 320], dtype="float"),
        intensities=np.array([10, 10, 500], dtype="float"),
        metadata={"precursor_mz": "1005.0"},
        metadata_harmonization=False,
    )
    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)

    for similarity_class in classes:
        caplog.clear()
        similarity = similarity_class(tolerance=1.0)
        similarity.pair(norm_spectrum_1, norm_spectrum_2)
        warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert warnings, "Expected at least one precursor warning."
        collected_messages.append(warnings[0])

    assert len(set(collected_messages)) == 1
    assert collected_messages[0] == EXPECTED_PRECURSOR_TYPE_WARNING
