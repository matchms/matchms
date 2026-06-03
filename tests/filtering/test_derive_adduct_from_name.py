import pytest
from testfixtures import LogCapture
from matchms import SpectraCollection
from matchms.filtering import derive_adduct_from_name
from matchms.filtering.metadata_processing.derive_adduct_from_name import _looks_like_adduct
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def matchms_info_logger():
    """Temporarily set the matchms logger to INFO and always reset it afterwards."""
    set_matchms_logger_level("INFO")
    yield
    reset_matchms_logger()


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, remove_adduct_from_name, expected_adduct, expected_name, expected_removed_adduct",
    [
        ({"compound_name": "peptideXYZ [M+H+K]"}, True, "[M+H+K]", "peptideXYZ", "[M+H+K]"),
        ({"compound_name": "GalCer(d18:2/16:1); [M+H]+"}, True, "[M+H]+", "GalCer(d18:2/16:1)", "[M+H]+"),
        ({"compound_name": "peptideXYZ [M+H+K]", "adduct": "M+H"}, True, "M+H", "peptideXYZ", "[M+H+K]"),
        ({"compound_name": "peptideXYZ [M+H+K]"}, False, "[M+H+K]", "peptideXYZ [M+H+K]", None),
        ({"Name": "peptideXYZ [M+H+K]", "adduct": "M+H"}, True, "M+H", "peptideXYZ", "[M+H+K]"),
        ({"compound_name": "peptideXYZ [M+H+K] C16H12"}, True, "[M+H+K]", "peptideXYZ C16H12", "[M+H+K]"),
        ({"name": ""}, True, None, "", None),
    ],
)
def test_derive_adduct_from_name_parametrized(
    metadata,
    remove_adduct_from_name,
    expected_adduct,
    expected_name,
    expected_removed_adduct,
    matchms_info_logger,
    as_collection,
):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    original_name = spectrum_in.get("compound_name")
    original_adduct = spectrum_in.get("adduct")

    with LogCapture("matchms") as log:
        spectrum = run_filter_as_spectrum_or_collection(
            derive_adduct_from_name,
            spectrum_in,
            as_collection,
            remove_adduct_from_name=remove_adduct_from_name,
        )

    assert spectrum.get("adduct") == expected_adduct
    assert spectrum.get("compound_name") == expected_name

    expected_log = []
    if expected_removed_adduct is not None and expected_name != original_name:
        expected_log.append(
            (
                "matchms",
                "INFO",
                f"Removed adduct ['{expected_removed_adduct}'] from compound name.",
            )
        )

    if expected_adduct != original_adduct:
        expected_log.append(
            (
                "matchms",
                "INFO",
                f"Added adduct {expected_adduct} from the compound name to metadata.",
            )
        )

    log.check(*expected_log)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected_adduct, expected_name",
    [
        # Test removing two adducts and selecting the interpretable one.
        [{"compound_name": "peptideXYZ [M+H+K] C16H12 [M+H+K]2+"}, "[M+H+K]2+", "peptideXYZ C16H12"],
        # Test removing two adducts and not adding if not interpretable.
        [{"compound_name": "peptideXYZ [M+H+K] C16H12 [M+2H+K]"}, None, "peptideXYZ C16H12"],
        # Test removing two adducts and check that clean adduct is run.
        [{"compound_name": "peptideXYZ [M+H+K] C16H12 [M+H]"}, "[M+H]+", "peptideXYZ C16H12"],
    ],
)
def test_derive_adduct_from_name_multiple_adducts_in_name(
    metadata,
    expected_adduct,
    expected_name,
    as_collection,
):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        derive_adduct_from_name,
        spectrum_in,
        as_collection,
        remove_adduct_from_name=True,
    )

    assert spectrum.get("adduct") == expected_adduct, "Expected different adduct."
    assert spectrum.get("compound_name") == expected_name, "Expected different cleaned name."


def test_derive_adduct_from_name_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"compound_name": "peptideXYZ [M+H]+"}).build(),
            SpectrumBuilder().with_metadata({"compound_name": "name without adduct"}).build(),
            SpectrumBuilder().with_metadata({"compound_name": "lipid [M-H]-"}).build(),
        ]
    )

    processed = derive_adduct_from_name(collection)

    assert processed is not collection
    assert processed.metadata.loc[0, "adduct"] == "[M+H]+"
    assert processed.metadata.loc[0, "compound_name"] == "peptideXYZ"
    assert processed.metadata.loc[1, "compound_name"] == "name without adduct"
    assert processed.metadata.loc[2, "adduct"] == "[M-H]-"
    assert processed.metadata.loc[2, "compound_name"] == "lipid"


def test_derive_adduct_from_name_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"compound_name": "peptideXYZ [M+H]+"}).build(),
        ]
    )

    processed = derive_adduct_from_name(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "adduct"] == "[M+H]+"
    assert collection.metadata.loc[0, "compound_name"] == "peptideXYZ"


def test_derive_adduct_from_name_empty_spectrum():
    assert derive_adduct_from_name(None) is None


@pytest.mark.parametrize(
    "adduct",
    [
        "M+",
        "M*+",
        "M+Cl",
        "[M+H]",
        "[2M+Na]+",
        "M+H+K",
        "[2M+ACN+H]+",
        "MS+Na",
        "MS+H",
        "M3Cl37+Na",
        "[M+H+H2O]",
    ],
)
def test_looks_like_adduct_accepts_adduct_like_strings(adduct):
    assert _looks_like_adduct(adduct)


@pytest.mark.parametrize(
    "adduct",
    [
        "N+",
        "B*+",
        "++",
        "--",
        "[--]",
        "H+M+K",
    ],
)
def test_looks_like_adduct_rejects_non_adduct_like_strings(adduct):
    assert not _looks_like_adduct(adduct)