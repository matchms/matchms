import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering import derive_inchi_from_smiles
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchi_from_smiles(as_collection):
    """Test if conversion to inchi works when only smiles is given."""
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": "C1CCCCC1"}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        derive_inchi_from_smiles,
        spectrum_in,
        as_collection,
    )

    inchi = spectrum.get("inchi").replace('"', "")
    assert inchi == "InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2", "Expected different InChI"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchi_from_defect_smiles(as_collection):
    """Test if no inchi is derived from invalid smiles."""
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": "CX1CCCCC1"}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        derive_inchi_from_smiles,
        spectrum_in,
        as_collection,
    )

    inchi = spectrum.get("inchi", None)
    assert inchi is None or pd.isna(inchi), "Expected no InChI"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchi_from_smiles_does_not_overwrite_valid_inchi(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "CCCO",
                "inchi": "InChI=1S/CH4/h1H4",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_inchi_from_smiles,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("inchi") == "InChI=1S/CH4/h1H4"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchi_from_smiles_without_smiles_does_nothing(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"inchi": None}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        derive_inchi_from_smiles,
        spectrum_in,
        as_collection,
    )

    inchi = spectrum.get("inchi", None)
    assert inchi is None or pd.isna(inchi)


def test_derive_inchi_from_smiles_empty_spectrum():
    spectrum = derive_inchi_from_smiles(None)

    assert spectrum is None, "Expected different handling of None spectrum."


@pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
def test_derive_inchi_from_smiles_collection_handles_missing_inchi_values(missing_value):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"smiles": "C1CCCCC1", "inchi": missing_value})
        .build()
    )

    collection = SpectraCollection([spectrum_in])
    processed = derive_inchi_from_smiles(collection)

    inchi = processed.metadata.loc[0, "inchi"]
    assert inchi.replace('"', "") == "InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"
