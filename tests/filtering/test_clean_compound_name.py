import pytest
from matchms.filtering import clean_compound_name
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "name, expected",
    [
        ["MLS000863588-01!2-methoxy-3-methyl-9H-carbazole", "2-methoxy-3-methyl-9H-carbazole"],
        ["NCGC00160217-01!SOPHOCARPINE", "SOPHOCARPINE"],
        ["0072_2-Mercaptobenzothiaz", "2-Mercaptobenzothiaz"],
        [r"MassbankEU:ET110206 NPE_327.1704_12.2|N-succinylnorpheniramine", "N-succinylnorpheniramine"],
        ["Massbank:CE000307 Trans-Zeatin-[d5]", "Trans-Zeatin-[d5]"],
        ["HMDB:HMDB00500-718 4-Hydroxybenzoic acid", "4-Hydroxybenzoic acid"],
        ["MoNA:2346734 Piroxicam (Feldene)", "Piroxicam (Feldene)"],
        ["ReSpect:PS013405 option1|option2|option3", "option3"],
        ["ReSpect:PS013405 option1name", "option1name"],
        ["4,4-Dimethylcholest-8(9),24-dien-3.beta.-ol  231.2", "4,4-Dimethylcholest-8(9),24-dien-3.beta.-ol"],
        ["", ""],
    ],
)
def test_clean_compound_name(name, expected):
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata({"compound_name": name}).build()
    spectrum = clean_compound_name(spectrum_in)

    assert spectrum.get("compound_name") == expected, "Expected different cleaned name."
