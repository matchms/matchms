import numpy
from matchms import Spectrum
from matchms.filtering import clean_compound_name


def test_clean_compound_name_from_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": "peptideXYZ [M+H+K]"})

    spectrum = clean_compound_name(spectrum_in)

    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."


def test_clean_compound_name_from_compound_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"compound_name": "peptideXYZ [M+H+K]"})

    spectrum = clean_compound_name(spectrum_in)

    assert spectrum.get("compound_name") == "peptideXYZ", "Expected different cleaned name."


def test_clean_compound_name_present_name_and_compound_name():
    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"name": "peptideXYZ [M+H+K]",
                                     "compound_name": "peptide_superwelldescribed"})

    spectrum = clean_compound_name(spectrum_in)

    assert spectrum.get("name") == "peptideXYZ [M+H+K]", "Expected different name."
    assert spectrum.get("compound_name") == "peptide_superwelldescribed", "Expected different compound name."


def test_clean_compound_name_removing_known_non_name_parts():
    """Test difficult but representative examples."""
    test_name_strings = [
        ["MLS000863588-01!2-methoxy-3-methyl-9H-carbazole",
         "2-methoxy-3-methyl-9H-carbazole"],
        ["NCGC00160217-01!SOPHOCARPINE",
         "SOPHOCARPINE"],
        ["0072_2-Mercaptobenzothiaz",
         "2-Mercaptobenzothiaz"],
        [r"MassbankEU:ET110206 NPE_327.1704_12.2|N-succinylnorpheniramine",
         "N-succinylnorpheniramine"],
        ["Massbank:CE000307 Trans-Zeatin-[d5]",
         "Trans-Zeatin-[d5]"],
        ["HMDB:HMDB00500-718 4-Hydroxybenzoic acid",
         "4-Hydroxybenzoic acid"],
        ["MoNA:2346734 Piroxicam (Feldene) [M+H]",
         "Piroxicam (Feldene)"]
    ]
    for name_strings in test_name_strings:
        spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                               intensities=numpy.array([], dtype="float"),
                               metadata={"name": name_strings[0]})

        spectrum = clean_compound_name(spectrum_in)

        assert spectrum.get("compound_name") == name_strings[1], "Expected different cleaned name."
