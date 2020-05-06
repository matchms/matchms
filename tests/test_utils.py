from matchms.utils import is_valid_inchikey


def test_is_valid_inchikey():
    """Test if strings are correctly classified."""
    inchikeys_true = ["XYLJNLCSTIOKRM-UHFFFAOYSA-N"]
    inchikeys_false = ["XYLJNLCSTIOKRM-UHFFFAOYSA",
                       "XYLJNLCSTIOKRMRUHFFFAOYSASN",
                       "XYLJNLCSTIOKR-MUHFFFAOYSA-N",
                       "XYLJNLCSTIOKRM-UHFFFAOYSA-NN",
                       "Brcc(NC2=NCN2)-ccc3nccnc1-3",
                       "2YLJNLCSTIOKRM-UHFFFAOYSA-N",
                       "XYLJNLCSTIOKRM-aaaaaaaaaa-a"]

    for inchikey in inchikeys_true:
        assert is_valid_inchikey(inchikey), "Expected inchikey is True."
    for inchikey in inchikeys_false:
        assert not is_valid_inchikey(inchikey), "Expected inchikey is False."
