def filterfun1(spectrum):
    """Toy filter function."""
    spectrum = spectrum.clone()
    spectrum.metadata["some_property1"] = "some_value1"
    return spectrum
