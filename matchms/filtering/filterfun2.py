def filterfun2(spectrum):
    """Toy filter function."""
    spectrum = spectrum.clone()
    spectrum.metadata["some_property2"] = "some_value2"
    return spectrum
