CHANGELOG.md[36m:[m- `add_[1;31mprecursor[m_mz()` filter now also checks for metadata in keys `[1;31mprecursor[mmz` and `[1;31mprecursor[m_mass` [#223](https://github.com/matchms/matchms/pull/223)
CHANGELOG.md[36m:[m- Added filter function 'require_[1;31mprecursor[m_mz' and added 1 assert function in 'ModifiedCosine' [#191](https://github.com/matchms/matchms/pull/191)
CHANGELOG.md[36m:[m- Minor bug in `add_[1;31mprecursor[m_mz` [#161](https://github.com/matchms/matchms/pull/161)
CHANGELOG.md[36m:[m- PrecursorMzMatch for deriving [1;31mprecursor[m m/z matches within a given tolerance [#156](https://github.com/matchms/matchms/pull/156)
README.rst[36m:[m       It can be recalculated from the [1;31mprecursor[m m/z by taking
README.rst[36m:[m   * - [1;31mprecursor[m m/z / :code:`[1;31mprecursor[m_mz`
matchms/Spectrum.py[36m:[m        Losses of spectrum, the difference between the [1;31mprecursor[m and all peaks.
matchms/Spectrum.py[36m:[m        Dict of metadata with for example the scan number of [1;31mprecursor[m m/z.
matchms/Spectrum.py[36m:[m            Dictionary with for example the scan number of [1;31mprecursor[m m/z.
matchms/exporting/save_as_json.py[36m:[m                                      "[1;31mprecursor[m_mz": 222.2})
matchms/exporting/save_as_mgf.py[36m:[m                                      "[1;31mprecursor[m_mz": 222.2})
matchms/exporting/save_as_msp.py[36m:[m                                      "[1;31mprecursor[m_mz": 222.2})
matchms/filtering/__init__.py[36m:[mfrom .add_[1;31mprecursor[m_mz import add_[1;31mprecursor[m_mz
matchms/filtering/__init__.py[36m:[mfrom .remove_peaks_around_[1;31mprecursor[m_mz import remove_peaks_around_[1;31mprecursor[m_mz
matchms/filtering/__init__.py[36m:[mfrom .require_[1;31mprecursor[m_below_mz import require_[1;31mprecursor[m_below_mz
matchms/filtering/__init__.py[36m:[m    "add_[1;31mprecursor[m_mz",
matchms/filtering/__init__.py[36m:[m    "remove_peaks_around_[1;31mprecursor[m_mz",
matchms/filtering/__init__.py[36m:[m    "require_[1;31mprecursor[m_below_mz",
matchms/filtering/add_losses.py[36m:[m    """Derive losses based on [1;31mprecursor[m mass.
matchms/filtering/add_losses.py[36m:[m    [1;31mprecursor[m_mz = spectrum.get("[1;31mprecursor[m_mz", None)
matchms/filtering/add_losses.py[36m:[m    if [1;31mprecursor[m_mz:
matchms/filtering/add_losses.py[36m:[m        assert isinstance([1;31mprecursor[m_mz, (float, int)), ("Expected '[1;31mprecursor[m_mz' to be a scalar number.",
matchms/filtering/add_losses.py[36m:[m                                                        "Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
matchms/filtering/add_losses.py[36m:[m        losses_mz = ([1;31mprecursor[m_mz - peaks_mz)[::-1]
matchms/filtering/add_losses.py[36m:[m        logger.warning("No [1;31mprecursor[m_mz found. Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
matchms/filtering/add_parent_mass.py[36m:[m    Method to calculate the parent mass from given [1;31mprecursor[m m/z together
matchms/filtering/add_parent_mass.py[36m:[m    with charge and/or adduct. Will take [1;31mprecursor[m m/z from "[1;31mprecursor[m_mz"
matchms/filtering/add_parent_mass.py[36m:[m    as provided by running `add_[1;31mprecursor[m_mz`.
matchms/filtering/add_parent_mass.py[36m:[m    [1;31mprecursor[m_mz = spectrum.get("[1;31mprecursor[m_mz", None)
matchms/filtering/add_parent_mass.py[36m:[m    if [1;31mprecursor[m_mz is None:
matchms/filtering/add_parent_mass.py[36m:[m        logger.warning("Missing [1;31mprecursor[m m/z to derive parent mass.")
matchms/filtering/add_parent_mass.py[36m:[m        parent_mass = [1;31mprecursor[m_mz * multiplier - correction_mass
matchms/filtering/add_parent_mass.py[36m:[m        [1;31mprecursor[m_mass = [1;31mprecursor[m_mz * abs(charge)
matchms/filtering/add_parent_mass.py[36m:[m        parent_mass = [1;31mprecursor[m_mass - protons_mass
matchms/filtering/add_precursor_mz.py[36m:[m_accepted_keys = ["[1;31mprecursor[m_mz", "[1;31mprecursor[mmz", "[1;31mprecursor[m_mass"]
matchms/filtering/add_precursor_mz.py[36m:[mdef add_[1;31mprecursor[m_mz(spectrum_in: SpectrumType) -> SpectrumType:
matchms/filtering/add_precursor_mz.py[36m:[m    """Add [1;31mprecursor[m_mz to correct field and make it a float.
matchms/filtering/add_precursor_mz.py[36m:[m    For missing [1;31mprecursor[m_mz field: check if there is "pepmass"" entry instead.
matchms/filtering/add_precursor_mz.py[36m:[m    For string parsed as [1;31mprecursor[m_mz: convert to float.
matchms/filtering/add_precursor_mz.py[36m:[m    [1;31mprecursor[m_mz_key = get_first_common_element(spectrum.metadata.keys(), _accepted_keys)
matchms/filtering/add_precursor_mz.py[36m:[m    [1;31mprecursor[m_mz = spectrum.get([1;31mprecursor[m_mz_key)
matchms/filtering/add_precursor_mz.py[36m:[m    if isinstance([1;31mprecursor[m_mz, _accepted_types):
matchms/filtering/add_precursor_mz.py[36m:[m        if isinstance([1;31mprecursor[m_mz, str):
matchms/filtering/add_precursor_mz.py[36m:[m                [1;31mprecursor[m_mz = float([1;31mprecursor[m_mz.strip())
matchms/filtering/add_precursor_mz.py[36m:[m                logger.warning("%s can't be converted to float.", [1;31mprecursor[m_mz)
matchms/filtering/add_precursor_mz.py[36m:[m        spectrum.set("[1;31mprecursor[m_mz", float([1;31mprecursor[m_mz))
matchms/filtering/add_precursor_mz.py[36m:[m    elif [1;31mprecursor[m_mz is None:
matchms/filtering/add_precursor_mz.py[36m:[m            spectrum.set("[1;31mprecursor[m_mz", pepmass[0])
matchms/filtering/add_precursor_mz.py[36m:[m            logger.info("Added [1;31mprecursor[m_mz entry based on field 'pepmass'.")
matchms/filtering/add_precursor_mz.py[36m:[m            logger.warning("No [1;31mprecursor[m_mz found in metadata.")
matchms/filtering/add_precursor_mz.py[36m:[m        logger.warning("Found [1;31mprecursor[m_mz of undefined type.")
matchms/filtering/default_filters.py[36m:[mfrom .add_[1;31mprecursor[m_mz import add_[1;31mprecursor[m_mz
matchms/filtering/default_filters.py[36m:[m    8. :meth:`~matchms.filtering.add_[1;31mprecursor[m_mz`
matchms/filtering/default_filters.py[36m:[m    spectrum = add_[1;31mprecursor[m_mz(spectrum)
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[mdef remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in: SpectrumType, mz_tolerance: float = 17) -> SpectrumType:
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m       the [1;31mprecursor[m mz, exlcuding the [1;31mprecursor[m peak.
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m        within the [1;31mprecursor[m mz. Default is 17 Da.
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m    [1;31mprecursor[m_mz = spectrum.get("[1;31mprecursor[m_mz", None)
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m    assert [1;31mprecursor[m_mz is not None, "Precursor mz absent."
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m    assert isinstance([1;31mprecursor[m_mz, (float, int)), ("Expected '[1;31mprecursor[m_mz' to be a scalar number.",
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m                                                    "Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
matchms/filtering/remove_peaks_around_precursor_mz.py[36m:[m    peaks_to_remove = ((numpy.abs([1;31mprecursor[m_mz-mzs) <= mz_tolerance) & (mzs != [1;31mprecursor[m_mz))
matchms/filtering/require_precursor_below_mz.py[36m:[mdef require_[1;31mprecursor[m_below_mz(spectrum_in: SpectrumType, max_mz: float = 1000) -> SpectrumType:
matchms/filtering/require_precursor_below_mz.py[36m:[m    """Returns None if the [1;31mprecursor[m_mz of a spectrum is above
matchms/filtering/require_precursor_below_mz.py[36m:[m        Maximum mz value for the [1;31mprecursor[m mz of a spectrum.
matchms/filtering/require_precursor_below_mz.py[36m:[m        All [1;31mprecursor[m mz values greater or equal to this
matchms/filtering/require_precursor_below_mz.py[36m:[m    [1;31mprecursor[m_mz = spectrum.get("[1;31mprecursor[m_mz", None)
matchms/filtering/require_precursor_below_mz.py[36m:[m    assert [1;31mprecursor[m_mz is not None, "Precursor mz absent."
matchms/filtering/require_precursor_below_mz.py[36m:[m    assert isinstance([1;31mprecursor[m_mz, (float, int)), ("Expected '[1;31mprecursor[m_mz' to be a scalar number.",
matchms/filtering/require_precursor_below_mz.py[36m:[m                                                    "Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
matchms/filtering/require_precursor_below_mz.py[36m:[m    if [1;31mprecursor[m_mz >= max_mz:
matchms/filtering/require_precursor_below_mz.py[36m:[m        logger.info("Spectrum with [1;31mprecursor[m_mz %s (>%s) was set to None.",
matchms/filtering/require_precursor_below_mz.py[36m:[m                    str([1;31mprecursor[m_mz), str(max_mz))
matchms/filtering/require_precursor_mz.py[36m:[mdef require_[1;31mprecursor[m_mz(spectrum_in: SpectrumType
matchms/filtering/require_precursor_mz.py[36m:[m    """Returns None if there is no [1;31mprecursor[m_mz or if <=0
matchms/filtering/require_precursor_mz.py[36m:[m    [1;31mprecursor[m_mz = spectrum.get("[1;31mprecursor[m_mz", None)
matchms/filtering/require_precursor_mz.py[36m:[m    if [1;31mprecursor[m_mz is None:
matchms/filtering/require_precursor_mz.py[36m:[m            "Found 'pepmass' but no '[1;31mprecursor[m_mz'. " \
matchms/filtering/require_precursor_mz.py[36m:[m            "Consider applying 'add_[1;31mprecursor[m_mz' filter first."
matchms/filtering/require_precursor_mz.py[36m:[m    assert isinstance([1;31mprecursor[m_mz, (float, int)), \
matchms/filtering/require_precursor_mz.py[36m:[m        ("Expected '[1;31mprecursor[m_mz' to be a scalar number.",
matchms/filtering/require_precursor_mz.py[36m:[m         "Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
matchms/filtering/require_precursor_mz.py[36m:[m    if [1;31mprecursor[m_mz <= 0:
matchms/filtering/require_precursor_mz.py[36m:[m        logger.info("Spectrum without [1;31mprecursor[m_mz was set to None.")
matchms/importing/load_from_usi.py[36m:[m        print(f"Found spectrum with [1;31mprecursor[m m/z of {spectrum.get("[1;31mprecursor[m_mz"):.2f}.")
matchms/importing/load_from_usi.py[36m:[m        metadata["[1;31mprecursor[m_mz"] = spectral_data.get("[1;31mprecursor[m_mz", None)
matchms/importing/parsing_utils.py[36m:[m        - [1;31mprecursor[m_mz, searched for in:
matchms/importing/parsing_utils.py[36m:[m            -->"[1;31mprecursor[m"/"[1;31mprecursor[mMz"--> ... --> "selected ion m/z"/"[1;31mprecursor[mMz"
matchms/importing/parsing_utils.py[36m:[m    [1;31mprecursor[m_mz = None
matchms/importing/parsing_utils.py[36m:[m    first_search = list(find_by_key(spectrum_dict, "[1;31mprecursor[m"))
matchms/importing/parsing_utils.py[36m:[m        first_search = list(find_by_key(spectrum_dict, "[1;31mprecursor[mMz"))
matchms/importing/parsing_utils.py[36m:[m        [1;31mprecursor[m_mz_search = list(find_by_key(first_search, "selected ion m/z"))
matchms/importing/parsing_utils.py[36m:[m        if not [1;31mprecursor[m_mz_search:
matchms/importing/parsing_utils.py[36m:[m            [1;31mprecursor[m_mz_search = list(find_by_key(first_search, "[1;31mprecursor[mMz"))
matchms/importing/parsing_utils.py[36m:[m        if [1;31mprecursor[m_mz_search:
matchms/importing/parsing_utils.py[36m:[m            [1;31mprecursor[m_mz = float([1;31mprecursor[m_mz_search[0])
matchms/importing/parsing_utils.py[36m:[m    [1;31mprecursor[m_charge = list(find_by_key(first_search, "charge state"))
matchms/importing/parsing_utils.py[36m:[m    if [1;31mprecursor[m_charge:
matchms/importing/parsing_utils.py[36m:[m        charge = int([1;31mprecursor[m_charge[0])
matchms/importing/parsing_utils.py[36m:[m            "[1;31mprecursor[m_mz": [1;31mprecursor[m_mz,
matchms/networking/SimilarityNetwork.py[36m:[m                              metadata={"[1;31mprecursor[m_mz": 100.0,
matchms/networking/SimilarityNetwork.py[36m:[m                              metadata={"[1;31mprecursor[m_mz": 105.0,
matchms/similarity/ModifiedCosine.py[36m:[m    simply the difference in [1;31mprecursor[m-m/z between the two spectra.
matchms/similarity/ModifiedCosine.py[36m:[m                              metadata={"[1;31mprecursor[m_mz": 100.0})
matchms/similarity/ModifiedCosine.py[36m:[m                              metadata={"[1;31mprecursor[m_mz": 105.0})
matchms/similarity/ModifiedCosine.py[36m:[m        def get_valid_[1;31mprecursor[m_mz(spectrum):
matchms/similarity/ModifiedCosine.py[36m:[m            """Extract valid [1;31mprecursor[m_mz from spectrum if possible. If not raise exception."""
matchms/similarity/ModifiedCosine.py[36m:[m            message_[1;31mprecursor[m_missing = \
matchms/similarity/ModifiedCosine.py[36m:[m                "Precursor_mz missing. Apply 'add_[1;31mprecursor[m_mz' filter first."
matchms/similarity/ModifiedCosine.py[36m:[m            message_[1;31mprecursor[m_no_number = \
matchms/similarity/ModifiedCosine.py[36m:[m                "Precursor_mz must be of type int or float. Apply 'add_[1;31mprecursor[m_mz' filter first."
matchms/similarity/ModifiedCosine.py[36m:[m            message_[1;31mprecursor[m_below_0 = "Expect [1;31mprecursor[m to be positive number." \
matchms/similarity/ModifiedCosine.py[36m:[m                                        "Apply 'require_[1;31mprecursor[m_mz' first"
matchms/similarity/ModifiedCosine.py[36m:[m            [1;31mprecursor[m_mz = spectrum.get("[1;31mprecursor[m_mz", None)
matchms/similarity/ModifiedCosine.py[36m:[m            assert [1;31mprecursor[m_mz, message_[1;31mprecursor[m_missing
matchms/similarity/ModifiedCosine.py[36m:[m            if isinstance([1;31mprecursor[m_mz, str):
matchms/similarity/ModifiedCosine.py[36m:[m                    [1;31mprecursor[m_mz = float([1;31mprecursor[m_mz.strip())
matchms/similarity/ModifiedCosine.py[36m:[m                    logger.warning("[1;31mprecursor[m_mz was found as string and converted to float."
matchms/similarity/ModifiedCosine.py[36m:[m                                   "Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
matchms/similarity/ModifiedCosine.py[36m:[m                    logger.exception("%s can't be converted to float.", [1;31mprecursor[m_mz)
matchms/similarity/ModifiedCosine.py[36m:[m            assert isinstance([1;31mprecursor[m_mz, (int, float)), message_[1;31mprecursor[m_no_number
matchms/similarity/ModifiedCosine.py[36m:[m            assert [1;31mprecursor[m_mz > 0, message_[1;31mprecursor[m_below_0
matchms/similarity/ModifiedCosine.py[36m:[m            return [1;31mprecursor[m_mz
matchms/similarity/ModifiedCosine.py[36m:[m            [1;31mprecursor[m_mz_ref = get_valid_[1;31mprecursor[m_mz(reference)
matchms/similarity/ModifiedCosine.py[36m:[m            [1;31mprecursor[m_mz_query = get_valid_[1;31mprecursor[m_mz(query)
matchms/similarity/ModifiedCosine.py[36m:[m            mass_shift = [1;31mprecursor[m_mz_ref - [1;31mprecursor[m_mz_query
matchms/similarity/PrecursorMzMatch.py[36m:[m    """Return True if spectrums match in [1;31mprecursor[m m/z (within tolerance), and False otherwise.
matchms/similarity/PrecursorMzMatch.py[36m:[m                              metadata={"id": "1", "[1;31mprecursor[m_mz": 100})
matchms/similarity/PrecursorMzMatch.py[36m:[m                              metadata={"id": "2", "[1;31mprecursor[m_mz": 110})
matchms/similarity/PrecursorMzMatch.py[36m:[m                              metadata={"id": "3", "[1;31mprecursor[m_mz": 103})
matchms/similarity/PrecursorMzMatch.py[36m:[m                              metadata={"id": "4", "[1;31mprecursor[m_mz": 111})
matchms/similarity/PrecursorMzMatch.py[36m:[m        """Compare [1;31mprecursor[m m/z between reference and query spectrum.
matchms/similarity/PrecursorMzMatch.py[36m:[m        [1;31mprecursor[mmz_ref = reference.get("[1;31mprecursor[m_mz")
matchms/similarity/PrecursorMzMatch.py[36m:[m        [1;31mprecursor[mmz_query = query.get("[1;31mprecursor[m_mz")
matchms/similarity/PrecursorMzMatch.py[36m:[m        assert [1;31mprecursor[mmz_ref is not None and [1;31mprecursor[mmz_query is not None, "Missing [1;31mprecursor[m m/z."
matchms/similarity/PrecursorMzMatch.py[36m:[m            return abs([1;31mprecursor[mmz_ref - [1;31mprecursor[mmz_query) <= self.tolerance
matchms/similarity/PrecursorMzMatch.py[36m:[m        mean_mz = ([1;31mprecursor[mmz_ref + [1;31mprecursor[mmz_query) / 2
matchms/similarity/PrecursorMzMatch.py[36m:[m        score = abs([1;31mprecursor[mmz_ref - [1;31mprecursor[mmz_query)/mean_mz <= self.tolerance
matchms/similarity/PrecursorMzMatch.py[36m:[m        def collect_[1;31mprecursor[mmz(spectrums):
matchms/similarity/PrecursorMzMatch.py[36m:[m            """Collect [1;31mprecursor[ms."""
matchms/similarity/PrecursorMzMatch.py[36m:[m            [1;31mprecursor[ms = []
matchms/similarity/PrecursorMzMatch.py[36m:[m                [1;31mprecursor[mmz = spectrum.get("[1;31mprecursor[m_mz")
matchms/similarity/PrecursorMzMatch.py[36m:[m                assert [1;31mprecursor[mmz is not None, "Missing [1;31mprecursor[m m/z."
matchms/similarity/PrecursorMzMatch.py[36m:[m                [1;31mprecursor[ms.append([1;31mprecursor[mmz)
matchms/similarity/PrecursorMzMatch.py[36m:[m            return numpy.asarray([1;31mprecursor[ms)
matchms/similarity/PrecursorMzMatch.py[36m:[m        [1;31mprecursor[ms_ref = collect_[1;31mprecursor[mmz(references)
matchms/similarity/PrecursorMzMatch.py[36m:[m        [1;31mprecursor[ms_query = collect_[1;31mprecursor[mmz(queries)
matchms/similarity/PrecursorMzMatch.py[36m:[m            return [1;31mprecursor[mmz_scores_symmetric([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query,
matchms/similarity/PrecursorMzMatch.py[36m:[m            return [1;31mprecursor[mmz_scores_symmetric_ppm([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query,
matchms/similarity/PrecursorMzMatch.py[36m:[m            return [1;31mprecursor[mmz_scores([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query,
matchms/similarity/PrecursorMzMatch.py[36m:[m        return [1;31mprecursor[mmz_scores_ppm([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query,
matchms/similarity/PrecursorMzMatch.py[36m:[mdef [1;31mprecursor[mmz_scores([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance):
matchms/similarity/PrecursorMzMatch.py[36m:[m    scores = numpy.zeros((len([1;31mprecursor[ms_ref), len([1;31mprecursor[ms_query)))
matchms/similarity/PrecursorMzMatch.py[36m:[m    for i, [1;31mprecursor[mmz_ref in enumerate([1;31mprecursor[ms_ref):
matchms/similarity/PrecursorMzMatch.py[36m:[m        for j, [1;31mprecursor[mmz_query in enumerate([1;31mprecursor[ms_query):
matchms/similarity/PrecursorMzMatch.py[36m:[m            scores[i, j] = (abs([1;31mprecursor[mmz_ref - [1;31mprecursor[mmz_query) <= tolerance)
matchms/similarity/PrecursorMzMatch.py[36m:[mdef [1;31mprecursor[mmz_scores_symmetric([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance):
matchms/similarity/PrecursorMzMatch.py[36m:[m    scores = numpy.zeros((len([1;31mprecursor[ms_ref), len([1;31mprecursor[ms_query)))
matchms/similarity/PrecursorMzMatch.py[36m:[m    for i, [1;31mprecursor[mmz_ref in enumerate([1;31mprecursor[ms_ref):
matchms/similarity/PrecursorMzMatch.py[36m:[m        for j in range(i, len([1;31mprecursor[ms_query)):
matchms/similarity/PrecursorMzMatch.py[36m:[m            scores[i, j] = (abs([1;31mprecursor[mmz_ref - [1;31mprecursor[ms_query[j]) <= tolerance)
matchms/similarity/PrecursorMzMatch.py[36m:[mdef [1;31mprecursor[mmz_scores_ppm([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance_ppm):
matchms/similarity/PrecursorMzMatch.py[36m:[m    scores = numpy.zeros((len([1;31mprecursor[ms_ref), len([1;31mprecursor[ms_query)))
matchms/similarity/PrecursorMzMatch.py[36m:[m    for i, [1;31mprecursor[mmz_ref in enumerate([1;31mprecursor[ms_ref):
matchms/similarity/PrecursorMzMatch.py[36m:[m        for j, [1;31mprecursor[mmz_query in enumerate([1;31mprecursor[ms_query):
matchms/similarity/PrecursorMzMatch.py[36m:[m            mean_mz = ([1;31mprecursor[mmz_ref + [1;31mprecursor[mmz_query)/2
matchms/similarity/PrecursorMzMatch.py[36m:[m            scores[i, j] = (abs([1;31mprecursor[mmz_ref - [1;31mprecursor[mmz_query)/mean_mz * 1e6 <= tolerance_ppm)
matchms/similarity/PrecursorMzMatch.py[36m:[mdef [1;31mprecursor[mmz_scores_symmetric_ppm([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance_ppm):
matchms/similarity/PrecursorMzMatch.py[36m:[m    scores = numpy.zeros((len([1;31mprecursor[ms_ref), len([1;31mprecursor[ms_query)))
matchms/similarity/PrecursorMzMatch.py[36m:[m    for i, [1;31mprecursor[mmz_ref in enumerate([1;31mprecursor[ms_ref):
matchms/similarity/PrecursorMzMatch.py[36m:[m        for j in range(i, len([1;31mprecursor[ms_query)):
matchms/similarity/PrecursorMzMatch.py[36m:[m            mean_mz = ([1;31mprecursor[mmz_ref + [1;31mprecursor[ms_query[j])/2
matchms/similarity/PrecursorMzMatch.py[36m:[m            diff_ppm = abs([1;31mprecursor[mmz_ref - [1;31mprecursor[ms_query[j])/mean_mz * 1e6
matchms/similarity/__init__.py[36m:[m* simple scores that only assess [1;31mprecursor[m m/z or parent mass matches
tests/test_PrecursormzMatch.py[36m:[mfrom matchms.similarity.PrecursorMzMatch import [1;31mprecursor[mmz_scores
tests/test_PrecursormzMatch.py[36m:[mfrom matchms.similarity.PrecursorMzMatch import [1;31mprecursor[mmz_scores_ppm
tests/test_PrecursormzMatch.py[36m:[mfrom matchms.similarity.PrecursorMzMatch import [1;31mprecursor[mmz_scores_symmetric
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[mmz_scores_symmetric_ppm
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 101.0})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_tolerance2():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 101.0})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_tolerance_ppm():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 600.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 600.001})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_missing_[1;31mprecursor[mmz():
tests/test_PrecursormzMatch.py[36m:[m    """Test with missing [1;31mprecursor[mmz."""
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m    expected_message_part = "Missing [1;31mprecursor[m m/z."
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_array():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 101.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 99.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 98.0})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_tolerance2_array():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 101.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 99.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 98.0})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_tolerance2_array_ppm():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 101.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 99.99})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 98.0})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_array_symmetric():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 101.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 99.95})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 98.0})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_match_array_symmetric_pmm():
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.0})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 100.01})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 99.99999})
tests/test_PrecursormzMatch.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 99.9})
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_scores(numba_compiled):
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[ms_ref = numpy.asarray([101, 200, 300])
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[ms_query = numpy.asarray([100, 301])
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance=2.0)
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores.py_func([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance=2.0)
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_scores_symmetric(numba_compiled):
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[ms = numpy.asarray([101, 100, 200])
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores_symmetric([1;31mprecursor[ms, [1;31mprecursor[ms, tolerance=2.0)
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores_symmetric.py_func([1;31mprecursor[ms, [1;31mprecursor[ms, tolerance=2.0)
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_scores_ppm(numba_compiled):
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[ms_ref = numpy.asarray([100.00001, 200, 300])
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[ms_query = numpy.asarray([100, 300.00001])
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores_ppm([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance_ppm=2.0)
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores_ppm.py_func([1;31mprecursor[ms_ref, [1;31mprecursor[ms_query, tolerance_ppm=2.0)
tests/test_PrecursormzMatch.py[36m:[mdef test_[1;31mprecursor[mmz_scores_symmetric_ppm(numba_compiled):
tests/test_PrecursormzMatch.py[36m:[m    [1;31mprecursor[ms = numpy.asarray([100.00001, 100, 200])
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores_symmetric_ppm([1;31mprecursor[ms, [1;31mprecursor[ms, tolerance_ppm=2.0)
tests/test_PrecursormzMatch.py[36m:[m        scores = [1;31mprecursor[mmz_scores_symmetric_ppm.py_func([1;31mprecursor[ms, [1;31mprecursor[ms, tolerance_ppm=2.0)
tests/test_SimilarityNetwork.py[36m:[m                                            "[1;31mprecursor[m_mz": 100+50*i}))
tests/test_SimilarityNetwork.py[36m:[m                                            "[1;31mprecursor[m_mz": 110+50*i}))
tests/test_add_losses.py[36m:[m    """Test if all losses are correctly generated form mz values and [1;31mprecursor[m-m/z."""
tests/test_add_losses.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": 445.0})
tests/test_add_losses.py[36m:[mdef test_add_losses_without_[1;31mprecursor[m_mz():
tests/test_add_losses.py[36m:[m    """Test if no changes are done without having a [1;31mprecursor[m-m/z."""
tests/test_add_losses.py[36m:[m         "No [1;31mprecursor[m_mz found. Consider applying 'add_[1;31mprecursor[m_mz' filter first.")
tests/test_add_losses.py[36m:[mdef test_add_losses_with_[1;31mprecursor[m_mz_wrong_type():
tests/test_add_losses.py[36m:[m    """Test if correct assert error is raised for [1;31mprecursor[m-mz as string."""
tests/test_add_losses.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": "445.0"})
tests/test_add_losses.py[36m:[m    assert "Expected '[1;31mprecursor[m_mz' to be a scalar number." in str(msg.value)
tests/test_add_losses.py[36m:[mdef test_add_losses_with_peakmz_larger_[1;31mprecursor[mmz():
tests/test_add_losses.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": 445.0})
tests/test_add_losses.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": 445.0})
tests/test_add_parent_mass.py[36m:[mdef test_add_parent_mass_pepmass_no_[1;31mprecursor[mmz(caplog):
tests/test_add_parent_mass.py[36m:[mdef test_add_parent_mass_no_[1;31mprecursor[mmz(caplog):
tests/test_add_parent_mass.py[36m:[m    assert "Missing [1;31mprecursor[m m/z to derive parent mass." in caplog.text
tests/test_add_parent_mass.py[36m:[mdef test_add_parent_mass_[1;31mprecursor[mmz_zero_charge(caplog):
tests/test_add_parent_mass.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0,
tests/test_add_parent_mass.py[36m:[mdef test_add_parent_mass_[1;31mprecursor[mmz(caplog):
tests/test_add_parent_mass.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0,
tests/test_add_parent_mass.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0,
tests/test_add_parent_mass.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0,
tests/test_add_parent_mass.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0}
tests/test_add_parent_mass.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0, "ionmode": ionmode}
tests/test_add_precursor_mz.py[36m:[mfrom matchms.filtering import add_[1;31mprecursor[m_mz
tests/test_add_precursor_mz.py[36m:[mdef test_add_[1;31mprecursor[m_mz():
tests/test_add_precursor_mz.py[36m:[m    """Test if [1;31mprecursor[m_mz is correctly derived. Here nothing should change."""
tests/test_add_precursor_mz.py[36m:[m    metadata = {"[1;31mprecursor[m_mz": 444.0}
tests/test_add_precursor_mz.py[36m:[m    spectrum = add_[1;31mprecursor[m_mz(spectrum_in)
tests/test_add_precursor_mz.py[36m:[m    assert spectrum.get("[1;31mprecursor[m_mz") == 444.0, "Expected different [1;31mprecursor[m_mz."
tests/test_add_precursor_mz.py[36m:[mdef test_add_[1;31mprecursor[m_mz_no_masses():
tests/test_add_precursor_mz.py[36m:[m    """Test if no [1;31mprecursor[m_mz is handled correctly. Here nothing should change."""
tests/test_add_precursor_mz.py[36m:[m    spectrum = add_[1;31mprecursor[m_mz(spectrum_in)
tests/test_add_precursor_mz.py[36m:[m    assert spectrum.get("[1;31mprecursor[m_mz") is None, "Outcome should be None."
tests/test_add_precursor_mz.py[36m:[mdef test_add_[1;31mprecursor[m_mz_only_pepmass_present(caplog):
tests/test_add_precursor_mz.py[36m:[m    """Test if [1;31mprecursor[m_mz is correctly derived if only pepmass is present."""
tests/test_add_precursor_mz.py[36m:[m    spectrum = add_[1;31mprecursor[m_mz(spectrum_in)
tests/test_add_precursor_mz.py[36m:[m    assert spectrum.get("[1;31mprecursor[m_mz") == 444.0, "Expected different [1;31mprecursor[m_mz."
tests/test_add_precursor_mz.py[36m:[m    assert "Added [1;31mprecursor[m_mz entry based on field 'pepmass'" in caplog.text, \
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mz", "444.0", 444.0],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[mmz", "15.6", 15.6],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[mmz", 15.0, 15.0],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mass", "17.887654", 17.887654],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mass", "N/A", None],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mass", "test", None],
tests/test_add_precursor_mz.py[36m:[mdef test_add_[1;31mprecursor[m_mz_no_[1;31mprecursor[m_mz(key, value, expected):
tests/test_add_precursor_mz.py[36m:[m    """Test if [1;31mprecursor[m_mz is correctly derived if "[1;31mprecursor[m_mz" is str."""
tests/test_add_precursor_mz.py[36m:[m    spectrum = add_[1;31mprecursor[m_mz(spectrum_in)
tests/test_add_precursor_mz.py[36m:[m    assert spectrum.get("[1;31mprecursor[m_mz") == expected, "Expected different [1;31mprecursor[m_mz."
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mz", "N/A", "N/A can't be converted to float."],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mass", "test", "test can't be converted to float."],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mz", None, "No [1;31mprecursor[m_mz found in metadata."],
tests/test_add_precursor_mz.py[36m:[m    ["pepmass", None, "No [1;31mprecursor[m_mz found in metadata."],
tests/test_add_precursor_mz.py[36m:[m    ["[1;31mprecursor[m_mz", [], "Found [1;31mprecursor[m_mz of undefined type."]])
tests/test_add_precursor_mz.py[36m:[mdef test_add_[1;31mprecursor[m_mz_logging(key, value, expected_log, caplog):
tests/test_add_precursor_mz.py[36m:[m    """Test if [1;31mprecursor[m_mz is correctly derived if "[1;31mprecursor[m_mz" is str."""
tests/test_add_precursor_mz.py[36m:[m    _ = add_[1;31mprecursor[m_mz(spectrum_in)
tests/test_add_precursor_mz.py[36m:[m    spectrum = add_[1;31mprecursor[m_mz(spectrum_in)
tests/test_hashing.py[36m:[m                    metadata={"[1;31mprecursor[m_mz": 505.0})
tests/test_hashing.py[36m:[m                         metadata={"[1;31mprecursor[m_mz": 505.0})
tests/test_load_from_mzml.py[36m:[m    assert int(spectrums[5].get("[1;31mprecursor[m_mz")) == 177, "Expected different [1;31mprecursor[m m/z"
tests/test_load_from_mzxml.py[36m:[m    assert int(spectrum.get("[1;31mprecursor[m_mz")) == 343, "Expected different [1;31mprecursor[m m/z"
tests/test_load_from_usi.py[36m:[m    expected_metadata = {"usi": "something", "server": "https://metabolomics-usi.ucsd.edu", "[1;31mprecursor[m_mz": None}
tests/test_modified_cosine.py[36m:[mdef test_modified_cosine_without_[1;31mprecursor[m_mz():
tests/test_modified_cosine.py[36m:[m    """Test without [1;31mprecursor[m-m/z. Should raise assertion error."""
tests/test_modified_cosine.py[36m:[m    expected_message = "Precursor_mz missing. Apply 'add_[1;31mprecursor[m_mz' filter first."
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1000.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1005.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1000.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1005})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1000.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1010.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1000.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1005.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1000.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1005})
tests/test_modified_cosine.py[36m:[mdef test_modified_cosine_[1;31mprecursor[m_mz_as_string():
tests/test_modified_cosine.py[36m:[m    """Test modified cosine on two spectra with [1;31mprecursor[m_mz given as string."""
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": 1000.0})
tests/test_modified_cosine.py[36m:[m                          metadata={"[1;31mprecursor[m_mz": "1005.0"})
tests/test_modified_cosine.py[36m:[m    expected_message = "Precursor_mz must be of type int or float. Apply 'add_[1;31mprecursor[m_mz' filter first."
tests/test_normalize_intensities.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": 45.0})
tests/test_remove_peaks_around_precursor_mz.py[36m:[mfrom matchms.filtering import remove_peaks_around_[1;31mprecursor[m_mz
tests/test_remove_peaks_around_precursor_mz.py[36m:[mdef test_remove_peaks_around_[1;31mprecursor[m_mz_no_params():
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    """Using defaults with [1;31mprecursor[m mz present."""
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 60.)
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum = remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in)
tests/test_remove_peaks_around_precursor_mz.py[36m:[mdef test_remove_peaks_around_[1;31mprecursor[m_mz_tolerance_20():
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 60.)
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum = remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in, mz_tolerance=20)
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 1.)
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum = remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in)
tests/test_remove_peaks_around_precursor_mz.py[36m:[mdef test_remove_peaks_around_[1;31mprecursor[m_without_[1;31mprecursor[m_mz():
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    """Test if correct assert error is raised for missing [1;31mprecursor[m-mz."""
tests/test_remove_peaks_around_precursor_mz.py[36m:[m        _ = remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in)
tests/test_remove_peaks_around_precursor_mz.py[36m:[mdef test_remove_peaks_around_[1;31mprecursor[m_with_wrong_[1;31mprecursor[m_mz():
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    """Test if correct assert error is raised for [1;31mprecursor[m-mz as string."""
tests/test_remove_peaks_around_precursor_mz.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": "445.0"})
tests/test_remove_peaks_around_precursor_mz.py[36m:[m        _ = remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in)
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    assert "Expected '[1;31mprecursor[m_mz' to be a scalar number." in str(msg.value)
tests/test_remove_peaks_around_precursor_mz.py[36m:[m    spectrum = remove_peaks_around_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_below_mz.py[36m:[mfrom matchms.filtering import require_[1;31mprecursor[m_below_mz
tests/test_require_precursor_below_mz.py[36m:[mdef test_require_[1;31mprecursor[m_below_mz_no_params():
tests/test_require_precursor_below_mz.py[36m:[m    """Using default parameterse with [1;31mprecursor[m mz present."""
tests/test_require_precursor_below_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 60.)
tests/test_require_precursor_below_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_below_mz(spectrum_in)
tests/test_require_precursor_below_mz.py[36m:[mdef test_require_[1;31mprecursor[m_below_mz_max_50():
tests/test_require_precursor_below_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 60.)
tests/test_require_precursor_below_mz.py[36m:[m        spectrum = require_[1;31mprecursor[m_below_mz(spectrum_in, max_mz=50)
tests/test_require_precursor_below_mz.py[36m:[m        ('matchms', 'INFO', 'Spectrum with [1;31mprecursor[m_mz 60.0 (>50) was set to None.')
tests/test_require_precursor_below_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 1.)
tests/test_require_precursor_below_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_below_mz(spectrum_in)
tests/test_require_precursor_below_mz.py[36m:[mdef test_require_[1;31mprecursor[m_below_without_[1;31mprecursor[m_mz():
tests/test_require_precursor_below_mz.py[36m:[m    """Test if correct assert error is raised for missing [1;31mprecursor[m-mz."""
tests/test_require_precursor_below_mz.py[36m:[m        _ = require_[1;31mprecursor[m_below_mz(spectrum_in)
tests/test_require_precursor_below_mz.py[36m:[mdef test_require_[1;31mprecursor[m_below_with_wrong_[1;31mprecursor[m_mz():
tests/test_require_precursor_below_mz.py[36m:[m    """Test if correct assert error is raised for [1;31mprecursor[m-mz as string."""
tests/test_require_precursor_below_mz.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": "445.0"})
tests/test_require_precursor_below_mz.py[36m:[m        _ = require_[1;31mprecursor[m_below_mz(spectrum_in)
tests/test_require_precursor_below_mz.py[36m:[m    assert "Expected '[1;31mprecursor[m_mz' to be a scalar number." in str(msg.value)
tests/test_require_precursor_below_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_below_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[mfrom matchms.filtering.require_[1;31mprecursor[m_mz import require_[1;31mprecursor[m_mz
tests/test_require_precursor_mz.py[36m:[mdef test_require_[1;31mprecursor[m_mz_pass():
tests/test_require_precursor_mz.py[36m:[m    """Test with correct [1;31mprecursor[m mz present."""
tests/test_require_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 60.)
tests/test_require_precursor_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[mdef test_require_[1;31mprecursor[m_mz_fail_because_zero():
tests/test_require_precursor_mz.py[36m:[m    """Test if spectrum is None when [1;31mprecursor[m_mz == 0"""
tests/test_require_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 0.0)
tests/test_require_precursor_mz.py[36m:[m        spectrum = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[m        ('matchms', 'INFO', 'Spectrum without [1;31mprecursor[m_mz was set to None.')
tests/test_require_precursor_mz.py[36m:[mdef test_require_[1;31mprecursor[m_mz_fail_because_below_zero():
tests/test_require_precursor_mz.py[36m:[m    """Test if spectrum is None when [1;31mprecursor[m_mz < 0"""
tests/test_require_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", -3.5)
tests/test_require_precursor_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[m    spectrum_in.set("[1;31mprecursor[m_mz", 1.)
tests/test_require_precursor_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[mdef test_require_[1;31mprecursor[m_mz_without_[1;31mprecursor[m_mz():
tests/test_require_precursor_mz.py[36m:[m    """Test if None is returned for missing [1;31mprecursor[m-mz."""
tests/test_require_precursor_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[mdef test_require_[1;31mprecursor[m_mz_with_wrong_[1;31mprecursor[m_mz():
tests/test_require_precursor_mz.py[36m:[m    """Test if correct assert error is raised for [1;31mprecursor[m-mz as string."""
tests/test_require_precursor_mz.py[36m:[m                           metadata={"[1;31mprecursor[m_mz": "445.0"})
tests/test_require_precursor_mz.py[36m:[m        _ = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_require_precursor_mz.py[36m:[m    assert "Expected '[1;31mprecursor[m_mz' to be a scalar number." in str(msg.value)
tests/test_require_precursor_mz.py[36m:[mdef test_require_[1;31mprecursor[m_mz_with_input_none():
tests/test_require_precursor_mz.py[36m:[m    spectrum = require_[1;31mprecursor[m_mz(spectrum_in)
tests/test_save_as_json_load_from_json.py[36m:[m                                  "[1;31mprecursor[m_mz": 222.2,
tests/testdata.mzXML[36m:[m    <[1;31mprecursor[mMz [1;31mprecursor[mIntensity="848424">343.0672302</[1;31mprecursor[mMz>
tests/testdata.mzXML[36m:[m    <[1;31mprecursor[mMz [1;31mprecursor[mIntensity="427921">191.0561371</[1;31mprecursor[mMz>
tests/testdata.mzXML[36m:[m    <[1;31mprecursor[mMz [1;31mprecursor[mIntensity="53046.2">169.0142059</[1;31mprecursor[mMz>
tests/testdata.mzXML[36m:[m    <[1;31mprecursor[mMz [1;31mprecursor[mIntensity="427921">191.0561371</[1;31mprecursor[mMz>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=1">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=10">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
tests/testdata.mzml[36m:[m          <[1;31mprecursor[mList count="1">
tests/testdata.mzml[36m:[m            <[1;31mprecursor[m spectrumRef="controllerType=0 controllerNumber=1 scan=10">
tests/testdata.mzml[36m:[m            </[1;31mprecursor[m>
tests/testdata.mzml[36m:[m          </[1;31mprecursor[mList>
