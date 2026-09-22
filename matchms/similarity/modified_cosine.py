"""Modified cosine similarity using direct and precursor-shifted peak matches."""
from .cosine import Cosine


class ModifiedCosine(Cosine):
    """Compare mass spectra using modified cosine similarity.

    Modified cosine extends ordinary cosine matching by allowing peaks to match
    either directly by fragment m/z or after accounting for the difference
    between the precursor m/z values of the two spectra.

    For spectra with precursor masses ``P1`` and ``P2``, a peak pair can become
    a candidate in two ways:

    1. the fragment m/z values match directly within ``tolerance``;
    2. the corresponding neutral-loss values
       ``precursor_mz - fragment_mz`` match within ``tolerance``.

    In absolute Da matching, the second condition is equivalent to shifting one
    spectrum by the precursor-mass difference before comparing fragment peaks.
    This allows structurally related spectra with shifted fragment ions to
    receive a high similarity score.

    Direct and precursor-shifted candidates participate in one common
    intensity-product greedy assignment. Each physical peak can be used at most
    once, regardless of whether its accepted match originated from the direct
    or shifted candidate set. The returned ``matches`` field is the number of
    peak pairs accepted by this one-to-one assignment.

    ``ModifiedCosine`` uses the same public interface as
    :class:`~matchms.similarity.Cosine` and supports three main workflows:

    - :meth:`pair` compares two individual spectra;
    - :meth:`matrix` calculates a complete matrix between spectrum collections;
    - :meth:`build_index` and :meth:`search` provide repeated querying of a fixed
      reference library using a persistent index.

    Although ``ModifiedCosine`` is implemented as a specialization of
    :class:`~matchms.similarity.Cosine`, its default indexed matching mode
    includes both direct fragment and neutral-loss candidates. Users therefore
    do not need to configure an internal matching mode explicitly.

    When the precursor-mass difference is itself within the fragment tolerance,
    direct and shifted matching describe the same candidate region and the
    indexed implementation avoids processing the redundant shifted candidate
    set.

    If precursor metadata are unavailable, precursor-shifted candidates cannot
    be generated. Direct fragment matching remains available.

    Set ``use_hungarian=True`` to use optimal modified-cosine assignment for
    :meth:`pair` and :meth:`matrix`. This alternative does not support
    persistent indexed searches. For the traditional pair-oriented greedy
    implementation, see
    :class:`~matchms.similarity.ModifiedCosineGreedy`.

    Parameters
    ----------
    tolerance
        Maximum difference for a direct fragment or neutral-loss match. The
        tolerance boundary is inclusive. Interpreted as Da unless
        ``use_ppm=True``.
    intensity_power
        Exponent applied to peak intensities before cosine scoring. The default
        of 1 uses the original intensities; values below 1 reduce the influence
        of very intense peaks.
    use_hungarian
        If False, use indexed greedy modified-cosine assignment. If True, use
        Hungarian assignment for ``pair`` and ``matrix``. Persistent
        ``build_index`` / ``search`` workflows are unavailable with Hungarian
        matching.
    noise_cutoff
        Remove peaks with intensity below this fraction of the largest remaining
        peak in the spectrum. Set to 0 or None to disable relative-intensity
        filtering.
    remove_precursor
        If True and ``precursor_mz`` is available, remove peaks above
        ``precursor_mz + offset_to_precursor`` before scoring.
    offset_to_precursor
        Signed offset in Da used for precursor-region removal.
    use_ppm
        If True, interpret ``tolerance`` as a symmetric ppm tolerance instead
        of absolute Da. Supported by the indexed greedy implementation.
    merge_within
        Optional within-spectrum peak-merging distance in Da. Set to 0 to
        disable merging. Supported by the indexed greedy implementation.
    dtype
        Floating-point dtype used for prepared peak intensities and similarity
        scores. Supported values are ``numpy.float32`` and ``numpy.float64``.

    Returns
    -------
    Scores
        ``matrix`` and ``search`` return a
        :class:`~matchms.scores.Scores` object containing ``"score"`` and
        ``"matches"`` by default. ``"score"`` contains modified-cosine
        similarity values and ``"matches"`` contains the number of accepted
        one-to-one peak matches.

    Notes
    -----
    A persistent index is useful when many query spectra are compared against
    the same reference library. Build the index once::

        similarity = ModifiedCosine(tolerance=0.01)

        library_index = similarity.build_index(library_spectra)

    and reuse it for subsequent query batches::

        scores = similarity.search(
            query_spectra,
            library_index,
            progress_bar=False,
            n_jobs=1,
        )

    ``search`` always returns query spectra as rows and reference-library spectra
    as columns, giving a matrix of shape
    ``(len(query_spectra), len(library_spectra))``.

    The reusable library representation is a
    :class:`~matchms.similarity.flash_index.FlashIndex`. For modified cosine,
    the index contains the information needed for both direct-fragment and
    precursor-dependent shifted matching. Index compatibility with the current
    preprocessing configuration is validated before searching.

    Because neutral-loss matching depends on precursor metadata, accurate
    ``precursor_mz`` values are particularly important for modified cosine.
    Missing precursor values disable shifted matches for the affected spectrum,
    while direct fragment matches remain possible.

    The greedy assignment is based on candidate intensity products rather than
    simply selecting the closest m/z pair. Consequently, when several candidate
    assignments overlap, the accepted peak pairs can differ from a
    nearest-neighbour assignment. ``use_hungarian=True`` instead finds an
    optimal assignment, at substantially higher computational cost.

    If only similarity scores are needed, the dense matched-peak-count output
    can be omitted::

        scores = similarity.search(
            query_spectra,
            library_index,
            score_fields=("score",),
            progress_bar=False,
            n_jobs=1,
        )

    Examples
    --------
    Compare two spectra::

        similarity = ModifiedCosine(tolerance=0.01)
        result = similarity.pair(spectrum_1, spectrum_2)

        modified_cosine = result["score"]
        n_matches = result["matches"]

    Compute a matrix between two collections::

        scores = similarity.matrix(
            reference_spectra,
            query_spectra,
            progress_bar=False,
            n_jobs=1,
        )

    Reuse a reference-library index::

        index = similarity.build_index(reference_spectra)

        first_scores = similarity.search(
            first_query_batch,
            index,
            progress_bar=False,
            n_jobs=1,
        )

        second_scores = similarity.search(
            second_query_batch,
            index,
            progress_bar=False,
            n_jobs=1,
        )
    """

    _default_mode = "hybrid"
