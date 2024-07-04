from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matchms.utils import rename_deprecated_params


_annotation_kws = {
    "horizontalalignment": "left",  # if not mirror_intensity else "right",
    "verticalalignment": "center",
    "fontsize": 7,
    "rotation": 90,
    "rotation_mode": "anchor",
    "zorder": 5,
}


def plot_spectrum(spectrum,
                  annotate_ions: bool = False,
                  mirror_intensity: bool = False,
                  grid: Union[bool, str] = True,
                  ax: plt.Axes = None,
                  peak_color="teal",
                  min_mz: float = None,
                  max_mz: float = None,
                  **plt_kwargs) -> plt.Axes:
    """
    Plot a single MS/MS spectrum.

    Code is largely taken from package "spectrum_utils".

    Parameters
    ----------
    spectrum: matchms.Spectrum
        The spectrum to be plotted.
    annotate_ions:
        Flag indicating whether or not to annotate fragment using peak comments
        (if present in the spectrum). The default is True.
    mirror_intensity:
        Flag indicating whether to flip the intensity axis or not.
    grid:
        Draw grid lines or not. Either a boolean to enable/disable both major
        and minor grid lines or 'major'/'minor' to enable major or minor grid
        lines respectively.
    ax:
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.
    peak_color:
        Set color of peaks in plot.
    min_mz:
        Set lower limit of the plots x-axis to min_mz. Default is None.
    max_mz:
        Set upper limit of the plots x-axis to min_mz. Default is None.

    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectrum is plotted.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if ax is None:
        ax = plt.gca()

    if min_mz is None:
        min_mz = max(0, np.floor(spectrum.peaks.mz[0] / 100 - 1) * 100)
    if max_mz is None:
        max_mz = np.ceil(spectrum.peaks.mz[-1] / 100 + 1) * 100
    max_intensity = spectrum.peaks.intensities.max()

    intensities = spectrum.peaks.intensities / max_intensity

    def make_stems():
        """calculate where the stems of the spectrum peaks are going to be"""
        x = np.zeros([2, spectrum.peaks.mz.size], dtype="float")
        y = np.zeros(x.shape)
        x[:, :] = np.tile(spectrum.peaks.mz, (2, 1))
        y[1, :] = intensities
        return x, y

    x, y = make_stems()
    if mirror_intensity is True:
        y = -y
    ax.plot(x, y, color=peak_color, linewidth=1.0, marker="", zorder=5, **plt_kwargs)
    if annotate_ions and isinstance(spectrum.get("peak_comments"), dict):
        for mz, comment in spectrum.get("peak_comments").items():
            idx = (-abs(spectrum.peaks.mz - mz)).argmax()
            ax.text(mz, intensities[idx], f"m/z: {mz} \n {comment}",
                    _annotation_kws)

    ax.set_xlim(min_mz, max_mz)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    y_max = 1.25 if annotate_ions else 1.10
    ax.set_ylim(*(0, y_max) if not mirror_intensity else (-y_max, 0))

    ax.xaxis.set_minor_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    if grid in (True, "both", "major"):
        ax.grid(visible=True, which="major", color="#9E9E9E", linewidth=0.2)
    if grid in (True, "both", "minor"):
        ax.grid(visible=True, which="minor", color="#9E9E9E", linewidth=0.2)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize="small")
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[y_ticks <= 1.0])

    ax.set_xlabel("m/z", style="italic")
    ax.set_ylabel("Intensity")
    title = "Spectrum" if spectrum.get("compound_name") is None else spectrum.get("compound_name")
    ax.set_title(title)
    return ax


def plot_spectra_mirror(spec_top,
                        spec_bottom,
                        ax: Optional[plt.Axes] = None,
                        color_top="darkblue",
                        color_bottom="teal",
                        **spectrum_kws) -> plt.Axes:
    """Mirror plot two MS/MS spectra.

    Code is largely taken from package "spectrum_utils".

    Parameters
    ----------
    spec_top: matchms.Spectrum.Spectrum.Spectrum
        The spectrum to be plotted on the top.
    spec_bottom: matchms.Spectrum.Spectrum.Spectrum
        The spectrum to be plotted on the bottom.
    ax:
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.
    color_top:
        Set color of peaks in top spectrum.
    color_bottom:
        Set color of peaks in bottom spectrum.
    spectrum_kws:
        Keyword arguments for `plot_spectrum()`.

    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectra are plotted.
    """
    if ax is None:
        ax = plt.gca()

    if spectrum_kws is None:
        spectrum_kws = {}

    if 'peak_color' in spectrum_kws:
        raise ValueError("'peak_color' should not be set for `plot_spectra_mirror`. "
                         "Use 'color_top' and 'color_bottom' instead.")

    # Top spectrum.
    plot_spectrum(spec_top, mirror_intensity=False, ax=ax, peak_color=color_top, **spectrum_kws)
    y_max = ax.get_ylim()[1]

    # Mirrored bottom spectrum.
    plot_spectrum(spec_bottom, mirror_intensity=True, ax=ax, peak_color=color_bottom, **spectrum_kws)
    y_min = ax.get_ylim()[0]
    ax.set_ylim(y_min, y_max)

    ax.axhline(0, color="#9E9E9E", zorder=10)

    # Update axes so that both spectra fit.
    min_mz = max(
        [
            0,
            np.floor(spec_top.peaks.mz[0] / 100 - 1) * 100,
            np.floor(spec_bottom.peaks.mz[0] / 100 - 1) * 100,
        ]
    )
    max_mz = max(
        [
            np.ceil(spec_top.peaks.mz[-1] / 100 + 1) * 100,
            np.ceil(spec_bottom.peaks.mz[-1] / 100 + 1) * 100,
        ]
    )

    # # Enforce min_mz and max_mz if provided
    max_mz = spectrum_kws.get("max_mz", max_mz)
    min_mz = spectrum_kws.get("min_mz", min_mz)

    ax.set_xlim(min_mz, max_mz)
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0%}")
    )

    name1 = "Spectrum 1" if spec_top.get("compound_name") is None else spec_top.get("compound_name")
    name2 = "Spectrum 2" if spec_bottom.get("compound_name") is None else spec_bottom.get("compound_name")

    x_text = 0.04 * (max_mz - min_mz) + min_mz
    ax.text(x_text, y_max, name1, ha="left", va="top", zorder=2, backgroundcolor="white")
    ax.text(x_text, y_min, name2, ha="left", va="bottom", zorder=2, backgroundcolor="white")
    ax.set_title("Spectrum comparison")
    return ax


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def plot_spectra_array(spectra,
                       n_cols: int = 2,
                       peak_color="darkblue",
                       dpi: int = 200,
                       title: str = None,
                       **spectrum_kws) -> plt.Axes:
    """Mirror plot two MS/MS spectra.

    Code is largely taken from package "spectrum_utils".

    Parameters
    ----------
    spectra: list of matchms.Spectrum.Spectrum
        List of spectra to be plotted in a single figure.
    n_cols:
        Number of spectra to be plotted per row. Default is 4.
    spectrum_kws:
        Keyword arguments for `plot_spectrum()`.

    """
    # pylint: disable=too-many-locals
    assert isinstance(spectra, list), "Expected list of Spectrum objects as input."
    n_spectra = len(spectra)
    n_rows = int(np.ceil(n_spectra / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3 * n_rows), dpi=dpi, squeeze=False)

    if spectrum_kws is None:
        spectrum_kws = {}

    for i in range(n_rows):
        for j in range(n_cols):
            counter = i * n_cols + j
            if counter >= n_spectra:
                break

            plot_spectrum(spectra[counter],
                          mirror_intensity=False, ax=axes[i, j],
                          peak_color=peak_color, **spectrum_kws)
            axes[i, j].set_title("")
            if spectra[counter].get("compound_name") is None:
                name = f"Spectrum {i * n_cols + j}"
            else:
                name = spectra[counter].get("compound_name")

            y_max = axes[i, j].get_ylim()[1]
            x_min = axes[i, j].get_xlim()[0]
            axes[i, j].text(x_min, y_max, name, va="bottom", zorder=2)

    if title is None:
        fig.suptitle("Spectrum array plot")
    else:
        fig.suptitle(title)
    return fig, axes
