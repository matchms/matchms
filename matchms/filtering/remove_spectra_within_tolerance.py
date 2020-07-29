from ..typing import SpectrumType


def remove_spectra_within_tolerance(spectrum: SpectrumType, mz_tolerance: float = 17) -> SpectrumType:
    
    """Remove peaks that are within mz_tolerance (in Da) of 
       the precursor mz, exlcuding the precursor peak
    
    Args:
    -----
    spectrum:
        Input spectrum.
    mz_tolerance:
        Tolerance of mz values that are not allowed to lie 
        within the precursor mz. Default is 17 Da.
    
    """
    
    assert mz_tolerance >= 0, "mz_tolerance must be a positive floating point."
    precursor_mz = spectrum.get("precursor_mz")
    mzs = spectrum.peaks.mz
    intensities = spectrum.peaks.intensities
    new_mzs = mzs
    new_intensities = intensities
    if precursor_mz:
        for i in range(len(mzs)):
            
            if abs(precursor_mz-mzs[i]) <= mz_tolerance and mzs[i] != precursor_mz:
                new_mzs[i] = np.nan
                new_intensities[i] = np.nan
                
        nans = np.isnan(new_mzs)
        new_mzs = new_mzs[~nans]
        new_intensities = new_intensities[~nans]
        spectrum.peaks = Spikes(mz=new_mzs, intensities=new_intensities)

    return spectrum

