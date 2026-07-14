"""Runtime compatibility patches for third-party audio libraries."""

try:
    import numpy as _np

    if not hasattr(_np, "NaN"):
        _np.NaN = _np.nan
    if not hasattr(_np, "NAN"):
        _np.NAN = _np.nan
except Exception:
    pass
