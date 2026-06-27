import shutil
import warnings

import numpy as np

""" Drawing some plots for MAB or etc. """


def enable_usetex(preamble: str | None = None) -> bool:
    """Enable LaTeX text rendering, falling back to mathtext if LaTeX is absent.

    The paper's figures use ``usetex=True``; on a machine without a LaTeX
    installation that would crash at render time. We probe for a ``latex``
    binary first. Returns True if usetex was enabled, False if it fell back
    (figures still render via mathtext, only the fonts differ from the paper).
    """
    import matplotlib as mpl

    if shutil.which('latex') is None:
        warnings.warn(
            "LaTeX not found on PATH; falling back to usetex=False "
            "(figures still render via mathtext, but fonts differ from the paper).",
            stacklevel=2,
        )
        mpl.rcParams['text.usetex'] = False
        return False
    mpl.rcParams['text.usetex'] = True
    if preamble is not None:
        mpl.rcParams['text.latex.preamble'] = preamble
    return True


def sparse_index(length, base_size=100):
    if length <= 2 * base_size:
        return np.arange(length)
    step = length // base_size  # >= 2
    if length % step == 0:
        temp = np.arange(1 + (length // step)) * step  # include length
        temp[-1] = length - 1
        return temp
    else:
        if (length // step) * step == length - 1:
            return np.arange(1 + (length // step)) * step
        else:
            temp = np.arange(2 + (length // step)) * step
            assert temp[-2] < length - 1
            temp[-1] = length - 1
            return temp
