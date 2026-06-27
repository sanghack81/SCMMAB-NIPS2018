"""Tests for npsem.viz_util — plotting helpers (no actual rendering)."""
import numpy as np
import pytest

import npsem.viz_util as vu
from npsem.viz_util import sparse_index, enable_usetex


def test_sparse_index_returns_everything_when_short():
    assert np.array_equal(sparse_index(50), np.arange(50))      # 50 <= 2*100


@pytest.mark.parametrize("length", [201, 500, 1000, 9999, 10000])
def test_sparse_index_is_sorted_in_range_and_hits_endpoints(length):
    idx = sparse_index(length, base_size=100)
    assert idx[0] == 0
    assert idx[-1] == length - 1                                # last point included
    assert np.all(np.diff(idx) > 0)                             # strictly increasing
    assert idx.max() < length                                  # valid as an index
    assert len(idx) <= 2 * 100 + 2                              # actually sparse


def test_enable_usetex_falls_back_without_latex(monkeypatch):
    import matplotlib as mpl
    monkeypatch.setattr(vu.shutil, "which", lambda name: None)
    with pytest.warns(UserWarning):
        result = enable_usetex(r"\usepackage{foo}")
    assert result is False
    assert mpl.rcParams["text.usetex"] is False


def test_enable_usetex_enables_when_latex_present(monkeypatch):
    import matplotlib as mpl
    monkeypatch.setattr(vu.shutil, "which", lambda name: "/usr/bin/latex")
    result = enable_usetex(r"\usepackage{bar}")
    assert result is True
    assert mpl.rcParams["text.usetex"] is True
    assert mpl.rcParams["text.latex.preamble"] == r"\usepackage{bar}"
