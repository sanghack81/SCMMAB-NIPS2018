"""Shared fixtures for the npsem test suite.

The example graphs / SCMs come from the NeurIPS-2018 paper code. SCMs are built
with ``seed=0`` so every expectation below is deterministic and reproducible.
"""
import pytest

from npsem.NIPS2018POMIS_exp.scm_examples import (
    IV_CD, XYZW, XYZWST, simple_markovian,
    IV_SCM, XYZW_SCM, XYZWST_SCM, simple_markovian_SCM,
)

# name -> zero-arg causal-diagram builder
_GRAPHS = {
    "IV": IV_CD,
    "XYZW": XYZW,
    "XYZWST": XYZWST,
    "markovian": simple_markovian,
}

# name -> zero-arg SCM builder (returns (M, mu_params)), all seeded for determinism
_SCMS = {
    "IV": lambda: IV_SCM(True, seed=0),
    "XYZW": lambda: XYZW_SCM(True, seed=0),
    "XYZWST": lambda: XYZWST_SCM(True, seed=0),
    "markovian": lambda: simple_markovian_SCM(seed=0),
}


@pytest.fixture
def iv():
    """The Instrumental-Variable causal diagram:  Z -> X -> Y,  X <-> Y."""
    return IV_CD()


@pytest.fixture
def iv_scm():
    """The devised IV SCM (seed=0):  (M, mu_params)."""
    return IV_SCM(True, seed=0)


@pytest.fixture(params=list(_GRAPHS), ids=list(_GRAPHS))
def graph(request):
    """Parametrized over every example causal diagram."""
    return _GRAPHS[request.param]()


@pytest.fixture(params=list(_SCMS), ids=list(_SCMS))
def scm(request):
    """Parametrized over every example SCM; yields (M, mu_params)."""
    return _SCMS[request.param]()
