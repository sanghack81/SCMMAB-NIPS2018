"""Tests for npsem.where_do — POMIS / MIS / MUCT / IB.

The strongest check is that the fast POMIS algorithm agrees with the brute-force
oracle on every example graph.
"""
import pytest

from npsem.where_do import (
    POMISs, MISs, bruteforce_POMISs, MUCT, IB, MUCT_IB, minimal_do,
)
from npsem.NIPS2018POMIS_exp.scm_examples import IV_CD, XYZW, XYZWST, simple_markovian


# ---- invariants that must hold for every graph ----

def test_pomis_equals_bruteforce(graph):
    assert set(POMISs(graph, "Y")) == set(bruteforce_POMISs(graph, "Y"))


def test_pomis_is_subset_of_mis(graph):
    assert set(POMISs(graph, "Y")) <= set(MISs(graph, "Y"))


def test_empty_set_is_always_a_mis(graph):
    # doing nothing is always a (minimal) intervention set
    assert frozenset() in MISs(graph, "Y")


def test_muct_contains_reward(graph):
    assert "Y" in MUCT(graph, "Y")


def test_muct_and_ib_are_disjoint(graph):
    Ts, Xs = MUCT_IB(graph, "Y")
    assert Ts.isdisjoint(Xs)
    assert Ts == MUCT(graph, "Y")
    assert Xs == IB(graph, "Y")


def test_results_are_frozensets(graph):
    pomiss = POMISs(graph, "Y")
    assert all(isinstance(s, frozenset) for s in pomiss)


# ---- exact, pinned values per graph ----

@pytest.mark.parametrize("builder, expected", [
    (IV_CD,            {frozenset({"X"}), frozenset({"Z"})}),
    (XYZW,             {frozenset({"W", "X"}), frozenset({"W"}), frozenset()}),
    (simple_markovian, {frozenset({"X1", "X2"})}),
])
def test_known_pomis_sets(builder, expected):
    assert set(POMISs(builder(), "Y")) == expected


@pytest.mark.parametrize("builder, n_pomis, n_mis", [
    (IV_CD, 2, 3),
    (XYZW, 3, 6),
    (XYZWST, 3, 18),
    (simple_markovian, 1, 13),
])
def test_known_counts(builder, n_pomis, n_mis):
    G = builder()
    assert len(POMISs(G, "Y")) == n_pomis
    assert len(MISs(G, "Y")) == n_mis


@pytest.mark.parametrize("builder, muct, ib", [
    (IV_CD,            {"X", "Y"},           {"Z"}),
    (XYZW,             {"W", "X", "Y", "Z"}, set()),
    (XYZWST,           {"W", "X", "Y", "Z"}, {"S", "T"}),
    (simple_markovian, {"Y"},                {"X1", "X2"}),
])
def test_known_muct_ib(builder, muct, ib):
    G = builder()
    assert set(MUCT(G, "Y")) == muct
    assert set(IB(G, "Y")) == ib


def test_minimal_do_is_subset_and_idempotent():
    G = XYZWST()
    full = G.V - {"Y"}
    md = minimal_do(G, "Y", full)
    assert md <= full
    assert minimal_do(G, "Y", md) == md           # idempotent
