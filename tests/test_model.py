"""Tests for npsem.model — CausalDiagram graph ops and SCM.query."""
import pytest

from npsem.model import CausalDiagram
from npsem.NIPS2018POMIS_exp.scm_examples import IV_CD


def test_iv_structure(iv):
    assert set(iv.V) == {"X", "Y", "Z"}
    assert set(iv.edges) == {("Z", "X"), ("X", "Y")}
    assert len(iv.confounded_dict) == 1
    assert set(map(frozenset, iv.confounded_dict.values())) == {frozenset({"X", "Y"})}


def test_ancestors_descendants_parents_children(iv):
    # capital = inclusive of the node itself, lowercase = strict
    assert set(iv.An("Y")) == {"X", "Y", "Z"}
    assert set(iv.an("Y")) == {"X", "Z"}
    assert set(iv.De("Z")) == {"X", "Y", "Z"}
    assert set(iv.de("Z")) == {"X", "Y"}
    assert set(iv.Pa("X")) == {"X", "Z"}
    assert set(iv.pa("X")) == {"Z"}
    assert set(iv.ch("X")) == {"Y"}
    assert set(iv.pa("Z")) == set()          # root has no parents


def test_c_components(iv):
    cc = {frozenset(c) for c in iv.c_components}
    assert cc == {frozenset({"X", "Y"}), frozenset({"Z"})}
    assert set(iv.c_component("X")) == {"X", "Y"}
    assert set(iv.c_component("Z")) == {"Z"}


def test_confounding_predicates(iv):
    assert iv.is_confounded("X", "Y")
    assert not iv.is_confounded("X", "Z")
    assert iv.has_edge("Z", "X")
    assert not iv.has_edge("X", "Z")
    assert set(iv.confounded_withs("X")) == {"Y"}


def test_causal_order_is_topological(graph):
    order = graph.causal_order()
    assert set(order) == set(graph.V)
    pos = {v: i for i, v in enumerate(order)}
    for x, y in graph.edges:
        assert pos[x] < pos[y]
    # backward order is the exact reverse
    assert graph.causal_order(backward=True) == tuple(reversed(order))


def test_do_cuts_incoming_edges_and_severs_confounding(iv):
    h = iv.do("X")
    assert ("Z", "X") not in h.edges          # directed edge into X removed
    assert ("X", "Y") in h.edges              # outgoing edge kept
    assert not h.is_confounded("X", "Y")      # bidirected arc into X severed
    assert len(h.confounded_dict) == 0
    assert set(h.V) == {"X", "Y", "Z"}        # do() keeps the vertex set


def test_induced_subgraph(iv):
    h = iv[{"X", "Y"}]
    assert set(h.V) == {"X", "Y"}
    assert ("X", "Y") in h.edges
    assert h.is_confounded("X", "Y")          # confounder within the subset survives
    assert "Z" not in h.V


def test_vertex_removal_operator(iv):
    h = iv - "Z"
    assert set(h.V) == {"X", "Y"}
    assert ("Z", "X") not in h.edges


def test_equality_ignores_confounder_names():
    a = IV_CD("U_XY")
    b = IV_CD("A_DIFFERENT_NAME")
    assert a == b                              # equality compares {x,y} pairs, not U names
    assert hash(a) == hash(b)


def test_query_normalizes_to_one(iv_scm):
    M, _ = iv_scm
    r = M.query(("Y",), intervention={"X": 0})
    assert r[(0,)] + r[(1,)] == pytest.approx(1.0)


def test_query_iv_interventional_expectations(iv_scm):
    # E[Y] = P(Y=1); devised IV SCM with seed=0 -> deterministic values
    M, _ = iv_scm

    def eY(do):
        return M.query(("Y",), intervention=do)[(1,)]

    assert eY({"X": 0}) == pytest.approx(0.493, abs=1e-6)
    assert eY({"X": 1}) == pytest.approx(0.507, abs=1e-6)
    assert eY({"Z": 0}) == pytest.approx(0.773, abs=1e-6)   # the optimal intervention
    assert eY({"Z": 1}) == pytest.approx(0.227, abs=1e-6)
