"""Tests for npsem.utils — small combinatorial / RNG helpers."""
import numpy as np
import pytest

from npsem.utils import (
    combinations, only, fzset_union, seeded, with_default, sortup, rand_bw,
    rand_argmax,
)


def test_combinations_count_and_order():
    combs = list(combinations(["a", "b", "c"]))
    assert len(combs) == 2 ** 3          # all subsets
    assert combs[0] == ()                # empty set first
    sizes = [len(c) for c in combs]
    assert sizes == sorted(sizes)        # increasing size
    assert {frozenset(c) for c in combs} == {
        frozenset(s) for s in
        [(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]
    }


def test_only_filters_and_preserves_order():
    assert only([3, 1, 2, 4], {1, 2}) == [1, 2]
    assert only([1, 2, 3], set()) == []          # empty Z short-circuits
    assert only([], {1}) == []


def test_fzset_union():
    out = fzset_union([{1, 2}, {2, 3}, {3, 4}])
    assert out == frozenset({1, 2, 3, 4})
    assert isinstance(out, frozenset)
    assert fzset_union([]) == frozenset()


def test_with_default():
    assert with_default(None, 7) == 7
    assert with_default(0, 7) == 0          # 0 is not None -> kept
    assert with_default("x", "y") == "x"


def test_sortup():
    assert sortup([3, 1, 2]) == (1, 2, 3)
    assert sortup({"b", "a"}) == ("a", "b")


def test_rand_bw_within_range():
    for _ in range(100):
        v = rand_bw(0.2, 0.8)
        assert 0.2 <= v <= 0.8
    assert rand_bw(0.5, 0.5) == 0.5          # degenerate range


def test_seeded_is_reproducible():
    with seeded(123):
        a = np.random.rand(5)
    with seeded(123):
        b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_seeded_restores_global_state():
    np.random.seed(0)
    expected = np.random.rand(3)
    np.random.seed(0)
    with seeded(999):                        # must not leak into the outer stream
        np.random.rand(10)
    assert np.array_equal(np.random.rand(3), expected)


def test_rand_argmax_basic_and_ties():
    assert rand_argmax(np.array([0.1, 0.9, 0.3])) == 1
    picks = {int(rand_argmax(np.array([0.5, 0.5, 0.1]))) for _ in range(50)}
    assert picks <= {0, 1}                   # only tied maxima are ever returned
