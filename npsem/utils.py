from itertools import combinations as itercomb, chain

import numpy as np
import os
from contextlib import contextmanager
from typing import Iterable, TypeVar, Generator, Tuple, Set, List, FrozenSet, AbstractSet

T = TypeVar('T')


def dict_or(d1: dict, d2: dict) -> dict:
    d3 = dict(d1)
    d3.update(d2)
    return d3


def random_seeds(n=None):
    """Random seeds of given size or a random seed if n is None"""
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def subseq(xs, indices):
    if isinstance(xs, tuple):
        return tuple(xs[i] for i in indices)
    else:
        return [xs[i] for i in indices]


def pick_randomly(xs):
    return xs[np.random.randint(len(xs))]


def rand_argmax(xs):
    max_val = np.nanmax(xs)
    if max_val is np.nan:
        return pick_randomly(np.arange(len(xs)))

    max_indices = np.where(xs == max_val)[0]
    if not len(max_indices):
        print(xs, max_val)

    if len(max_indices) == 1:
        return max_indices[0]
    else:
        return pick_randomly(max_indices)


def with_default(x, dflt=None):
    return x if x is not None else dflt


def disjoint(set1: Set, set2: Set) -> bool:
    if len(set2) < len(set1):
        set1, set2 = set2, set1
    return not any(x in set2 for x in set1)


def rand_bw(lower, upper, precision=None):
    assert lower <= upper
    if lower == upper:
        return lower
    if precision is not None:
        return round(np.random.rand() * (upper - lower) + lower, precision)
    else:
        return np.random.rand() * (upper - lower) + lower


@contextmanager
def seeded(seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
        yield
        np.random.set_state(st0)
    else:
        yield


def only(W: List[T], Z: AbstractSet[T]) -> List[T]:
    if not Z:
        return []
    return [w for w in W if w in Z]


def pop(xs: Set):
    x = next(iter(xs))
    xs.remove(x)
    return x


def fzset_union(sets) -> FrozenSet:
    return frozenset(chain(*sets))


def sortup(xs: Iterable[T]) -> Tuple[T, ...]:
    return tuple(sorted(xs))


def sortup2(xxs):
    return sortup([sortup(xs) for xs in xxs])


def shuffled(xs: Iterable[T]) -> List[T]:
    xs = list(xs)
    np.random.shuffle(xs)
    return xs


def combinations(xs: Iterable[T]) -> Generator[Tuple[T, ...], None, None]:
    """ all combinations of given in the order of increasing its size """
    xs = list(xs)
    for i in range(len(xs) + 1):
        for comb in itercomb(xs, i):
            yield comb


def ors(values):
    x = 0
    for v in values:
        x = x | v
    return x


def mkdirs(newdir):
    os.makedirs(newdir, mode=0o777, exist_ok=True)
