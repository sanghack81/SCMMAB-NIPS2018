import itertools
from collections import defaultdict
from itertools import product

import functools
import networkx as nx
import numpy as np
from typing import Dict, Iterable, Optional, Set, Sequence, AbstractSet
from typing import FrozenSet, Tuple

from npsem.utils import fzset_union, sortup, sortup2, with_default


def default_P_U(mu: Dict):
    """ P(U) function given a dictionary of probabilities for each U_i being 1, P(U_i=1) """

    def P_U(d):
        p_val = 1.0
        for k in mu.keys():
            p_val *= (1 - mu[k]) if d[k] == 0 else mu[k]
        return p_val

    return P_U


def dict_only(a_dict: dict, keys: AbstractSet) -> Dict:
    return {k: a_dict[k] for k in keys if k in a_dict}


def dict_except(a_dict: dict, keys: AbstractSet) -> Dict:
    return {k: v for k, v in a_dict.items() if k not in keys}


def pairs2dict(xys, backward=False):
    dd = defaultdict(set)
    if backward:
        for x, y in xys:
            dd[y].add(x)
    else:
        for x, y in xys:
            dd[x].add(y)

    return defaultdict(frozenset, {key: frozenset(vals) for key, vals in dd.items()})


def wrap(v_or_vs, wrap_with=frozenset):
    if v_or_vs is None:
        return None
    if isinstance(v_or_vs, str):
        return wrap_with({v_or_vs})
    else:
        return wrap_with(v_or_vs)


class CausalDiagram:
    def __init__(self,
                 vs: Optional[Iterable[str]],
                 directed_edges: Optional[Iterable[Tuple[str, str]]] = frozenset(),
                 bidirected_edges: Optional[Iterable[Tuple[str, str, str]]] = frozenset(),
                 copy: 'CausalDiagram' = None,
                 with_do: Optional[Set[str]] = None,
                 with_induced: Optional[Set[str]] = None):
        with_do = wrap(with_do)
        with_induced = wrap(with_induced)
        if copy is not None:
            if with_do is not None:
                self.V = copy.V
                self.U = wrap(u for u in copy.U if with_do.isdisjoint(copy.confounded_dict[u]))
                self.confounded_dict = {u: val for u, val in copy.confounded_dict.items() if u in self.U}

                # copy cautiously
                dopa = copy.pa(with_do)
                doAn = copy.An(with_do)
                doDe = copy.De(with_do)

                self._pa = defaultdict(frozenset, {k: frozenset() if k in with_do else v for k, v in copy._pa.items()})
                self._ch = defaultdict(frozenset, {k: (v - with_do) if k in dopa else v for k, v in copy._ch.items()})
                self._an = dict_except(copy._an, doDe)
                self._de = dict_except(copy._de, doAn)

            elif with_induced is not None:
                assert with_induced <= copy.V
                removed = copy.V - with_induced
                self.V = with_induced
                self.confounded_dict = {u: val for u, val in copy.confounded_dict.items() if val <= self.V}
                self.U = wrap(self.confounded_dict)

                children_are_removed = copy.pa(removed) & self.V
                parents_are_removed = copy.ch(removed) & self.V
                ancestors_are_removed = copy.de(removed) & self.V
                descendants_are_removed = copy.an(removed) & self.V

                self._pa = defaultdict(frozenset, {x: (copy._pa[x] - removed) if x in parents_are_removed else copy._pa[x] for x in self.V})
                self._ch = defaultdict(frozenset, {x: (copy._ch[x] - removed) if x in children_are_removed else copy._ch[x] for x in self.V})
                self._an = dict_only(copy._an, self.V - ancestors_are_removed)
                self._de = dict_only(copy._de, self.V - descendants_are_removed)
            else:
                self.V = copy.V
                self.U = copy.U
                self.confounded_dict = copy.confounded_dict
                self._ch = copy._ch
                self._pa = copy._pa
                self._an = copy._an
                self._de = copy._de
        else:
            directed_edges = list(directed_edges)
            bidirected_edges = list(bidirected_edges)
            self.V = frozenset(vs) | fzset_union(directed_edges) | fzset_union((x, y) for x, y, _ in bidirected_edges)
            self.U = frozenset(u for _, _, u in bidirected_edges)
            self.confounded_dict = {u: frozenset({x, y}) for x, y, u in
                                    bidirected_edges}

            self._ch = pairs2dict(directed_edges)
            self._pa = pairs2dict(directed_edges, backward=True)
            self._an = dict()  # cache
            self._de = dict()  # cache
            assert self._ch.keys() <= self.V and self._pa.keys() <= self.V

        self.edges = tuple((x, y) for x, ys in self._ch.items() for y in ys)
        self.causal_order = functools.lru_cache()(self.causal_order)
        self._do_ = functools.lru_cache()(self._do_)
        self.__cc = None
        self.__cc_dict = None
        self.__h = None
        self.__characteristic = None
        self.__confoundeds = None
        self.u_pas = defaultdict(set)
        for u, xy in self.confounded_dict.items():
            for v in xy:
                self.u_pas[v].add(u)
        self.u_pas = defaultdict(set, {v: frozenset(us) for v, us in self.u_pas.items()})

    def UCs(self, v):
        return self.u_pas[v]

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.V or item in self.U
        if len(item) == 2:
            if isinstance(item, AbstractSet):
                x, y = item
                return self.is_confounded(x, y)
            else:
                return tuple(item) in self.edges
        if len(item) == 3:
            x, y, u = item
            return self.is_confounded(x, y) and u in self.confounded_dict and self.confounded_dict[u] == frozenset({x, y})
        return False

    def __lt__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self <= other and self != other

    def __le__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self.V <= other.V and set(self.edges) <= set(other.edges) and set(self.confounded_dict.values()) <= set(other.confounded_dict.values())

    def __ge__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self.V >= other.V and set(self.edges) >= set(other.edges) and set(self.confounded_dict.values()) >= set(other.confounded_dict.values())

    def __gt__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        return self >= other and self != other

    def Pa(self, v_or_vs) -> FrozenSet:
        return self.pa(v_or_vs) | wrap(v_or_vs, frozenset)

    def pa(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self._pa[v_or_vs]
        else:
            return fzset_union(self._pa[v] for v in v_or_vs)

    def ch(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self._ch[v_or_vs]
        else:
            return fzset_union(self._ch[v] for v in v_or_vs)

    def Ch(self, v_or_vs) -> FrozenSet:
        return self.ch(v_or_vs) | wrap(v_or_vs, frozenset)

    def An(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__an(v_or_vs) | {v_or_vs}
        return self.an(v_or_vs) | wrap(v_or_vs, frozenset)

    def an(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__an(v_or_vs)
        return fzset_union(self.__an(v) for v in wrap(v_or_vs))

    def De(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__de(v_or_vs) | {v_or_vs}
        return self.de(v_or_vs) | wrap(v_or_vs, frozenset)

    def de(self, v_or_vs) -> FrozenSet:
        if isinstance(v_or_vs, str):
            return self.__de(v_or_vs)
        return fzset_union(self.__de(v) for v in wrap(v_or_vs))

    def __an(self, v) -> FrozenSet:
        if v in self._an:
            return self._an[v]
        self._an[v] = fzset_union(self.__an(parent) for parent in self._pa[v]) | self._pa[v]
        return self._an[v]

    def __de(self, v) -> FrozenSet:
        if v in self._de:
            return self._de[v]
        self._de[v] = fzset_union(self.__de(child) for child in self._ch[v]) | self._ch[v]
        return self._de[v]

    def do(self, v_or_vs) -> 'CausalDiagram':
        return self._do_(wrap(v_or_vs))

    def _do_(self, v_or_vs) -> 'CausalDiagram':
        return CausalDiagram(None, None, None, self, wrap(v_or_vs))

    def has_edge(self, x, y) -> bool:
        return y in self._ch[x]

    def is_confounded(self, x, y) -> bool:
        return {x, y} in self.confounded_dict.values()

    def u_of(self, x, y):
        key = {x, y}
        for u, ab in self.confounded_dict.items():
            if ab == key:
                return u
        return None

    def confounded_with(self, u):
        return self.confounded_dict[u]

    def confounded_withs(self, v):
        return {next(iter(xy - {v})) for xy in self.confounded_dict.values() if v in xy}

    def __getitem__(self, item) -> 'CausalDiagram':
        return self.induced(item)

    def induced(self, v_or_vs) -> 'CausalDiagram':
        if set(v_or_vs) == self.V:
            return self
        return CausalDiagram(None, None, None, copy=self, with_induced=v_or_vs)

    @property
    def characteristic(self):
        if self.__characteristic is None:
            self.__characteristic = (len(self.V),
                                     len(self.edges),
                                     len(self.confounded_dict),
                                     sortup([(len(self.ch(v)), len(self.pa(v)), len(self.confounded_withs(v))) for v in self.V]))
        return self.__characteristic

    def edges_removed(self, edges_to_remove: Iterable[Sequence[str]]) -> 'CausalDiagram':
        edges_to_remove = [tuple(edge) for edge in edges_to_remove]

        dir_edges = {edge for edge in edges_to_remove if len(edge) == 2}
        bidir_edges = {edge for edge in edges_to_remove if len(edge) == 3}
        bidir_edges = frozenset((*sorted([x, y]), u) for x, y, u in bidir_edges)
        return CausalDiagram(self.V, set(self.edges) - dir_edges, self.confounded_to_3tuples() - bidir_edges)

    def __sub__(self, v_or_vs_or_edges) -> 'CausalDiagram':
        if not v_or_vs_or_edges:
            return self
        if isinstance(v_or_vs_or_edges, str):
            return self[self.V - wrap(v_or_vs_or_edges)]
        v_or_vs_or_edges = list(v_or_vs_or_edges)
        if isinstance(v_or_vs_or_edges[0], str):
            return self[self.V - wrap(v_or_vs_or_edges)]
        return self.edges_removed(v_or_vs_or_edges)

    def causal_order(self, backward=False) -> Tuple:
        gg = nx.DiGraph(self.edges)
        gg.add_nodes_from(self.V)
        top_to_bottom = list(nx.topological_sort(gg))
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)

    def __add__(self, edges):
        if isinstance(edges, CausalDiagram):
            return merge_two_cds(self, edges)

        directed_edges = {edge for edge in edges if len(edge) == 2}
        bidirected_edges = {edge for edge in edges if len(edge) == 3}
        return CausalDiagram(self.V, set(self.edges) | directed_edges, self.confounded_to_3tuples() | bidirected_edges)

    def __ensure_confoundeds_cached(self):
        if self.__confoundeds is None:
            self.__confoundeds = dict()
            for u, (x, y) in self.confounded_dict.items():
                if x not in self.__confoundeds:
                    self.__confoundeds[x] = set()
                if y not in self.__confoundeds:
                    self.__confoundeds[y] = set()
                self.__confoundeds[x].add(y)
                self.__confoundeds[y].add(x)
            self.__confoundeds = {x: frozenset(ys) for x, ys in self.__confoundeds.items()}
            for v in self.V:
                if v not in self.__confoundeds:
                    self.__confoundeds[v] = frozenset()

    def __ensure_cc_cached(self):
        if self.__cc is None:
            self.__ensure_confoundeds_cached()
            ccs = []
            remain = set(self.V)
            found = set()
            while remain:
                v = next(iter(remain))
                a_cc = set()
                to_expand = [v]
                while to_expand:
                    v = to_expand.pop()
                    a_cc.add(v)
                    to_expand += list(self.__confoundeds[v] - a_cc)
                ccs.append(a_cc)
                found |= a_cc
                remain -= found
            self.__cc2 = frozenset(frozenset(a_cc) for a_cc in ccs)
            self.__cc_dict2 = {v: a_cc for a_cc in self.__cc2 for v in a_cc}

            self.__cc = self.__cc2
            self.__cc_dict = self.__cc_dict2

    @property
    def c_components(self) -> FrozenSet:
        self.__ensure_cc_cached()
        return self.__cc

    def c_component(self, v_or_vs) -> FrozenSet:
        assert isinstance(v_or_vs, str)
        self.__ensure_cc_cached()
        return fzset_union(self.__cc_dict[v] for v in wrap(v_or_vs))

    def confounded_to_3tuples(self) -> FrozenSet[Tuple[str, str, str]]:
        return frozenset((*sorted([x, y]), u) for u, (x, y) in self.confounded_dict.items())

    def __eq__(self, other):
        if not isinstance(other, CausalDiagram):
            return False
        if self.V != other.V:
            return False
        if set(self.edges) != set(other.edges):
            return False
        if set(self.confounded_dict.values()) != set(other.confounded_dict.values()):  # does not care about U's name
            return False
        return True

    def __hash__(self):
        if self.__h is None:
            self.__h = hash(sortup(self.V)) ^ hash(sortup(self.edges)) ^ hash(sortup2(self.confounded_dict.values()))
        return self.__h

    def __repr__(self):
        return cd2qcd(self)

    def __str__(self):
        nxG = nx.DiGraph(sortup(self.edges))
        paths = []
        while nxG.edges:
            path = nx.dag_longest_path(nxG)
            paths.append(path)
            for x, y in zip(path, path[1:]):
                nxG.remove_edge(x, y)
        nxG = nx.Graph([(x, y) for x, y in self.confounded_dict.values()])
        bipaths = []
        while nxG.edges:
            temppaths = []
            for x, y in itertools.combinations(sortup(nxG.nodes), 2):
                for spath in nx.all_simple_paths(nxG, x, y):
                    temppaths.append(spath)
            selected = sorted(temppaths, key=lambda _spath: len(_spath), reverse=True)[0]
            bipaths.append(selected)
            for x, y in zip(selected, selected[1:]):
                nxG.remove_edge(x, y)

        modified = True
        while modified:
            modified = False
            for i, path1 in enumerate(bipaths):
                for j, path2 in enumerate(bipaths[i + 1:], i + 1):
                    if path1[-1] == path2[0]:
                        newpath = path1 + path2[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[0] == path2[-1]:
                        newpath = path2 + path1[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[0] == path2[0]:
                        newpath = list(reversed(path2)) + path1[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                    elif path1[-1] == path2[-1]:
                        newpath = path2 + list(reversed(path1))[1:]
                        bipaths.pop(j)
                        bipaths[i] = newpath
                        break
                modified = path1 != bipaths[i]
                if modified:
                    break

        # a -> b -> c
        # e -> d -> c
        # == a->b->c<-d<-e
        paths_string = [' ⟶ '.join(path) for path in paths]
        bipaths_string = [' ⟷ '.join(path) for path in bipaths]
        alone = self.V - {x for path in paths for x in path} - {x for path in bipaths for x in path}
        if alone:
            return f'[{",".join([str(x) for x in alone])} / ' + (', '.join(paths_string) + ' / ' + ', '.join(bipaths_string)) + ']'
        else:
            return f'[' + (', '.join(paths_string) + ' / ' + ', '.join(bipaths_string)) + ']'


class StructuralCausalModel:
    def __init__(self, G: CausalDiagram, F=None, P_U=None, D=None, more_U=None):
        self.G = G
        self.F = F
        self.P_U = P_U
        self.D = with_default(D, defaultdict(lambda: (0, 1)))
        self.more_U = set() if more_U is None else set(more_U)

        self.query00 = functools.lru_cache(1024)(self.query00)

    def query(self, outcome: Tuple, condition: dict = None, intervention: dict = None, verbose=False) -> defaultdict:
        if condition is None:
            condition = dict()
        if intervention is None:
            intervention = dict()
        new_condition = tuple(sorted([(x, y) for x, y in condition.items()]))
        new_intervention = tuple(sorted([(x, y) for x, y in intervention.items()]))
        return self.query00(outcome, new_condition, new_intervention, verbose)

    def query00(self, outcome: Tuple, condition: Tuple, intervention: Tuple, verbose=False) -> defaultdict:
        condition = dict(condition)
        intervention = dict(intervention)

        prob_outcome = defaultdict(lambda: 0)

        U = list(sorted(self.G.U | self.more_U))
        D = self.D
        P_U = self.P_U
        V_ordered = self.G.causal_order()
        if verbose:
            print(f"ORDER: {V_ordered}")
        normalizer = 0

        for u in product(*[D[U_i] for U_i in U]):  # d^|U|
            assigned = dict(zip(U, u))
            p_u = P_U(assigned)
            if p_u == 0:
                continue
            # evaluate values
            for V_i in V_ordered:
                if V_i in intervention:
                    assigned[V_i] = intervention[V_i]
                else:
                    assigned[V_i] = self.F[V_i](assigned)  # pa_i including unobserved

            if not all(assigned[V_i] == condition[V_i] for V_i in condition):
                continue
            normalizer += p_u
            prob_outcome[tuple(assigned[V_i] for V_i in outcome)] += p_u

        if prob_outcome:
            # normalize by prob condition
            return defaultdict(lambda: 0, {k: v / normalizer for k, v in prob_outcome.items()})
        else:
            return defaultdict(lambda: np.nan)  # nan or 0?


def quick_causal_diagram(paths, bidirectedpaths=None):
    if bidirectedpaths is None:
        bidirectedpaths = []
    dir_edges = []
    for path in paths:
        for x, y in zip(path, path[1:]):
            dir_edges.append((x, y))
    bidir_edges = []
    u_count = 0
    for path in bidirectedpaths:
        for x, y in zip(path, path[1:]):
            bidir_edges.append((x, y, 'U' + str(u_count)))
            u_count += 1
    return CausalDiagram(set(), dir_edges, bidir_edges)


qcd = quick_causal_diagram


def merge_two_cds(g1: CausalDiagram, g2: CausalDiagram) -> CausalDiagram:
    VV = g1.V | g2.V
    EE = set(g1.edges) | set(g2.edges)
    VWU = set(g1.confounded_to_3tuples()) | set(g2.confounded_to_3tuples())
    return CausalDiagram(VV, EE, VWU)


def cd2qcd(G: CausalDiagram) -> str:
    nxG = nx.DiGraph(sortup(G.edges))
    paths = []
    while nxG.edges:
        path = nx.dag_longest_path(nxG)
        paths.append(path)
        for x, y in zip(path, path[1:]):
            nxG.remove_edge(x, y)
    nxG = nx.Graph([(x, y) for x, y in G.confounded_dict.values()])
    bipaths = []
    while nxG.edges:
        temppaths = []
        for x, y in itertools.combinations(sortup(nxG.nodes), 2):
            for spath in nx.all_simple_paths(nxG, x, y):
                temppaths.append(spath)
        selected = sorted(temppaths, key=lambda _spath: len(_spath), reverse=True)[0]
        bipaths.append(selected)
        for x, y in zip(selected, selected[1:]):
            nxG.remove_edge(x, y)

    if all(len(v) == 1 for path in paths for v in path) and all(len(v) == 1 for path in bipaths for v in path):
        paths = [''.join(path) for path in paths]
        bipaths = [''.join(path) for path in bipaths]

    return f'qcd({paths}, {bipaths})'
