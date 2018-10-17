from typing import AbstractSet

from npsem.model import StructuralCausalModel, CausalDiagram
from npsem.utils import ors


def all_v(xs, v):
    return all(x == v for x in xs)


def all_1(xs):
    return all(x == 1 for x in xs)


def all_0(xs):
    return all(x == 0 for x in xs)


def all_2(xs):
    return all(x == 2 for x in xs)


def V_Q_i(v, Q, i):
    try:
        return v[Q + f'^({i})']
    except KeyError:
        return (int(v[Q]) >> (2 * i)) % 4


def construct_SCM_for_POMIS_empty_W(G: CausalDiagram, Y, T: AbstractSet, X: AbstractSet, W: AbstractSet, jitter=0, verbose=False) -> StructuralCausalModel:
    assert jitter == 0, "jitter not supported yet"
    assert not W, "intervened not supported yet"

    G = G[T | X]

    U_over_MUCT = sorted(G[T].U)

    def P_U(d):
        if verbose:
            print()
            print(f'u={d}')
        prob = pow(0.5, len(U_over_MUCT))
        return prob

    def red_template(i_, U_i_, V_k, S_i_, T_i_, red_):
        def func1(v):
            assert V_k != S_i_
            outside = {v[Q] for Q in G.pa(V_k) - T}
            if ors(outside) != 0:
                return int(2)
            if V_k == T_i_:
                return 1 - v[U_i_]
            else:
                red_parents = {V_Q_i(v, Q, i_) for Q in G.pa(V_k) & red_}
                assert len(red_parents) > 0
                if red_parents in ({0}, {1}):
                    return next(iter(red_parents))
                return int(2)

        func1.__name__ = f'{V_k}^{i}'
        return func1

    def blue_template(i_, U_i_, V_k, S_i_, T_i_, blue_):
        def func1(v):
            assert V_k != T_i_
            outside = {v[Q] for Q in G.pa(V_k) - T}
            if ors(outside) != 0:
                return int(2)
            if V_k == S_i_:
                return int(v[U_i_])
            else:
                blue_parents = {V_Q_i(v, Q, i_) for Q in G.pa(V_k) & blue_}
                assert len(blue_parents) > 0
                if blue_parents in ({0}, {1}):
                    return next(iter(blue_parents))
                return int(2)

        func1.__name__ = f'{V_k}^{i}'
        return func1

    def purple_template(i_, U_i_, V_k, S_i_, T_i_, red_, blue_, purple_):
        def func1(v):
            try:
                outside = {v[Q] for Q in G.pa(V_k) - T}
                if ors(outside) != 0:
                    return int(3)
                blue_parents = {V_Q_i(v, Q, i_) for Q in G.pa(V_k) & blue_}
                red_parents = {V_Q_i(v, Q, i_) for Q in G.pa(V_k) & red_}
                purple_parents = {V_Q_i(v, Q, i_) for Q in G.pa(V_k) & purple_}
                if V_k == S_i_:
                    blue_parents |= {v[U_i_]}
                elif V_k == T_i_:
                    red_parents |= {1 - v[U_i_]}
                if all_1(red_parents) and all_2(purple_parents) and all_0(blue_parents):
                    return int(2)
                elif all_0(red_parents) and all_1(purple_parents) and all_1(blue_parents):
                    return int(1)
                return int(3)
            except KeyError as err:
                print("==========")
                print(f"{V_k}")
                print(f"{G.pa(V_k)}")
                print(f"{red_}")
                print(f"{i_}")
                print(f"{sorted(v.keys())}")
                print("==========", flush=True)
                import time
                time.sleep(1)
                raise err

        func1.__name__ = f'{V_k}^{i}'
        return func1

    # coloring
    functions = dict()
    for i, U_i in enumerate(sorted(U_over_MUCT)):
        functions[i] = dict()
        S_i, T_i = G.confounded_with(U_i)  # blue, red, red 1-u
        assert S_i != T_i
        if S_i < T_i:  # This is only for reproducibility.
            S_i, T_i = T_i, S_i
        red = G.De(T_i)
        blue = G.De(S_i)
        purple = red & blue
        red, blue = red - purple, blue - purple
        assert (red | blue)
        if verbose:
            print(f'{U_i} for ({i}): red={red}, blue={blue}, purple={purple}')

        for V_j in T - (red | blue | purple):
            functions[i][V_j] = lambda v: int(0)

        for V_j in red:
            functions[i][V_j] = red_template(i, U_i, V_j, S_i, T_i, red)

        for V_j in blue:
            functions[i][V_j] = blue_template(i, U_i, V_j, S_i, T_i, blue)

        for V_j in purple:
            functions[i][V_j] = purple_template(i, U_i, V_j, S_i, T_i, red, blue, purple)
    pass

    # time to integrate functions

    def merge_template(V_k):
        def func00(v):
            summed = 0
            for i0 in range(len(U_over_MUCT)):
                V_k_i = v[V_k + f'^({i0})'] = functions[i0][V_k](v)
                if verbose:
                    print(f'{V_k}^({i0}) = {functions[i0][V_k](v)}')
                summed += pow(4, i0) * V_k_i
            if V_k != Y:
                if verbose:
                    print(f'{V_k}     = {summed}')
            return int(summed)

        func00.__name__ = f'{V_k}'
        return func00

    F = dict()
    for V_i in T:
        if V_i != Y:
            F[V_i] = merge_template(V_i)
        else:
            temp_f = merge_template(V_i)

            def func000(v):
                y_prime = temp_f(v)
                if verbose:
                    print(f"y'    = " + f'{y_prime:b}'.zfill(2 * len(U_over_MUCT)))
                if all(((y_prime >> 2 * bit_i) & 3) in {1, 2} for bit_i in range(len(U_over_MUCT))):
                    if verbose:
                        print(f"y     = 1")
                    return int(1)
                else:
                    if verbose:
                        print(f"y     = 0")
                    return int(0)

            F[V_i] = func000

    D = {u_i: (0, 1) for u_i in G.U}
    # print(list(F.keys()))
    return StructuralCausalModel(G, F, P_U, D)
