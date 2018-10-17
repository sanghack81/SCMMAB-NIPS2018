from npsem.NIPS2018POMIS_exp.construct_pomis import construct_SCM_for_POMIS_empty_W
from npsem.NIPS2018POMIS_exp.scm_examples import XYZWST
from npsem.where_do import MUCT_IB

if __name__ == '__main__':
    G = XYZWST()
    Y = G.causal_order(backward=True)[0]  # choose randomly?
    G = G[G.An(Y)]

    # construct an SCM w/ the causal diagram
    T, X = MUCT_IB(G, Y)
    G = G[T | X]
    M = construct_SCM_for_POMIS_empty_W(G, Y, T, X, set(), verbose=True)

    result = M.query((Y,), intervention={X_i: 0 for X_i in X})
    print(dict(result))
