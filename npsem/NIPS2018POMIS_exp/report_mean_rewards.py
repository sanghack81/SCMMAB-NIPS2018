from itertools import product

from npsem.NIPS2018POMIS_exp.scm_examples import IV_SCM, simple_markovian_SCM, XYZWST_SCM
from npsem.utils import combinations

if __name__ == '__main__':
    for name, (model, p_u) in [('marc', simple_markovian_SCM(seed=0)),
                               ('iv', IV_SCM(True, seed=0)),
                               ('xyzwst', XYZWST_SCM(True, seed=0))]:
        print('=========================================================================')
        print(f'========================={str(name).center(23)}=========================')
        print(p_u)
        for x_var in combinations(model.G.V - {'Y'}):
            for x_val in product(*[(0, 1) for x in x_var]):
                results = model.query(('Y',), intervention=dict(zip(x_var, x_val)))
                print(f'{str(dict(zip(x_var,x_val))).rjust(45)}:   {results[(1,)]:.2f} ({results[(1,)]})')
        print('=========================================================================')
        print('\n\n\n\n')
