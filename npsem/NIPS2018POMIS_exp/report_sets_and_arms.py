from npsem.NIPS2018POMIS_exp.scm_examples import XYZWST, simple_markovian, IV_CD
from npsem.utils import combinations
from npsem.where_do import MISs, POMISs

if __name__ == '__main__':
    for G in [simple_markovian(), IV_CD(), XYZWST()]:
        all_ISs = {frozenset(xx) for xx in combinations(G.V - {'Y'})}
        miss = MISs(G, 'Y')
        pomiss = POMISs(G, 'Y')

        print(f'{len(all_ISs)} ISs')
        print(f'{len(miss)} MISs')
        print(f'{len(pomiss)} POMISs')
        print(f'Brute-force arms: {sum([2**(len(iset)) for iset in all_ISs])}')
        print(f'        MIS arms: {sum([2**(len(mis)) for mis in miss])}')
        print(f'      POMIS arms: {sum([2**(len(pomis)) for pomis in pomiss])}')

        print('POMISs')
        for _, pomis in sorted([(len(pomis), tuple(sorted(pomis))) for pomis in pomiss]):
            print('  {', end='')
            print(*list(pomis), sep=', ', end='')
            print('}')

        print('MISs (but not POMISs)')
        for _, mis in sorted([(len(mis), tuple(sorted(mis))) for mis in miss - pomiss]):
            print('  {', end='')
            print(*list(mis), sep=', ', end='')
            print('}')

        print('ISs (but not MISs)')
        for _, iset in sorted([(len(iset), tuple(sorted(iset))) for iset in all_ISs - miss]):
            print('  {', end='')
            print(*list(iset), sep=', ', end='')
            print('}')

        print('\n ================================================= \n')
