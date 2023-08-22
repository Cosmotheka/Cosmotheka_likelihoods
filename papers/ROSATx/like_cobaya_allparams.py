import numpy as np
import rosatX as rx
import emcee
import sys


whichbins = sys.argv[1]
fname_out = sys.argv[2]
if whichbins == 'all':
    bins = [0, 1, 2, 3]
else:
    bins = [int(whichbins)]

params_vary = ['lMc', 'alpha_T', 'eta_b', 'gamma']
pars_BF = {'lMc': 14.2, 'alpha_T': 1.0,
           'eta_b': 0.5, 'gamma': 1.19}
l = rx.ROSATxLike(bins=bins,
                  params_vary=params_vary)


def logp_cobaya(lMc, alpha_T, eta_b, gamma):
    return l.logp([lMc, alpha_T, eta_b, gamma])

pBF = np.array([pars_BF[k] for k in params_vary])


info = {'likelihood':{'rosatx': logp_cobaya}}
info['params'] = {k: {'prior': {'min': l.priors[k][0], 'max': l.priors[k][1]},
                      'ref': pars_BF[k], 'proposal': 0.01}
                  for k in params_vary}
info["sampler"] = {'mcmc':
                   {'max_samples': 1000,
                    'Rminus1_stop': 0.01}}
info['output'] = 'chaintest'
chi2 = -2*logp_cobaya(*pBF)
print(chi2, l.ndata)
print(info)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from cobaya.run import run
from cobaya.log import LoggedError
success = False
try:
    updated_info, sampler = run(info)
    success = True
except LoggedError as err:
    pass

## Did it work? (e.g. did not get stuck)
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")
print("OK")
