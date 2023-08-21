import numpy as np
import rosatX as rx
import emcee
import sys


whichbins = sys.argv[1]
params_vary = sys.argv[2].split(',')
fname_out = sys.argv[3]
if whichbins == 'all':
    bins = [0, 1, 2, 3]
else:
    bins = [int(whichbins)]

pars_BF = {'lMc': 14.2, 'alpha_T': 1.0,
           'eta_b': 0.5, 'gamma': 1.19}
l = rx.ROSATxLike(bins=bins, params_vary=params_vary)
pBF = np.array([pars_BF[k] for k in params_vary])
chi2 = -2*l.logp(pBF)
print(chi2, l.ndata)

npars = len(pBF)
nwalkers = 2*npars
nsteps = 10000

pos = (pBF +
       0.001 * np.random.randn(nwalkers, npars))
sampler = emcee.EnsembleSampler(nwalkers, npars, l.logp)
sampler.run_mcmc(pos, nsteps, progress=True)

chain = sampler.get_chain(discard=0, flat=True)
probs = sampler.get_log_prob(flat=True)
np.savez(fname_out, chain=chain, probs=probs)
