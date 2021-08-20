from cobaya.model import get_model
from cobaya.run import run
import yaml
import os
import time
import numpy as np
import sacc
import scipy.stats as st
import sys

# Read in the yaml file
config_fn = sys.argv[1]
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Get the mean proposed in the yaml file for each parameter
p0 = {}
p_all = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']
             p_all[p] = p0[p]
     else:
         p_all[p] = info['params'][p]
os.system('mkdir -p ' + info['output'])

print("params_dict = ", p0)

# Compute the likelihood
model = get_model(info)
l = model.likelihood['yxgxk_like.YxGxKLike']
loglikes, derived = model.loglikes(p0)
print("chi2 = ", -2 * loglikes[0])

# Minimize
updated_info, sampler = run(info)
bf = sampler.products()['minimum'].bestfit()
pf = {k: bf[k] for k in p0.keys()}
print("Final params: ")
print(pf)
loglikes, derived = model.loglikes(pf)
chi2 = -2*loglikes[0]
ndata = l.ndata
print("chi2 = ", chi2)
print("p = ", 1-st.chi2.cdf(chi2, ndata))

'''
l = model.likelihood['yxgxk_like.YxGxKLike']
p_all.update(pf)
s_th = l.get_sacc_file(**p_all)
s_dt = sacc.Sacc.load_fits(l.input_file)

import matplotlib.pyplot as plt
for t1, t2 in s_th.get_tracer_combinations():
    plt.figure()
    plt.title(f'{t1}x{t2}')
    ld, cld, cov = s_dt.get_ell_cl('cl_00', t1, t2, return_cov=True)
    lt, clt = s_th.get_ell_cl('cl_00', t1, t2)
    ids = np.array([l in lt for l in ld])
    #plt.errorbar(ld, cld, yerr=np.sqrt(np.diag(cov)), fmt='k.')
    #plt.plot(lt, clt, 'r-')
    #plt.loglog()
    plt.errorbar(lt, (cld[ids]-clt)/np.sqrt(np.diag(cov))[ids], yerr=np.ones(len(lt)), fmt='k.')
    plt.xscale('log')
plt.show()
'''
