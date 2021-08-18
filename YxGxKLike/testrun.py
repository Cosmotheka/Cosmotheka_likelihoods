from cobaya.model import get_model
from cobaya.run import run
import yaml
import os
import time
import numpy as np
import sacc


# Read in the yaml file
config_fn = 'test.yml'
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

s_th = l.get_sacc_file(**p_all)
s_dt = sacc.Sacc.load_fits(l.input_file)

import matplotlib.pyplot as plt
for t1, t2 in s_th.get_tracer_combinations():
    plt.figure()
    plt.title(f'{t1}x{t2}')
    ld, cld, cov = s_dt.get_ell_cl('cl_00', t1, t2, return_cov=True)
    lt, clt = s_th.get_ell_cl('cl_00', t1, t2)
    plt.errorbar(ld, cld, yerr=np.sqrt(np.diag(cov)), fmt='k.')
    plt.plot(lt, clt, 'r-')
    plt.loglog()
plt.show()
print(s_th.get_tracer_combinations())

exit(1)
# Time it
p = p0.copy()
k0 = list(p0.keys())[0]
start = time.time()
for i in range(5):
    p[k0] += 1E-3
    model.loglikes(p)
stop = time.time()
print("%.3lf s per evaluation" % ((stop-start)/5))

# Minimize
updated_info, sampler = run(info)
bf = sampler.products()['minimum'].bestfit()
pf = {k: bf[k] for k in p0.keys()}
print("Final params: ")
print(pf)
