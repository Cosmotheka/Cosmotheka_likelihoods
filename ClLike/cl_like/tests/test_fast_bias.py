from cobaya.model import get_model
from cobaya.run import run
import yaml
import os

import numpy as np
import numpy.linalg as LA 

# Read in the yaml file
config_fn = 'tests/config_fast_bias.yaml'
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Get the mean proposed in the yaml file for each parameter
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
            if 'ref' in info['params'][p]:
                # bug: there are params with 'ref' but no 'loc'
                p0[p] = info['params'][p]['ref']['loc']
            # debugged: all other sampled params have 'loc' in 'prior'
            elif 'prior' in info['params'][p]:
                p0[p] = info['params'][p]['prior']['loc']
os.system('mkdir -p ' + info['output'])

print("params = ", p0)

# Compute the likelihood at that point
model = get_model(info)
#print('sampled params:', list(model.parameterization.sampled_params()))
#print('model.requested()', model.requested())

loglikes, derived = model.loglikes(p0)
print("chi2 = ", -2 * loglikes[0])
