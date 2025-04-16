from cobaya.model import get_model
from cobaya.run import run
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import baccoemu
import pyccl as ccl


# Read in the yaml file
config_fn = '../papers/S8z_lowz/yamls/wisc_noprof.yaml'
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)
model = get_model(info)
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']
#os.system('mkdir -p ' + info['output'])
print(p0)
loglikes, derived = model.loglikes(p0)
print("chi2 = ", -2 * loglikes[0])
