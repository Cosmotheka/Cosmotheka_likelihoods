import numpy as np
import rosatX as rx
import sys

whichbins = sys.argv[1]
fname_out = sys.argv[2]
if whichbins == "all":
    bins = [0, 1, 2, 3]
else:
    bins = [int(whichbins)]

# Parameters to sample over and initial values
params_vary = ["lMc", "alpha_T", "eta_b", "gamma"]
params_names = {"lMc": "\log_{10}M_c",
                "alpha_T": "\alpha_T",
                "eta_b": "\eta_b",
                "gamma": "\Gamma"}
pars_BF = {"lMc": 14.2,
            "alpha_T": 1.0,
            "eta_b": 0.5,
            "gamma": 1.19}
like = rx.ROSATxLike(bins=bins, params_vary=params_vary)

# Log probability function for Cobaya
def logp_cobaya(lMc, alpha_T, eta_b, gamma):
    return like.logp([lMc, alpha_T, eta_b, gamma])

pBF = np.array([pars_BF[k] for k in params_vary])

# Cobaya configuration
info = {"likelihood": {"rosatx": logp_cobaya}}
info["params"] = {k:
                  {"prior": {"min": like.priors[k][0],
                             "max": like.priors[k][1]},
                   "ref": pars_BF[k],
                   "proposal": 0.01,
                   "latex": params_names[k]}
                  for k in params_vary}
info["sampler"] = {"mcmc": {"max_samples": 1000, "Rminus1_stop": 0.01}}
info["output"] = "chains_cobaya/"+fname_out

# Calculate chi-squared
chi2 = -2 * logp_cobaya(*pBF)
print(chi2, like.ndata)
print(info)


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from cobaya import run
from cobaya.log import LoggedError

success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    pass

# Did it work? (e.g. did not get stuck)
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")
