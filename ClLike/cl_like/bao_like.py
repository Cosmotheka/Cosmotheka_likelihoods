import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import copy
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.optimize import minimize


class BAOLike(Likelihood):
    # Bins to use
    bins: list = [0, 1, 2, 3]

    # All data in:
    #  https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-only/
    def initialize(self):
        bincov = np.array([[2*b, 2*b+1] for b in self.bins]).flatten()
        self.r_d_fid = 147.78
        self.a_s = 1/(1+self.zs_all[self.bins])
        dMs = self.dMs_all[self.bins]
        dHs = self.dHs_all[self.bins]
        self.cov = self.cov_all[bincov][:, bincov]
        self.mean = np.concatenate((dMs, dHs)).T.flatten()
        self.icov = np.linalg.inv(self.cov)

    def get_requirements(self):
        return {"CCL": None}

    @property
    def dMs_all(self):
        return np.array([10.23406, 13.36595, 17.85824, 30.68760])

    @property
    def dHs_all(self):
        return np.array([24.98058, 22.31656, 19.32575, 13.26090])

    @property
    def zs_all(self):
        return np.array([0.38, 0.51, 0.698, 1.48])

    @property
    def cov_all(self):
        return np.array([[ 2.860520e-02, -4.939281e-02,  1.489688e-02, -1.387079e-02,
                          0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000],
                         [-4.939281e-02,  5.307187e-01, -2.423513e-02,  1.767087e-01,
                          0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000],
                         [ 1.489688e-02, -2.423513e-02,  4.147534e-02, -4.873962e-02,
                          0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000],
                         [-1.387079e-02,  1.767087e-01, -4.873962e-02,  3.268589e-01,
                          0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000],
                         [ 0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,
                          0.1076634008, -0.0583182034,  0.0000000000, 0.0000000000],
                         [ 0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,
                          -0.0583182034, 0.28381763863,  0.0000000000, 0.0000000000],
                         [ 0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,
                          0.0000000000,  0.0000000000,  0.6373160400, 0.1706891000],
                         [ 0.0000000000,  0.0000000000,  0.0000000000,  0.0000000000,
                          0.0000000000,  0.0000000000,  0.1706891000, 0.3046841500]])

    def get_rd(self, cosmo):
        om = cosmo['Omega_m']*cosmo['h']**2
        ob = cosmo['Omega_b']*cosmo['h']**2
        rd = 45.5337*np.log(7.20376/om)/np.sqrt(1+9.98592*ob**0.801347)
        return rd

    def get_theory(self, cosmo):
        r_d = self.get_rd(cosmo)
        H0 = cosmo['h']/ccl.physical_constants.CLIGHT_HMPC
        dM = ccl.comoving_radial_distance(cosmo, self.a_s)/r_d
        dH = 1/(H0*ccl.h_over_h0(cosmo, self.a_s)*r_d)
        return np.concatenate((dM, dH)).T.flatten()

    def logp(self, **params_values):
        cosmo = self.provider.get_CCL()["cosmo"]
        theory = self.get_theory(cosmo)
        res = theory-self.mean
        return -0.5 * np.dot(res, np.dot(self.icov, res))
