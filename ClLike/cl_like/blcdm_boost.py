import numpy as np
import pyccl as ccl
import pickle
import tensorflow as tf
import os


class BLCDMCalculator(object):
    """ This class implements the BLCDM boost on the matter power spectra
    respect to LCDM; i.e. Qk = Pk_BLCDM / Pk_LCDM; and the Weyl boost
    Wk^2 = Sigma(a)^2 * Qk
    """
    def __init__(self, emu_folder, parametrizaton='1_minus_mu0OmegaDE'):
        with open(os.path.join(emu_folder, 'details.pickle'), 'rb') as f:
            details = pickle.load(f)
        self.bounds = details['bounds']
        self.k = details['k']
        self.a_arr = np.linspace(self.bounds[-1][0], self.bounds[-1][1], 30)
        self.scaler = details['scaler']
        self.parnames = ['Omega_m', 'lnE9As', 'h', 'mu0', 'a']
        self.model = tf.keras.models.load_model(os.path.join(emu_folder, 'model.h5'), compile=False)

        self.pk2d_computed = None

        self.parametrizaton = parametrizaton

        # Set params to None
        self.mu0 = None
        self.Sigma0 = None
        self.cosmo = None

    def _transform_space(self, x):
        return (x - self.bounds[:, 0])/(self.bounds[:, 1] - self.bounds[:, 0])

    def _check_pars(self, x):
        assert len(x) == len(self.bounds), 'Make sure all necessary parameters are passed'
        for i, xx in enumerate(x):
            if (xx < self.bounds[i, 0]) | (xx > self.bounds[i, 1]):
                raise ValueError(f'parameter {self.parnames[i]} has value {xx}, out of bounds {self.bounds[i]}')

    def get_mg_boost(self, Omega_m=None, lnE9As=None, h=None, mu0=None, a=None):
        pars = np.array([Omega_m, lnE9As, h, mu0, a])
        self._check_pars(pars)
        return self.k, np.squeeze(np.exp(self.scaler.inverse_transform(self.model(np.array([self._transform_space(pars)])))))

    def get_boost_factor(self, pars, k, a):
        return self.get_mg_boost(Omega_m=pars['Omega_m'],
                                 lnE9As=np.log(pars['As']*1e-9),
                                 h=pars['h'], mu0=pars['mu0'], a=a)

    def get_mg_boost_parallel(self, Omega_m=None, lnE9As=None, h=None, mu0=None, a_arr=None):
        pars = np.array([Omega_m, lnE9As, h, mu0, a_arr.min()])
        self._check_pars(pars)

        # Create an array of parameter sets for all values in a_arr
        pars_batch = np.array([
            [Omega_m, lnE9As, h, mu0, a] for a in a_arr
        ])

        # Transform the parameter space for all inputs at once
        transformed_batch = np.array([self._transform_space(pars) for pars in pars_batch])

        # Pass the batch through the model in a single call
        model_output = self.model(transformed_batch)

        # Inverse transform and process the output
        results = np.squeeze(np.exp(self.scaler.inverse_transform(model_output)))

        return self.k, results

    def apply_boost(self, kind, pk2d):
        """
        Apply the boost to the matter power spectrum. It will reduce the output
        power spectrum to the range of k's and a's in the emulator:
        k < 3 h/Mpc and a > 0.25

        Args:
            pk2d: Pk2D object
        """
        Qk = self.get_boost(kind)
        return Qk * pk2d

    def update_pk(self, cosmo, mu0, Sigma0, **kwargs):
        """ Update the internal pk arrays

        Args:
            cosmo:
            mu0:
        """
        # Set params
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.cosmo = cosmo

        # Matter power spectrum boost
        # Emulator units: k [h/Mpc]; Qk [Mpc^3/h^3]
        kh, Qkh3_arr = self.get_mg_boost_parallel(Omega_m=cosmo.Omega_m,
                                           lnE9As=np.log(cosmo.As*1e-9),
                                           h=cosmo.h, mu0=mu0,
                                           a_arr=self.a_arr)
        k = kh * cosmo.h
        Qk_arr = Qkh3_arr / cosmo.h**3

        lk_arr = np.log(k)
        Qk = ccl.Pk2D(a_arr=self.a_arr, lk_arr=lk_arr, pk_arr=np.log(Qk_arr),
                      is_logp=True, extrap_order_lok=0, extrap_order_hik=2)

        # Weyl boost for mw
        Sigma_a = self.get_Sigma_a()
        Wk_arr = Sigma_a[None, :] * Qk_arr
        Wk = ccl.Pk2D(a_arr=self.a_arr, lk_arr=lk_arr, pk_arr=np.log(Wk_arr),
                      is_logp=True, extrap_order_lok=0, extrap_order_hik=2)

        # Weyl boost for ww
        Wk2_arr = Sigma_a[None, :]**2 * Qk_arr
        Wk2 = ccl.Pk2D(a_arr=self.a_arr, lk_arr=lk_arr, pk_arr=np.log(Wk2_arr),
                      is_logp=True, extrap_order_lok=0, extrap_order_hik=2)

        self.pk2d_computed = {'mm': Qk,
                              'wm': Wk,
                              'mw': Wk,
                              'ww': Wk2}

    def get_boost(self, kind):
        return self.pk2d_computed[kind]

    def get_mu_a(self, a):
        if self.parametrizaton == '1_minus_mu0OmegaDE':
            mu_a = 1 - self.cosmo.omega_x(a, 'dark_energy') / \
                self.cosmo.omega_x(1, 'dark_energy')
        else:
            raise NotImplementedError('Only 1_minus_mu0OmegaDE implemented')

        return mu_a

    def get_Sigma_a(self, cosmo=None, Sigma0=None, a_arr=None):
        if cosmo is None:
            cosmo = self.cosmo

        if Sigma0 is None:
            Sigma0 = self.Sigma0

        if a_arr is None:
            a_arr = self.a_arr

        if self.parametrizaton == '1_minus_mu0OmegaDE':
            Sigma_a = 1 - cosmo.omega_x(a_arr, 'dark_energy') / \
                cosmo.omega_x(1, 'dark_energy')
        else:
            raise NotImplementedError('Only 1_minus_mu0OmegaDE implemented')

        return Sigma_a

