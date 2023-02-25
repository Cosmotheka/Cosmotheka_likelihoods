"""
This is based on an earlier implementation in MontePython:
https://github.com/carlosggarcia/montepython_public/blob/beyondLCDM/montepython/ccl_class_blcdm.py

It could be refactored to use the official classy theory class.
"""
import numpy as np
import pyccl as ccl
import numpy as np
from cobaya.theory import Theory
from classy import Class, CosmoSevereError, CosmoComputationError
from scipy.interpolate import interp1d


class CCL_BLCDM(Theory):
    """
    This implements CCL as a `Theory` object that takes in cosmological
    parameters directly (i.e. cannot be used downstream from camb/CLASS.
    """
    # CCL options
    baryons_pk: str = 'nobaryons'
    nonlinear_model: str = 'muSigma'
    classy_arguments: dict = {}

    def initialize(self):
        self._required_results = {}

        cosmo = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class")
        # Copied from ccl/pk2d.py
        # These lines are needed to compute the Pk2D array
        self.nk = ccl.ccllib.get_pk_spline_nk(cosmo.cosmo)
        self.na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
        self.a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, self.na, 0)
        self.z_arr = 1/self.a_arr - 1
        lk_arr, _ = ccl.ccllib.get_pk_spline_lk(cosmo.cosmo, self.nk, 0)
        self.k_arr = np.exp(lk_arr)

        # Reduce the computations needed (in particular, going up to very high
        # k makes unfeasable to compute the Pk with hi_class
        self.a_arr = self.a_arr[self.z_arr < 8]
        self.z_arr = self.z_arr[self.z_arr < 8]
        self.lk_arr = lk_arr[self.k_arr < 10]
        self.k_arr = self.k_arr[self.k_arr < 10]

        # Initialize Class
        self.cosmo_class = Class()
        self.cosmo_class.set({'output': 'mPk,mTk',
                              'z_max_pk': np.max(self.z_arr),
                              'P_k_max_1/Mpc': np.max(self.k_arr),
                              'k_per_decade_for_pk': 200,
                              'k_per_decade_for_bao': 200})

        self.cosmo_class.set(self.classy_arguments)

        if "gravity_model" in self.classy_arguments:
            self.cosmo_class.set({'output_background_smg': 10})

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return ['S8', 'sigma8', "Omega_m", "A_s"]

    def get_allow_agnostic(self):
        return True

    def _get_params_for_classy(self, params):
        """
        Return a parameters dictionary for classy.

        It will understand the parameter names with "__" as a member of a list,
        like in MontePython; e.g. parameters_smg__1: 0, parameters_smg__2:0,
        would be transformed to "parameters_smg": "0, 0".
        """
        params_new = {}
        keys = list(params.keys())
        for k in keys:
            if "__" not in k:
                continue
            base_name = k.split("__")[0]
            val = str(params.pop(k))
            if base_name in params_new:
                params_new[base_name] += f", {val}"
            else:
                params_new[base_name] = val

        params.update(params_new)

        return params

    def calculate(self, state, want_derived=True, **params_values_dict):
        hc = self.cosmo_class
        hc.set(self._get_params_for_classy(params_values_dict))
        try:
            hc.compute()
        # Based on the official classy theory class:
        # https://github.com/CobayaSampler/cobaya/blob/master/cobaya/theories/classy/classy.py
        except CosmoComputationError as e:
            self.log.debug("Computation of cosmological products failed. "
                           "Assigning 0 likelihood and going on. "
                           "The output of the CLASS error was %s" % e)
            return False
        except CosmoSevereError:
            self.log.error("Serious error setting parameters or computing results. "
                           "The parameters passed were %r. To see the original "
                           "CLASS' error traceback, make 'debug: True'.",
                           state["params"])
            raise

        bhc = hc.get_background()
        # Background
        H = bhc['H [1/Mpc]']
        background = {'a': 1 / (bhc['z'] + 1), 'chi': bhc['comov. dist.'],
                      'h_over_h0':  H / H[-1]}
        # Growth
        D_arr = np.array([hc.scale_independent_growth_factor(z) for z in self.z_arr])
        f_arr = np.array([hc.scale_independent_growth_factor_f(z) for z in self.z_arr])
        growth = {'a': self.a_arr, 'growth_factor': D_arr,
                  'growth_rate': f_arr}
        # Pk
        # pkln = np.array([[hc.pk_lin(k, z) for k in self.k_arr] for z in self.z_arr])
        pkln_mm = np.array([self._get_pklin_pair(('delta_matter', 'delta_matter'),
                                              z) for z in self.z_arr])
        pkln_mw = np.array([self._get_pklin_pair(('delta_matter', 'Weyl'), z) for
                            z in self.z_arr])
        pkln_ww = np.array([self._get_pklin_pair(('Weyl', 'Weyl'), z) for z in
                            self.z_arr])

        pk_linear = {'a': self.a_arr, 'k': self.k_arr,
                     # 'delta_matter:delta_matter': pkln,
                     'delta_matter:delta_matter': pkln_mm,
                     'delta_matter:Weyl': pkln_mw,
                     'Weyl:Weyl': pkln_ww
                     }

        if self.nonlinear_model == 'Linear':
            pk_nonlin = pk_linear
        elif self.nonlinear_model == 'muSigma':
            pk_mm = np.array([self._get_pknonlin_pair_muSigma(('delta_matter', 'delta_matter'),
                                                z) for z in self.z_arr])
            pk_mw = np.array([self._get_pknonlin_pair_muSigma(('delta_matter', 'Weyl'), z) for
                              z in self.z_arr])
            pk_ww = np.array([self._get_pknonlin_pair_muSigma(('Weyl', 'Weyl'), z) for z in
                                self.z_arr])

            pk_nonlin = {'a': self.a_arr, 'k': self.k_arr,
                         'delta_matter:delta_matter': pk_mm,
                         'delta_matter:Weyl': pk_mw,
                         'Weyl:Weyl': pk_ww}

            # from matplotlib import pyplot as plt
            # plt.loglog(self.k_arr, pk_ww[-1])
            # plt.loglog(self.k_arr, pkln_ww[-1])
            # plt.title(f'z = {self.z_arr[-1]}')
            # plt.show()
            # plt.close()

            # plt.semilogx(self.k_arr, pkln_ww[-1] / pk_ww[-1] - 1)
            # plt.title(f'z = {self.z_arr[-1]}')
            # plt.show()
            # plt.close()
        else:
            raise NotImplementedError("nonlinear_model = "
                                      f"{self.nonlinear_model} not Implemented")

        sigma8 = hc.sigma8()
        Omega_m = hc.Omega_m()
        cosmo = ccl.CosmologyCalculator(Omega_c=hc.Omega0_cdm(),
                                        Omega_b=hc.Omega_b(), h=hc.h(),
                                        sigma8=sigma8, n_s=hc.n_s(),
                                        background=background, growth=growth,
                                        pk_linear=pk_linear,
                                        pk_nonlin=pk_nonlin,
                                        nonlinear_model=None)



        self.cosmo_class.struct_cleanup()

        state['CCL'] = {'cosmo': cosmo}

        # Derived

        params = {}
        if 'A_s' in self.input_params:
            params.update({'sigma8': sigma8})
        else:
            params.update({'A_s':
                           hc.get_current_derived_parameters(['A_s'])['A_s']})
        params.update({'S8': sigma8*np.sqrt(Omega_m/0.3), 'Omega_m': Omega_m})

        state['derived'] = params
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def _get_primordial_pk(self):
        cosmo = self.cosmo_class

        Pk = cosmo.get_primordial()
        lpPk = np.log(Pk['P_scalar(k)'])
        lkPk = np.log(Pk['k [1/Mpc]'])

        pPk = np.exp(interp1d(lkPk, lpPk)(self.lk_arr))

        return (2 * np.pi ** 2) * pPk / self.k_arr ** 3

    def _get_pklin_pair(self, pair, z):
        cosmo = self.cosmo_class
        pPk = self._get_primordial_pk()

        H0 = self.cosmo_class.Hubble(0)
        Omega_m = self.cosmo_class.Omega_m()

        Tks = cosmo.get_transfer(z)
        kTk = Tks['k (h/Mpc)'] * cosmo.h()
        lkTk = np.log(kTk)

        Tk = []
        for ti in pair:
            if (ti == 'delta_matter'):
                iTk = Tks['d_m']
            elif (ti.lower() == 'weyl'):
                iTk = (Tks['phi'] + Tks['psi']) / 2.
                # Correct by the GR W->d factor. CCL devides by it
                iTk *= (- kTk**2 * 2 / 3 / (H0)**2 / Omega_m / (1 + z))
            else:
                raise ValueError(f'Tracer {ti} not implemented')

            Tk.append(interp1d(lkTk, iTk, kind='cubic')(self.lk_arr))

        return Tk[0] * Tk[1] * pPk

    def _get_pknonlin_pair_muSigma(self, pair, z):
        cosmo = self.cosmo_class
        mPk = np.array([cosmo.pk(k, z) for k in self.k_arr])

        H0 = self.cosmo_class.Hubble(0)
        Omega_m = self.cosmo_class.Omega_m()

        back = cosmo.get_background()
        mu = interp1d(back['z'], back['mgclass_dmu'] + 1)(z)
        Sigma = interp1d(back['z'], back['mgclass_dSigma'] + 1)(z)

        pk = mPk.copy()
        for ti in pair:
            if (ti == 'delta_matter'):
                factor = 1
            elif (ti.lower() == 'weyl'):
                # This is the factor that goes in the lensing equations and
                # relates Weyl and the matter density fluctuation
                #     factor = 3 * (H0)**2 * Omega_m * (1 + z) * Sigma / (-self.k_arr**2 * 2)
                #
                # CCL has the GR factor already in, so the only missing
                # term is Sigma
                factor = Sigma
            else:
                raise ValueError(f'Tracer {ti} not implemented')
            pk *= factor

        return pk

    def get_CCL(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['CCL']
