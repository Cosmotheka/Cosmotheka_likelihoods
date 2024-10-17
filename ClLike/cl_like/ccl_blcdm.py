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
    parameters directly (i.e. cannot be used downstream from camb/CLASS).
    """
    # hi_class options
    nonlinear_model: str = 'muSigma'
    classy_arguments: dict = {}

    def initialize(self):
        self._required_results = {}

        # cosmo = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class")
        # Copied from ccl/pk2d.py
        # These lines are needed to compute the Pk2D array
        # self.nk = ccl.ccllib.get_pk_spline_nk(cosmo.cosmo)
        # self.na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
        # self.a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, self.na, 0)
        # self.z_arr = 1/self.a_arr - 1
        # self.lk_arr, _ = ccl.ccllib.get_pk_spline_lk(cosmo.cosmo, self.nk, 0)
        # self.k_arr = np.exp(self.lk_arr)

        # Initialize Class
        self.cosmo_class = Class()
        self.cosmo_class.set({'output': ''})
        self.cosmo_class.set(self.classy_arguments)

        if "gravity_model" in self.classy_arguments:
            self.cosmo_class.set({'output_background_smg': 10})

        nl = self.classy_arguments.get("non linear", '')
        if (nl.lower() == "hmcode") and \
                ("extrapolation_method" not in self.classy_arguments):
            # Use extrap_hmcode in nonlinear.c
            self.cosmo_class.set({"extrapolation_method": 4})

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return ['S8', 'sigma8', "Omega_m", "A_s"]

    def must_provide(self, **requirements):
        pars = self.cosmo_class.pars

        Cl = requirements.get('Cl')
        self.return_Cl = False if Cl is None else True
        if self.return_Cl:
            lmax = max(Cl.values())
            lmax_pars = self.cosmo_class.pars.get("l_max_scalars", 0)
            pars["l_max_scalars"] = max(lmax, lmax_pars)
            cl_keys = ''.join(list(Cl.keys()))
            if 't' in cl_keys and 'tCl' not in pars['output']:
                pars['output'] = ','.join([pars['output'], 'tCl'])
            if (('e' in cl_keys) or ('b' in cl_keys)) and \
                    ('pCl' not in pars['output']):
                pars['output'] = ','.join([pars['output'], 'pCl'])
            if 'p' in cl_keys and 'lCl' not in pars['output']:
                pars['output'] = ','.join([pars['output'], 'lCl'])

            if "lensing" not in pars:
                # This will add lensing if it hasn't been defined by the user
                # Plik will fail if the user sets lensing to false, though.
                pars["lensing"] = True

        self.return_CCL = True if 'CCL' in requirements else False
        if self.return_CCL:
            if 'mPk' not in pars['output']:
                pars['output'] = ','.join([pars['output'], 'mPk'])
            if 'z_max_pk' not in pars:
                pars['z_max_pk'] = 4
            if 'P_k_max_1/Mpc' not in pars:
                pars['P_k_max_1/Mpc'] = 100
            # High k_max to prevent the following error. It might be reduced,
            # though. Needs exploration.  Fails with
            # CosmoSevereError("get_pk_and_k_and_z() is trying to return P(k,z)
            # up to z_max=5.000000e+00 (to encompass your requested maximum
            # value of z); but the input parameters sent to CLASS were such
            # that the non-linear P(k,z) could only be consistently computed up
            # to z=4.710776e+00; increase the input parameter 'P_k_max_h/Mpc'
            # or 'P_k_max_1/Mpc', or increase the precision parameters
            # 'halofit_min_k_max' and/or 'hmcode_min_k_max', or decrease your
            # requested z_max")

        return {}


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

        output = hc.pars['output']
        cosmo = self._get_cosmo_ccl(hc) if self.return_CCL else None
        cl = self._get_Cl(hc) if 'Cl' in output else None
        hc.struct_cleanup()

        state['CCL'] = {'cosmo': cosmo}
        state['Cl'] = cl

        # Derived
        params = {}
        Omega_m = hc.Omega_m()
        params['Omega_m'] =  Omega_m
        if 'mPk' in hc.pars['output']:
            sigma8 = hc.sigma8()
            params['S8'] = sigma8*np.sqrt(Omega_m/0.3)
        if ('A_s' in self.input_params) and ('mPk' in hc.pars['output']):
            params.update({'sigma8': sigma8})
        else:
            params.update({'A_s':
                           hc.get_current_derived_parameters(['A_s'])['A_s']})
        state['derived'] = params

        # Extra methods
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def _get_cosmo_ccl(self, hc, derived=True):
        bhc = hc.get_background()
        # Background
        H = bhc['H [1/Mpc]']
        a_bhc = 1 / (bhc['z'] + 1)
        background = {'a': a_bhc, 'chi': bhc['comov. dist.'],
                      'h_over_h0':  H / H[-1]}
        # Growth & Pks
        growth, pk_linear, pk_nonlin = self._get_growth_and_pks_muSigma(hc, bhc)

        sigma8 = hc.sigma8()
        Omega_m = hc.Omega_m()
        cosmo = ccl.CosmologyCalculator(Omega_c=hc.Omega0_cdm(),
                                        Omega_b=hc.Omega_b(), h=hc.h(),
                                        sigma8=sigma8, n_s=hc.n_s(),
                                        background=background, growth=growth,
                                        pk_linear=pk_linear,
                                        pk_nonlin=pk_nonlin,
                                        nonlinear_model=None)
        return cosmo

    def _get_Cl(self, hc):
        cl = hc.lensed_cl()
        # This is the MontePython way. Note that Cobaya uses T = 2.7255e6,
        # which is the same as long as you don't change T_cmb in classy.
        T = hc.T_cmb()
        for key in cl.keys():
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T * 1e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T * 1e6)

        return cl

    def _get_growth_and_pks_muSigma(self, hc, bhc):
        # Passing bhc to avoid overhead
        pkln_mm, k, z_pk = hc.get_pk_and_k_and_z(nonlinear=False)
        a_pk = 1 / (1+z_pk)

        mu = interp1d(bhc['z'], bhc['mgclass_dmu'] + 1)(z_pk)
        Sigma = interp1d(bhc['z'], bhc['mgclass_dSigma'] + 1)(z_pk)

        pkln_mw = Sigma * pkln_mm
        pkln_ww = Sigma**2 * pkln_mm

        pk_linear = {'a': a_pk, 'k': k,
                     'delta_matter:delta_matter': pkln_mm.T,
                     'delta_matter:Weyl': pkln_mw.T,
                     'Weyl:Weyl': pkln_ww.T
                     }

        # Growth measured from the transfers as recommended by Emilio
        # The growth factor and rate columns in the background tables are only
        # valid for dust-only cosmologies
        # They solve: delta'' + 2H delta' + delta = 0
        growth_factor = np.zeros(z_pk.size)
        growth_rate = np.zeros(z_pk.size)
        for i, zi in enumerate(z_pk):
            growth_factor[i] = hc.scale_dependent_growth_factor_at_k_and_z(0.01, zi)
            growth_rate[i] = hc.scale_dependent_growth_factor_f_at_k_and_z(0.01, zi)

        growth = {'a': a_pk, 'growth_factor': growth_factor,
                  "growth_rate": growth_rate}

        # Non linear Pk
        if self.nonlinear_model == "Linear":
            pk_nonlin = pk_linear
        elif self.nonlinear_model == "muSigma":
            try:
                pk_mm, k, z = hc.get_pk_and_k_and_z(nonlinear=True)
            except CosmoSevereError as e:
                # If get_pk_and_k_and_z fails because the nonlinear pk has not
                # been requested up to a high k, check the reason. If k_max >=
                # 100, we can safely return the linear Pk because those models
                # will have a) a very low As or b) very high redshift. In both
                # cases we do not care about what happens there.
                pkmax = hc.pars.get('P_k_max_1/Mpc', 0)
                hmkmax = hc.pars.get('hmcode_min_k_max', pkmax)
                hfkmax = hc.pars.get('halofit_min_k_max', pkmax)
                nonlin = hc.pars['non linear']
                if "hmcode_min_k_max" not in str(e):
                    raise e
                elif ((nonlin == 'hmcode') and (hmkmax < 100)) or \
                    ((nonlin == 'halofit') and (hfkmax < 100)):
                    print("You need at least k_max=100 Mpc^-1 to default to "
                          "the linear pk in case of failure")
                    raise e
                self.log.debug(str(e))
                pk_mm = np.zeros_like(pkln_mm[:-1])
                k = k[:-1]  # To avoid going out of boundaries
                for i, zi in enumerate(z_pk):
                    pk_mm[:, i] = [hc.pk(ki, zi) for ki in k]

            pk_mw = Sigma * pk_mm
            pk_ww = Sigma**2 * pk_mm

            pk_nonlin = {'a': a_pk, 'k': k,
                         'delta_matter:delta_matter': pk_mm.T,
                         'delta_matter:Weyl': pk_mw.T,
                         'Weyl:Weyl': pk_ww.T}
        else:
            raise NotImplementedError("nonlinear_model = "
                                      f"{self.nonlinear_model} not Implemented")

        return growth, pk_linear, pk_nonlin

    # Commented out. Old way but could be useful for models that are not
    # mu/Sigma
    #

    # def _get_primordial_pk(self):
    #     cosmo = self.cosmo_class

    #     Pk = cosmo.get_primordial()
    #     lpPk = np.log(Pk['P_scalar(k)'])
    #     lkPk = np.log(Pk['k [1/Mpc]'])

    #     pPk = np.exp(interp1d(lkPk, lpPk)(self.lk_arr))

    #     return (2 * np.pi ** 2) * pPk / self.k_arr ** 3

    # def _get_pklin_pair(self, pair, z):
    #     cosmo = self.cosmo_class
    #     pPk = self._get_primordial_pk()

    #     H0 = self.cosmo_class.Hubble(0)
    #     Omega_m = self.cosmo_class.Omega_m()

    #     Tks = cosmo.get_transfer(z)
    #     kTk = Tks['k (h/Mpc)'] * cosmo.h()
    #     lkTk = np.log(kTk)

    #     Tk = []
    #     for ti in pair:
    #         if (ti == 'delta_matter'):
    #             iTk = Tks['d_m']
    #         elif (ti.lower() == 'weyl'):
    #             iTk = (Tks['phi'] + Tks['psi']) / 2.
    #             # Correct by the GR W->d factor. CCL devides by it
    #             iTk *= (- kTk**2 * 2 / 3 / (H0)**2 / Omega_m / (1 + z))
    #         else:
    #             raise ValueError(f'Tracer {ti} not implemented')

    #         Tk.append(interp1d(lkTk, iTk, kind='cubic')(self.lk_arr))

    #     return Tk[0] * Tk[1] * pPk

    def get_CCL(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['CCL']

    def get_Cl(self, units=None):
        """
        Get dictionary of Cls.

        :return: dict of results
        """
        return self._current_state['Cl']
