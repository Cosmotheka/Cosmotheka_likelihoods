"""
This is a module that will use Cobaya's classy implemetation and pass the 
results to the CCL CosmologyCalculator. This way we can do CMB primary and
Cells
"""
import numpy as np
import pyccl as ccl
import numpy as np
from cobaya.theory import Theory


class CCL_CosmologyCalculator(Theory):
    """
    This implements CCL as a `Theory` object that takes in cosmological
    parameters directly (i.e. cannot be used downstream from camb/CLASS).
    """
    ccl_arguments: dict = {}
    # TODO: Determine z_max. Consider computing linear Pk in CLASS and non-linear in CCL
    # Problem, non-lin is needed for CMB lensing.
    z_max: float = 4

    def initialize(self):
        self._required_results = {}
        self.baccompk = None

        # When ccl_arguments is not provided, Cobaya saves it in the
        # updated.yaml as null. When resuming the chains, we need to change the
        # type.
        if self.ccl_arguments is None:
            self.ccl_arguments = {}

        # cosmo = ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class")
        # Copied from ccl/pk2d.py
        # These lines are needed to compute the Pk2D array
        # self.nk = ccl.ccllib.get_pk_spline_nk(cosmo.cosmo)
        # self.na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
        # self.a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, self.na, 0)
        # self.z_arr = 1/self.a_arr - 1
        # self.lk_arr, _ = ccl.ccllib.get_pk_spline_lk(cosmo.cosmo, self.nk, 0)
        # self.k_arr = np.exp(self.lk_arr)

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return []

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def get_requirements(self):
        return {
            "Omega_cdm": {'z': [0.0]},
            "Omega_b": {'z': [0.0]},
            "Omega_nu_massive": {'z': [0.0]},
            "CLASS_background": None,
            "Hubble": {"z": [0.0]},
            "sigma8_z": {"z": [0.0]},
            "Pk_grid": {
                "vars_pairs": (("delta_tot", "delta_tot")),
                "z": [0.0, self.z_max],
                "k_max": 50.0,
                "nonlinear": [False, True],
            },
        }

    def must_provide(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        # CCL currently has no way to infer the required inputs from
        # the required outputs
        # So a lot of this is fixed
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        return {}

    def calculate(self, state, want_derived=True, **params_values_dict):
        cosmo = self._get_CosmologyCalculator(params_values_dict)

        state['CCL'] = {'cosmo': cosmo}

        # Extra methods
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def _get_CosmologyCalculator(self, params):
        provider = self.provider

        # Background
        b = provider.get_CLASS_background()
        # NOTE: There is no need to select z<z_max since z_max is for the Pk
        # grid.
        z_arr = b['z']
        z_arr = z_arr
        a_arr = 1 / (z_arr + 1)
        H = b['H [1/Mpc]']
        chi = b['comov. dist.']
        background = {'a': a_arr, 'chi': chi,
                      'h_over_h0':  H / H[-1]}

        # Growth
        growth_factor = b['gr.fac. D']
        growth_rate = b['gr.fac. f']
        growth = {'a': a_arr, 'growth_factor': growth_factor,
                  "growth_rate": growth_rate}

        # TODO: use Weyl instead of assume matter
        k_arr, z_arr, pkln_mm = self.provider.get_Pk_grid(var_pair=("delta_tot", "delta_tot"), nonlinear=False)
        pkln_mw = pkln_ww = pkln_mm
        a_arr = 1/(z_arr + 1)
        pk_linear = {
            "a": a_arr,
            "k": k_arr,
            "delta_matter:delta_matter": pkln_mm,
            "delta_matter:Weyl": pkln_mw,
            "Weyl:Weyl": pkln_ww
        }

        k_arr, z_arr, pk_mm = self.provider.get_Pk_grid(var_pair=("delta_tot", "delta_tot"), nonlinear=True)
        pk_mw = pk_ww = pk_mm
        a_arr = 1/(z_arr + 1)
        pk_nonlin = {
            "a": a_arr,
            "k": k_arr,
            "delta_matter:delta_matter": pk_mm,
            "delta_matter:Weyl": pk_mw,
            "Weyl:Weyl": pk_ww
        }

        Omega_cdm = provider.get_param("Omega_cdm")
        Omega_b = provider.get_param("Omega_b")
        h = provider.get_param("h")
        m_nu = np.sum(provider.get_Omega_nu_massive(z=0) * 93.14 * h**2)
        n_s = provider.get_param("n_s")
        sigma8 = provider.get_sigma8_z(z=0)[0]
        cosmo = ccl.CosmologyCalculator(Omega_c=Omega_cdm,
                                        Omega_b=Omega_b,
                                        h=h,
                                        sigma8=sigma8,
                                        n_s=n_s,
                                        m_nu=m_nu,
                                        background=background, growth=growth,
                                        pk_linear=pk_linear,
                                        pk_nonlin=pk_nonlin,
                                        nonlinear_model=None)
        return cosmo

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
