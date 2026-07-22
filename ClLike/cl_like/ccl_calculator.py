"""
This is a module that will use Cobaya's classy implemetation and pass the
results to the CCL CosmologyCalculator. This way we can do CMB primary and
Cells
"""
import numpy as np
import pyccl as ccl
import numpy as np
from cobaya.theory import Theory
from cobaya.log import get_logger

logger = get_logger(__name__)


class CCL_CosmologyCalculator(Theory):
    """
    This implements CCL as a `Theory` object that takes in cosmological
    parameters directly (i.e. cannot be used downstream from camb/CLASS).
    """
    ccl_arguments: dict = {}
    # TODO: Determine z_max. Consider computing linear Pk in CLASS and non-linear in CCL
    # Problem, non-lin is needed for CMB lensing.
    z_max: float = 4
    log10k_min: float = -4
    log10k_max: float = np.log10(50)
    pk_nk_per_decade: int = 100
    pk_na: int = 128

    def initialize(self):
        self._required_results = {}

        # When ccl_arguments is not provided, Cobaya saves it in the
        # updated.yaml as null. When resuming the chains, we need to change the
        # type.
        if self.ccl_arguments is None:
            self.ccl_arguments = {}

        # Precompute the k and z arrays for the P(k) interpolators. This is needed

        pk_nk = int((self.log10k_max - self.log10k_min) * self.pk_nk_per_decade)
        self.pk_k_arr = np.logspace(self.log10k_min, self.log10k_max, pk_nk)
        self.pk_z_arr = np.linspace(0, self.z_max, self.pk_na)[::-1]
        self.pk_a_arr = 1 / (self.pk_z_arr + 1)

        # Log precomputed arrays for debugging
        logger.debug(f"Precomputed k array for P(k) interpolators: {self.pk_k_arr}")
        logger.debug(f"Precomputed z array for P(k) interpolators: {self.pk_z_arr}")
        logger.debug(f"Precomputed a array for P(k) interpolators: {self.pk_a_arr}")
        logger.info(f"P(k) grid: nk_per_decade={self.pk_nk_per_decade} (nk_total={pk_nk}), na={self.pk_na}, z_max={self.z_max}")

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return []

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def get_requirements(self):
        return {
            # "Omega_cdm": {'z': [0.0]},
            # "Omega_b": {'z': [0.0]},
            # "Omega_nu_massive": {'z': [0.0]},
            "CLASS_background": None,
            "Hubble": {"z": [0.0]},
            "sigma8_z": {"z": [0.0]},
            "Pk_interpolator": {
                "vars_pairs": (("delta_tot", "delta_tot")),
                "z": [0.0, self.z_max],
                "k_max": 10**self.log10k_max,
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
        pk_lin_interp = self.provider.get_Pk_interpolator(var_pair=("delta_tot", "delta_tot"), nonlinear=False).P
        pk_nonlin_interp = self.provider.get_Pk_interpolator(var_pair=("delta_tot", "delta_tot"), nonlinear=True).P

        # Evaluate interpolators on the k-z grid
        pkln_mm = np.zeros((len(self.pk_z_arr), len(self.pk_k_arr)))
        pk_mm = np.zeros((len(self.pk_z_arr), len(self.pk_k_arr)))
        for i, zi in enumerate(self.pk_z_arr):
            pkln_mm[i, :] = pk_lin_interp(zi, self.pk_k_arr)
            pk_mm[i, :] = pk_nonlin_interp(zi, self.pk_k_arr)

        pkln_mw = pkln_ww = pkln_mm
        pk_linear = {
            "a": self.pk_a_arr,
            "k": self.pk_k_arr,
            "delta_matter:delta_matter": pkln_mm,
            "delta_matter:Weyl": pkln_mw,
            "Weyl:Weyl": pkln_ww
        }

        pk_mw = pk_ww = pk_mm
        pk_nonlin = {
            "a": self.pk_a_arr,
            "k": self.pk_k_arr,
            "delta_matter:delta_matter": pk_mm,
            "delta_matter:Weyl": pk_mw,
            "Weyl:Weyl": pk_ww
        }

        h = provider.get_Hubble(z=0, units="km/s/Mpc")[0] / 100
        rho_crit = b["(.)rho_crit"][-1]
        Omega_cdm = b['(.)rho_cdm'][-1] / rho_crit
        Omega_b = b['(.)rho_b'][-1] / rho_crit
        m_nu = []
        for i in range(3):
            key = f'(.)rho_ncdm[{i}]'
            if key in b.keys():
                Omega_nu = b[key][-1] / rho_crit
                m_nu.append(Omega_nu * 93.14 * h**2)
            else:
                break
        n_s = provider.get_param("n_s")  # This will only be relevant if Pk's are not passed
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