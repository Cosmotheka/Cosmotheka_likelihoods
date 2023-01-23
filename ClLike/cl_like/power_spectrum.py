"""
Theory class that computes the power spectrum. It uses the CCL theory
class.
"""
from cobaya.theory import Theory
import pyccl as ccl
import numpy as np

# Try to import LPT and EPT. If it fails due to some missing library. Raise an
# error when checking the bias_model requested
try:
    from .lpt import LPTCalculator, get_lpt_pk2d
    HAVE_LPT = True
    LPT_exception = None
except ImportError as e:
    LPT_exception = e
    HAVE_LPT = False

try:
    from .ept import EPTCalculator, get_ept_pk2d
    HAVE_EPT = True
    EPT_exception = None
except ImportError as e:
    EPT_exception = e
    HAVE_EPT = False


class Pk(Theory):
    """Computes the power spectrum"""
    # b(z) model name
    bias_model: str = "BzNone"
    # k shot noise suppression scale
    k_SN_suppress: float = 0.01
    # min k 3D power spectra
    l10k_min_pks: float = -4.0
    # max k 3D power spectra
    l10k_max_pks: float = 2.0
    # zmax for 3D power spectra
    zmax_pks: float = 4.
    # #z for 3D power spectra
    nz_pks: int = 30
    # #k for 3D power spectra
    nk_per_dex_pks: int = 25

    def initialize(self):
        # Bias model
        self.is_PT_bias = self.bias_model in ['LagrangianPT', 'EulerianPT']
        # Pk sampling
        self.a_s_pks = 1./(1+np.linspace(0., self.zmax_pks, self.nz_pks)[::-1])
        self.nk_pks = int((self.l10k_max_pks - self.l10k_min_pks) *
                          self.nk_per_dex_pks)
        # Initialize parameterless Halo model stuff
        # if self.bias_model == 'HaloModel':
        #     # Mass definition
        #     if self.mass_def_str == 'fof':
        #         self.massdef = ccl.halos.MassDef('fof', 'critical')
        #     else:
        #         rt = self.mass_def_str[-1]
        #         if rt == 'c':
        #             rhotyp = 'critical'
        #         elif rt == 'm':
        #             rhotyp = 'matter'
        #         else:
        #             raise ValueError(f"Unknown density type {rt}")
        #         if self.mass_def_str[:-1] == 'Vir':
        #             Delta = 'vir'
        #         else:
        #             Delta = float(self.mass_def_str[:-1])
        #         self.massdef = ccl.halos.MassDef(Delta, rhotyp)
        #     # Mass function
        #     self.mfc = ccl.halos.mass_function_from_name(self.mf_name)
        #     # Halo bias
        #     self.hbc = ccl.halos.halo_bias_from_name(self.hb_name)
        #     # Concentration
        #     cmc = ccl.halos.concentration_from_name(self.cm_name)
        #     self.cm = cmc(mdef=self.massdef)
        #     # Default profiles for different quantities
        #     self.profs = {'galaxy_density': None,
        #                   'galaxy_shear': ccl.halos.HaloProfileNFW(self.cm),
        #                   'cmb_convergence': ccl.halos.HaloProfileNFW(self.cm),
        #                   'cmb_tSZ': ccl.halos.HaloProfilePressureGNFW()}
        #     # Profile 2-point function for HOD
        #     self.p2pt_HOD = ccl.halos.Profile2ptHOD()
        #     # Halo model correction for the transition regime
        #     if self.HM_correction == 'halofit':
        #         from .hm_extra import HalomodCorrection
        #         self.hmcorr = HalomodCorrection()
        #     else:
        #         self.hmcorr = None
        if self.bias_model == 'LagrangianPT' and not HAVE_LPT:
            raise LPT_exception
        elif self.bias_model == 'EulerianPT' and not HAVE_EPT:
            raise EPT_exception

    def must_provide(self, **requirements):
        if "Pk" not in requirements:
            return {}

        return {"CCL": None}

    def calculate(self, state, want_derived=True, **params_values_dict):
        cosmo = self.provider.get_CCL()["cosmo"]
        state['Pk'] = {'pk_data': self._get_pk_data(cosmo)}

    def get_Pk(self):
        return self._current_state['Pk']

    def _get_pk_data(self, cosmo):
        cosmo.compute_nonlin_power()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        if self.bias_model == 'Linear':
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_d1m'] = pkmm
            pkd['pk_d1d1'] = pkmm
        elif self.is_PT_bias:
            if self.k_SN_suppress > 0:
                k_filter = self.k_SN_suppress
            else:
                k_filter = None
            if self.bias_model == 'EulerianPT':
                from .ept import EPTCalculator
                EPTkwargs = self.pk_options["EPTkwargs"]
                ptc = EPTCalculator(with_NC=True, with_IA=False,
                                    log10k_min=self.l10k_min_pks,
                                    log10k_max=self.l10k_max_pks,
                                    nk_per_decade=EPTkwargs["nk_per_dex_pks"],
                                    a_arr=EPTkwargs["a_s_pks"],
                                    k_filter=k_filter)
            else:
                raise NotImplementedError("Not yet: " + self.bias_model)
            pk_lin_z0 = ccl.linear_matter_power(cosmo, ptc.ks, 1.)
            Dz = ccl.growth_factor(cosmo, ptc.a_s)
            ptc.update_pk(pk_lin_z0, Dz)
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_md2'] = ptc.get_pk('d1d2')
            pkd['pk_ms2'] = ptc.get_pk('d1s2')
            pkd['pk_mk2'] = ptc.get_pk('d1k2', pgrad=pkmm, cosmo=cosmo)
            pkd['pk_d1m'] = pkd['pk_md1']
            pkd['pk_d1d1'] = pkmm
            pkd['pk_d1d2'] = pkd['pk_md2']
            pkd['pk_d1s2'] = pkd['pk_ms2']
            pkd['pk_d1k2'] = pkd['pk_mk2']
            pkd['pk_d2m'] = pkd['pk_md2']
            pkd['pk_d2d1'] = pkd['pk_d1d2']
            pkd['pk_d2d2'] = ptc.get_pk('d2d2')
            pkd['pk_d2s2'] = ptc.get_pk('d2s2')
            pkd['pk_d2k2'] = None
            pkd['pk_s2m'] = pkd['pk_ms2']
            pkd['pk_s2d1'] = pkd['pk_d1s2']
            pkd['pk_s2d2'] = pkd['pk_d2s2']
            pkd['pk_s2s2'] = ptc.get_pk('s2s2')
            pkd['pk_s2k2'] = None
            pkd['pk_k2m'] = pkd['pk_mk2']
            pkd['pk_k2d1'] = pkd['pk_d1k2']
            pkd['pk_k2d2'] = pkd['pk_d2k2']
            pkd['pk_k2s2'] = pkd['pk_s2k2']
            pkd['pk_k2k2'] = None
        return pkd

    def get_can_provide(self):
        return ["is_PT_bias"]

    def get_is_PT_bias(self):
        return self.is_PT_bias
