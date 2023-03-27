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
    from .lpt import LPTCalculator
    HAVE_LPT = True
    LPT_exception = None
except ImportError as e:
    LPT_exception = e
    HAVE_LPT = False

try:
    from .ept import EPTCalculator
    HAVE_EPT = True
    EPT_exception = None
except ImportError as e:
    EPT_exception = e
    HAVE_EPT = False

try:
    from .bacco import BaccoCalculator
    HAVE_BACCO = True
    BACCO_exception = None
except ImportError as e:
    BACCO_exception = e
    HAVE_BACCO = False

    
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
        self.is_PT_bias = self.bias_model in ['LagrangianPT', 'EulerianPT', 'BaccoPT']
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
        elif self.bias_model == 'BaccoPT' and not HAVE_BACCO:
            raise BACCO_exception

        if self.bias_model == 'BaccoPT':
            self.bacco_calc = BaccoCalculator(a_arr=self.a_s_pks)

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
        # cosmo.compute_nonlin_power()
        # pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        pkmm = None
        if self.bias_model == 'Linear':
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
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
                cosmo.compute_nonlin_power()
                pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
                ptc = EPTCalculator(with_NC=True, with_IA=False,
                                    log10k_min=self.l10k_min_pks,
                                    log10k_max=self.l10k_max_pks,
                                    nk_per_decade=self.nk_per_dex_pks,
                                    a_arr=self.a_s_pks,
                                    k_filter=k_filter)
            elif self.bias_model == 'LagrangianPT':
                from .lpt import LPTCalculator
                cosmo.compute_nonlin_power()
                pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
                ptc = LPTCalculator(log10k_min=self.l10k_min_pks,
                                    log10k_max=self.l10k_max_pks,
                                    nk_per_decade=self.nk_per_dex_pks,
                                    a_arr=self.a_s_pks, h=cosmo['h'],
                                    k_filter=k_filter)
            elif self.bias_model == 'BaccoPT':
                ptc = self.bacco_calc
            else:
                raise NotImplementedError("Not yet: " + self.bias_model)
            ptc.update_pk(cosmo)
            pkd = {}
            operators = ['m', 'd1', 'd2', 's2', 'k2']
            for i1, op1 in enumerate(operators):
                for op2 in operators[i1:]:
                    comb_12 = op1+op2
                    pkd[f'pk_{comb_12}'] = ptc.get_pk(comb_12, pnl=pkmm, cosmo=cosmo)
                    # Symmetric terms for convenience
                    if op1 != op2:
                        comb_21 = op2+op1
                        pkd[f'pk_{comb_21}'] = pkd[f'pk_{comb_12}']
        return pkd

    def get_can_provide(self):
        return ["is_PT_bias", "bias_model"]

    def get_bias_model(self):
        return self.bias_model

    def get_is_PT_bias(self):
        return self.is_PT_bias
