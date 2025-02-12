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
    #for baccoemu
    nonlinear_emu_path = None
    nonlinear_emu_details = None
    use_baryon_boost : bool = False
    baryon_model: str = ''
    ignore_lbias : bool = False
    allow_bcm_emu_extrapolation_for_shear : bool = True
    allow_halofit_extrapolation_for_shear : bool = False
    allow_halofit_extrapolation_for_shear_on_k: bool = False

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

        if self.baryon_model not in ['', 'Bacco', 'CCL_BCM', 'Amon-Efstathiou']:
            raise ValueError("baryon_model must be one of '', 'Bacco' or "
                             "'CCL_BCM', 'Amon-Efstathiou'")

        if self.bias_model == 'BaccoPT':
            if self.use_baryon_boost and self.baryon_model == '':
                    self.baryon_model = 'Bacco'

            use_baryon_boost = self.use_baryon_boost and \
                               self.baryon_model == 'Bacco'
            self.bacco_calc = BaccoCalculator(a_arr=self.a_s_pks,
                                              nonlinear_emu_path=self.nonlinear_emu_path,
                                              nonlinear_emu_details=self.nonlinear_emu_details,
                                              use_baryon_boost=use_baryon_boost,
                                              ignore_lbias=self.ignore_lbias,
                                              allow_bcm_emu_extrapolation_for_shear=self.allow_bcm_emu_extrapolation_for_shear,
                                              allow_halofit_extrapolation_for_shear=self.allow_halofit_extrapolation_for_shear,
                                              allow_halofit_extrapolation_for_shear_on_k=self.allow_halofit_extrapolation_for_shear_on_k
                                             )
        else:
            if self.baryon_model == 'Bacco':
                raise ValueError("baryon_model 'Bacco' can only be used with "
                                 "bias_model 'BaccoPT' at the moment.")

    def must_provide(self, **requirements):
        if "Pk" not in requirements:
            return {}

        return {"CCL": None}

    def get_can_support_params(self):
        # TODO: We should better use an input_params_prefix to avoid confusion
        # with CCL params and maybe split the baryons stuff
        return ["M_c", "eta", "beta", "M1_z0_cen", "theta_out", "theta_inn",
                "M_inn", "A_AE"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        cosmo = self.provider.get_CCL()["cosmo"]
        bcmpar = None
        if self.use_baryon_boost:
            if self.baryon_model == 'Bacco':
                M_c = self.provider.get_param('M_c')
                eta = self.provider.get_param('eta')
                beta = self.provider.get_param('beta')
                M1_z0_cen = self.provider.get_param('M1_z0_cen')
                theta_out = self.provider.get_param('theta_out')
                theta_inn = self.provider.get_param('theta_inn')
                M_inn = self.provider.get_param('M_inn')
                bcmpar = {
                    'M_c'  : M_c,
                    'eta' : eta,
                    'beta' : beta,
                    'M1_z0_cen' : M1_z0_cen,
                    'theta_out' : theta_out,
                    'theta_inn' : theta_inn,
                    'M_inn' : M_inn
                }
            elif self.baryon_model == 'Amon-Efstathiou':
                bcmpar = {'A_AE': self.provider.get_param('A_AE')}

        state['Pk'] = {'pk_data': self._get_pk_data(cosmo, bcmpar=bcmpar)}

    def get_Pk(self):
        return self._current_state['Pk']

    def _get_pk_data(self, cosmo, bcmpar=None):
        # cosmo.compute_nonlin_power()
        # pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        pkmm = None
        if self.bias_model == 'Linear':
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            if 'delta_matter:Weyl' in cosmo._pk_nl:
                pkwm = pkmw = cosmo.get_nonlin_power(name='delta_matter:Weyl')
            else:
                pkwm = pkmw = pkmm
            if 'Weyl:Weyl' in cosmo._pk_nl:
                pkww = cosmo.get_nonlin_power(name='Weyl:Weyl')
            else:
                pkww = pkmm
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_d1m'] = pkmm
            pkd['pk_d1d1'] = pkmm
            pkd['pk_d1w'] = pkd['pk_wd1'] = pkwm
            pkd['pk_mw'] = pkd['pk_wm'] = pkwm
            pkd['pk_ww'] = pkww
        elif self.is_PT_bias:
            if ('delta_matter:Weyl' in cosmo._pk_nl) or \
                    ('Weyl:Weyl' in cosmo._pk_nl):
                raise RuntimeError('Pk involving the Weyl potential not '
                                   'implemented for PT_bias')
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
            ptc.update_pk(cosmo, bcmpar=bcmpar)
            pkd = {}
            operators = ['m', 'w', 'd1', 'd2', 's2', 'k2']
            for i1, op1 in enumerate(operators):
                for op2 in operators[i1:]:
                    comb_12 = op1+op2
                    # Since PT models are not meant to work with Weyl and we
                    # have already checked if Weyl is in cosmo._pk_nl, let's
                    # fill pkd weyl pk's with matter ones.
                    kind = comb_12.replace('w', 'm')
                    pkd[f'pk_{comb_12}'] = ptc.get_pk(kind, pnl=pkmm,
                                                      cosmo=cosmo)
                    # Symmetric terms for convenience
                    if op1 != op2:
                        comb_21 = op2+op1
                        pkd[f'pk_{comb_21}'] = pkd[f'pk_{comb_12}']
            if (self.bias_model == 'BaccoPT') and self.ignore_lbias:
                # TODO: Move this to bacco.py
                # In case we don't request the bias expansion, use the pk from
                # the matter pk emulator
                pkd['pk_ww'] = ptc.get_pk('mm_sh_sh', pnl=pkmm, cosmo=cosmo)

        # Add baryon correction
        baryons_in_cosmo = cosmo._config_init_kwargs['baryons_power_spectrum']
        if self.use_baryon_boost or (baryons_in_cosmo != 'nobaryons'):
            if self.is_PT_bias and (self.bias_model == 'BaccoPT') and \
                (self.baryon_model == 'Bacco'):
                # TODO: This assumes LCDM, but as above. BACCOemu is
                # trained only in LCDM, anyway. What to do with the cross
                # pk's? At the moment they don't have the correction.
                pkd['pk_ww'] = ptc.get_pk('mm_sh_sh', pnl=pkmm, cosmo=cosmo)
                pkd['Sk'] = ptc.get_pk('Sk')
            elif (self.baryon_model == 'CCL_BCM') or \
                (baryons_in_cosmo == 'bcm'):
                # This can be optimized using BaryonsClass.update_params()
                if self.is_PT_bias:
                    # The correction happens in place
                    # If bias is Linear, then the pk already has the baryon
                    # boost applied.
                    pkd['pk_ww'] = cosmo.baryons.include_baryonic_effects(cosmo, pkd['pk_ww'])
                a_arr, lnk, pkww = pkd['pk_ww'].get_spline_arrays()
                k = np.exp(lnk)
                Sk = np.zeros_like(pkww)
                for i, ai in enumerate(a_arr):
                    Sk[i] = cosmo.baryons.boost(cosmo, k, ai)
                pkd['Sk'] = ccl.Pk2D(a_arr=a_arr, lk_arr=lnk,
                                     pk_arr=np.log(Sk), is_logp=True)
            elif self.baryon_model == 'Amon-Efstathiou':
                pklin = cosmo.get_linear_power()
                a, lnk, pklin = pklin.get_spline_arrays()
                k = np.exp(lnk)
                boost = np.zeros_like(pklin)
                pknonlin = np.zeros_like(pklin)
                for i, ai in enumerate(a):
                    pknonlin[i] = pkd['pk_ww'].eval(k, ai, cosmo)
                    boost[i] = pknonlin[i] - pklin[i]

                pkb = pklin + bcmpar['A_AE']*boost
                pkd['pk_ww'] = ccl.Pk2D(a_arr=a, lk_arr=lnk,
                                        pk_arr=np.log(pkb),
                                        is_logp=True)
                Sk = pkb / pknonlin
                pkd['Sk'] = ccl.Pk2D(a_arr=a, lk_arr=lnk, pk_arr=np.log(Sk),
                                     is_logp=True)
            else:
                # TODO: Bacco returns a pk2d of 1's, maybe homogenize this
                pkd['Sk'] = None

        return pkd

    def get_can_provide(self):
        return ["is_PT_bias", "bias_model"]

    def get_bias_model(self):
        return self.bias_model

    def get_is_PT_bias(self):
        return self.is_PT_bias
