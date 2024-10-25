import numpy as np
import pyccl as ccl
import baccoemu
import warnings
import copy
from scipy import optimize

class BaccoCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    Args:
        a_arr (array_like): array of scale factors at which
            growth/bias will be evaluated.
    """
    def __init__(self, log10k_min=np.log10(0.008), log10k_max=np.log10(0.5), nk_per_decade=20,
                 log10k_sh_sh_min=np.log10(0.0001), log10k_sh_sh_max=np.log10(50), nk_sh_sh_per_decade=20,
                 a_arr=None, nonlinear_emu_path=None, nonlinear_emu_details=None,
                 nonlinear_emu_model_name=None, use_baryon_boost=False,
                 ignore_lbias=False, allow_bcm_emu_extrapolation_for_shear=True,
                 allow_halofit_extrapolation_for_shear=False,
                 allow_halofit_extrapolation_for_shear_on_k=False):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        nk_sh_sh_total = int((log10k_sh_sh_max - log10k_sh_sh_min) * nk_sh_sh_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        self.ks_sh_sh = np.logspace(log10k_sh_sh_min, log10k_sh_sh_max, nk_sh_sh_total)
        self.use_baryon_boost = use_baryon_boost
        self.ignore_lbias = ignore_lbias
        self.allow_bcm_emu_extrapolation_for_shear = allow_bcm_emu_extrapolation_for_shear
        self.allow_halofit_extrapolation_for_shear = allow_halofit_extrapolation_for_shear
        self.allow_halofit_extrapolation_for_shear_on_k = allow_halofit_extrapolation_for_shear_on_k

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            self.lbias = baccoemu.Lbias_expansion()
            self.mpk = baccoemu.Matter_powerspectrum(nonlinear_emu_path=nonlinear_emu_path,
                                                     nonlinear_emu_details=nonlinear_emu_details,
                                                     model_name=nonlinear_emu_model_name
                                                    )

        # check with the currently loaded version of baccoemu if the a array is
        # all within the allowed ranges
        emu_kind = 'baryon' if self.use_baryon_boost else 'nonlinear'
        amin = self.mpk.emulator[emu_kind]['bounds'][-1][0]
        if a_arr is None:
            zmax = 1/amin - 1
            # Only 20 a's to match the a's in the other PT classes with
            # a < ~0.275
            a_arr = 1./(1+np.linspace(0., zmax, 20)[::-1])
        if np.any(a_arr < amin):
            # This check is done by baccoemu but is not printed by Cobaya, so I
            # add the test here.
            raise ValueError("baccoemu only defined for scale factors between "
                             f"1 and {amin}")
        self.a_s = a_arr

    def _check_baccoemu_baryon_pars_for_extrapolation(self, cosmopars_in):
        """ Check passed parameters, if pars for baryon emu out of range,
        return a new dictionary apt for extrapolation.

        Extrapolation of the bcm emulator in cosmology is done by evaluating
        the emu at the closest cosmology within the allowed parameter space,
        while modifying Ob and Oc to keep the baryon fraction fixed
        """
        cosmopars = copy.deepcopy(cosmopars_in)

        # Return cosmopars to get the sigma8_cold equivalent to the input As
        within_bounds, cosmopars = self._check_within_bounds(cosmopars,
                                                             return_cosmopars=True)
        if (not self.allow_bcm_emu_extrapolation_for_shear) or \
            within_bounds['baryon']:
            return copy.deepcopy(cosmopars), copy.deepcopy(self.a_s)

        cosmopars_out = copy.deepcopy(cosmopars)

        b_frac_orig = cosmopars['omega_baryon']/cosmopars['omega_cold']

        emulator = self.mpk.emulator['baryon']

        for i, par in enumerate(emulator['keys']):
            if (par in self.mpk.emulator['nonlinear']['keys']):
                if par in cosmopars:
                    if cosmopars[par] is None:
                        del cosmopars_out[par]
                    else:
                        if (cosmopars[par] < emulator['bounds'][i][0]):
                            cosmopars_out[par] = emulator['bounds'][i][0]
                        elif (cosmopars[par] > emulator['bounds'][i][1]):
                            cosmopars_out[par] = emulator['bounds'][i][1]

        b_frac = cosmopars_out['omega_baryon']/cosmopars_out['omega_cold']
        if np.round(b_frac_orig, 4) != np.round(b_frac, 4):
            min_func = lambda o: np.abs(o[1] / o[0] - b_frac_orig)
            Oc_bounds = emulator['bounds'][0]
            Ob_bounds = emulator['bounds'][2]
            res = optimize.minimize(min_func,
                                    np.array([cosmopars_out['omega_cold'], cosmopars_out['omega_baryon']]),
                                    bounds=(Oc_bounds, Ob_bounds))
            cosmopars_out['omega_cold'] = res.x[0]
            cosmopars_out['omega_baryon'] = res.x[1]

        a_s_out = copy.deepcopy(self.a_s)
        a_s_out[a_s_out < emulator['bounds'][-1][0]] = emulator['bounds'][-1][0]

        return cosmopars_out, a_s_out

    def _check_within_bounds(self, cosmopars, return_cosmopars=False):
        """
        Check if cosmological parameters are within bounds

        Return: dict with keys 'nonlinear' and 'baryon'. If return_cosmopars is
        True, returns the cosmopars with sigma8_cold instead of A_s.
        """
        cosmopars = copy.deepcopy(cosmopars)
        if 'A_s' in cosmopars:
            cosmopars['sigma8_cold'] = self.mpk.get_sigma8(**cosmopars, cold=True)
            del cosmopars['A_s']
        within_bounds = []
        within_bounds_mpk = []
        for i, parname in enumerate(self.mpk.emulator['nonlinear']['keys']):
            if parname != 'expfactor':
                val = cosmopars[parname]
            else:
                val = copy.deepcopy(self.a_s)
            within_bounds.append(np.all(val >= self.mpk.emulator['baryon']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['baryon']['bounds'][i][1]))
            within_bounds_mpk.append(np.all(val >= self.mpk.emulator['nonlinear']['bounds'][i][0]) & np.all(val <= self.mpk.emulator['nonlinear']['bounds'][i][1]))

        output = {'nonlinear': np.all(within_bounds_mpk),
                'baryon': np.all(within_bounds)}
        if return_cosmopars:
            return output, cosmopars

        return output

    def _sigma8tot_2_sigma8cold(self, emupars, sigma8tot):
        """Use baccoemu to convert sigma8 total matter to sigma8 cdm+baryons
        """
        if hasattr(emupars['omega_cold'], '__len__'):
            _emupars = {}
            for pname in emupars:
                _emupars[pname] = emupars[pname][0]
        else:
            _emupars = emupars
        A_s_fid = 2.1e-9
        sigma8tot_fid = self.mpk.get_sigma8(cold=False,
                                            A_s=A_s_fid, **_emupars)
        A_s = (sigma8tot / sigma8tot_fid)**2 * A_s_fid
        return self.mpk.get_sigma8(cold=True, A_s=A_s, **_emupars)


    def update_pk(self, cosmo, bcmpar=None, **kwargs):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        cospar = self._get_bacco_pars_from_cosmo(cosmo)
        h = cospar['hubble']
        cospar_and_a = self._get_pars_and_a_for_bacco(cospar, self.a_s)

        # HEFT
        k_for_bacco = self.ks/h
        # TODO: Use lbias.emulator['nonlinear']['k'].max() instead of 0.75?
        self.mask_ks_for_bacco = np.squeeze(np.where(k_for_bacco <= 0.75))
        k_for_bacco = k_for_bacco[self.mask_ks_for_bacco]
        if self.ignore_lbias:
            self.pk_temp = None
        else:
            self.pk_temp = self.lbias.get_nonlinear_pnn(k=k_for_bacco,
                                                        **cospar_and_a)[1]/h**3

        # Shear - Shear (and baryons)
        baryonic_boost = self.use_baryon_boost and (bcmpar is not None)

        k_sh_sh_for_bacco = self.ks_sh_sh/h
        emu_type_for_setting_kmax = 'baryon' if baryonic_boost else 'nonlinear'
        self.mask_ks_sh_sh_for_bacco = np.squeeze(np.where(k_sh_sh_for_bacco <= self.mpk.emulator[emu_type_for_setting_kmax]['k'].max()))
        k_sh_sh_for_bacco = k_sh_sh_for_bacco[self.mask_ks_sh_sh_for_bacco]

        within_bounds_mpk = self._check_within_bounds(cospar)['nonlinear']

        if (not within_bounds_mpk) & self.allow_halofit_extrapolation_for_shear:
            cosmo.compute_nonlin_power()
            pknl = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            pk = np.array([pknl.eval(self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco], a, cosmo) for a in self.a_s])
        else:
            # TODO: This is going to be called even if no baryons are
            # requested. Shouldn't it have a flag?
            pk = self.mpk.get_nonlinear_pk(baryonic_boost=False, cold=False,
                                           k=k_sh_sh_for_bacco,
                                           **cospar_and_a)[1]/h**3

        if baryonic_boost:
            Sk = self.get_baryonic_boost(cosmo, bcmpar, k_sh_sh_for_bacco)
        else:
            Sk = np.ones_like(pk)

        if self.allow_halofit_extrapolation_for_shear_on_k:
            cosmo.compute_nonlin_power()
            pknl = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            kix = self.mask_ks_sh_sh_for_bacco[-1] + 1
            pkhfit = [pknl(self.ks_sh_sh[kix:], a) for a in self.a_s]
            pk = np.concatenate([pk, pkhfit], axis=1)
            # Extrapolating as in CCL. We could come up with different
            # extrapolation schemes (e.g. Sk = constant?)
            Sk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco]),
                            pk_arr=np.log(Sk), is_logp=True)
            Sk = np.array([Sk2d(self.ks_sh_sh, ai) for ai in self.a_s])

        self.pk_temp_sh_sh = pk * Sk
        self.Sk_temp = Sk
        self.pk2d_computed = {}

    def _get_pars_and_a_for_bacco(self, pars, a):
        combined_pars = {}
        for key in pars.keys():
            combined_pars[key] = np.full((len(a)), pars[key])
        combined_pars['expfactor'] = a

        return combined_pars

    def _get_bacco_pars_from_cosmo(self, cosmo):
        cospar = {
            'omega_cold': cosmo['Omega_c'] + cosmo['Omega_b'],
            'omega_baryon': cosmo['Omega_b'],
            'ns': cosmo['n_s'],
            'hubble': cosmo['h'],
            'neutrino_mass': np.sum(cosmo['m_nu']),
            'w0': cosmo['w0'],
            'wa': cosmo['wa']}
        if np.isnan(cosmo['A_s']):
            cospar['sigma8_cold'] = self._sigma8tot_2_sigma8cold(cospar, cosmo.sigma8())
        else:
            cospar['A_s'] = cosmo['A_s']

        return cospar

    def get_baryonic_boost(self, cosmo, bcmpar, k_arr):
        cospar = self._get_bacco_pars_from_cosmo(cosmo)
        cospar_for_bcm, these_a_s = self._check_baccoemu_baryon_pars_for_extrapolation(cospar)
        cospar_for_bcm.update(bcmpar)
        cospar_for_bcm = self._get_pars_and_a_for_bacco(cospar_for_bcm,
                                                        these_a_s)
        Sk = self.mpk.get_baryonic_boost(k=k_arr, **cospar_for_bcm)[1]
        return Sk

    def get_pk(self, kind, pnl=None, cosmo=None, sub_lowk=False, alt=None):
        # Clarification:
        # We are expanding the galaxy overdensity as:
        #   1+ d_g = 1 + b1 d + b2 d2^2/2 + bs s^2/2 + bk k^2 d
        # But Bacco assumes
        #   1+d_g = 1 + b1 d + b2 d2^2 + bs s^2 + bk k^2 d
        # The order of pk_Temp:
        #  11, 1d, 1d2, 1s2, 1k2, dd, dd2, ds2, dk2, d2d2, d2s2, d2k2, s2s2, s2k2, k2k2
        # This will return
        # mm -> <1*1> (from bacco)
        # md1 -> <1*d> (from bacco)
        # md2 -> <1*d^2/2>
        # ms2 -> <1*s^2/2>
        # mk2 -> <k2*k2> (with <d*d> as pnl)
        # d1d1 -> <d*d> (returns pnl)
        # d1d2 -> <d*d^2/2>
        # d1s2 -> <d*s^2/2>
        # d1k2 -> k^2 <d*d> (with <d*d> as pnl)
        # d2d2 -> <d^2/2*d^2/2>
        # d2s2 -> <d^2/2*s^2/2>
        # d2k2 -> k^2 <d*d^2/2>, not provided
        # s2s2 -> <s^2/2*s^2/2>
        # s2k2 -> k^2 <d*s^2/2>, not provided
        # k2k2 -> k^4 <d*d>, not provided
        # When not provided, this function just returns `alt`

        if kind in self.pk2d_computed:
            return self.pk2d_computed[kind]

        inds = {'mm': 0,
                'md1': 1,
                'md2': 2,
                'ms2': 3,
                'mk2': 4,
                'd1d1': 5,
                'd1d2': 6,
                'd1s2': 7,
                'd1k2': 8,
                'd2d2': 9,
                'd2s2': 10,
                'd2k2': 11,
                's2s2': 12,
                's2k2': 13,
                'k2k2': 14}
        pfac = {'mm': 1.0,
                'md1': 1.0,
                'md2': 0.5,
                'ms2': 0.5,
                'mk2': 1.0,
                'd1d1': 1.0,
                'd1d2': 0.5,
                'd1s2': 0.5,
                'd1k2': 1.0,
                'd2d2': 0.25,
                'd2s2': 0.25,
                'd2k2': 0.5,
                's2s2': 0.25,
                's2k2': 0.5,
                'k2k2': 1.0}

        if kind == 'Sk':
            pk = np.log(self.Sk_temp)
            if self.allow_halofit_extrapolation_for_shear_on_k:
                k = self.ks_sh_sh
            else:
                k = self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco]
            pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(k), pk_arr=pk,
                            is_logp=True)
            self.pk2d_computed[kind] = pk2d
        elif kind == 'mm_sh_sh':
            if self.allow_halofit_extrapolation_for_shear_on_k:
                k = self.ks_sh_sh
            else:
                k = self.ks_sh_sh[self.mask_ks_sh_sh_for_bacco]
            pk = np.log(self.pk_temp_sh_sh)
            pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(k), pk_arr=pk,
                            is_logp=True)
            self.pk2d_computed[kind] = pk2d
        else:
            if not self.ignore_lbias:
                pk = pfac[kind]*self.pk_temp[inds[kind], :, :]
                if kind in ['mm']:
                    pk = np.log(pk)
                pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks[self.mask_ks_for_bacco]),
                                pk_arr=pk, is_logp=kind in ['mm'])
                self.pk2d_computed[kind] = pk2d
            else:
                pk2d = None

        return pk2d
