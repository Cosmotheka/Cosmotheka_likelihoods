import numpy as np
import pyccl as ccl
import baccoemu
from scipy.interpolate import interp1d


_heft_inds = {'mm': 0,
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
_heft_logd = {'mm': True,
              'md1': True,
              'md2': False,
              'ms2': False,
              'mk2': False,
              'd1d1': True,
              'd1d2': False,
              'd1s2': False,
              'd1k2': False,
              'd2d2': True,
              'd2s2': False,
              'd2k2': False,
              's2s2': True,
              's2k2': False,
              'k2k2': True}
_heft_pfac = {'mm': 1.0,
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


class BaccoCalculatorHEFT(object):
    def __init__(self):
        self.lbias = baccoemu.Lbias_expansion()
        self.mpk = baccoemu.Matter_powerspectrum()

        self.z_s = np.linspace(0.0, 1.0, 20)[::-1]
        self.a_s = 1/(1+self.z_s)
        self.nk_ccl = 64
        self.k_ccl = np.geomspace(1E-3, 10, self.nk_ccl)

        self.dx_par = {'omega_cold': 0.01,
                       'omega_baryon': 0.01,
                       'sigma8_cold': 0.01,
                       'ns': 0.01,
                       'hubble': 0.01,
                       'neutrino_mass': 0.01,
                       'w0': 0.01,
                       'wa': 0.01}
        self.lbias_bounds = self._get_bounds()
        self.kh_bacco = self.lbias.emulator['nonlinear']['k']
        self.log_der = np.ones(15, dtype=bool)
        for k, v in _heft_logd.items():
            self.log_der[_heft_inds[k]] = v
        self.lin_der = ~ self.log_der

    def _get_bounds(self):
        bounds = {}
        for i, k in enumerate(self.lbias.emulator['nonlinear']['keys']):
            b = self.lbias.emulator['nonlinear']['bounds'][i]
            bounds[k] = [b[0], b[1]]
        return bounds

    def _ccl2bacco(self, cosmo):
        cospar = {
            'omega_cold': cosmo['Omega_c'] + cosmo['Omega_b'],
            'omega_baryon': cosmo['Omega_b'],
            'ns': cosmo['n_s'],
            'hubble': cosmo['h'],
            'neutrino_mass': np.sum(cosmo['m_nu']),
            'w0': cosmo['w0'],
            'wa': cosmo['wa']}

        if np.isnan(cosmo['A_s']):
            s8tot = cosmo.sigma8()
            A_s_fid = 2.1E-9
            s8tot_fid = self.mpk.get_sigma8(cold=False, A_s=A_s_fid, **cospar)
            A_s = (s8tot / s8tot_fid)**2 * A_s_fid
            s8cold = self.mpk.get_sigma8(cold=True, A_s=A_s, **cospar)
            cospar['sigma8_cold'] = s8cold
        else:
            cospar['A_s'] = cosmo['A_s']
        return cospar

    def _bacco2ccl(self, cospar, tf=None):
        Omega_b = cospar['omega_baryon']
        Omega_m = cospar['omega_cold']
        Omega_c = Omega_m - Omega_b
        n_s = cospar['ns']
        h = cospar['hubble']
        m_nu = cospar['neutrino_mass']
        w0 = cospar['w0']
        wa = cospar['wa']

        A_s = cospar.get('A_s', None)
        sigma8 = None
        if A_s is None:
            s8cold = cospar['sigma8_cold']
            cpar = cospar.copy()
            cpar.pop('sigma8_cold')
            A_s_fid = 2.1E-9
            s8cold_fid = self.mpk.get_sigma8(cold=True, A_s=A_s_fid, **cpar)
            s8tot_fid = self.mpk.get_sigma8(cold=False, A_s=A_s_fid, **cpar)
            sigma8 = s8cold * s8tot_fid / s8cold_fid
            transfer = 'boltzmann_camb' if tf is None else tf

        cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b,
                              h=h, m_nu=m_nu, w0=w0, wa=wa,
                              sigma8=sigma8, A_s=A_s, n_s=n_s,
                              transfer_function=transfer)
        return cosmo

    def _bacco_pars_a(self, pars, a):
        combined_pars = {}
        for key in pars.keys():
            combined_pars[key] = np.full((len(a)), pars[key])
            combined_pars['expfactor'] = a
        return combined_pars

    def _get_out_of_bounds(self, cospar):
        in_bounds = {}
        par_0 = {}
        for par, val in cospar.items():
            val0, valf = self.lbias_bounds[par]
            in_bounds[par] = (val >= val0) & (val <= valf)
            if not in_bounds[par]:
                if val < val0:
                    par_0[par] = val0+self.dx_par[par]
                else:
                    par_0[par] = valf-self.dx_par[par]
            else:
                par_0[par] = val
        return in_bounds, par_0

    def _get_lbias_pks_exact(self, cospar, mm_only=False):
        cosmo = self._bacco2ccl(cospar, tf='eisenstein_hu')
        h = cosmo['h']

        k_ccl_here = self.kh_bacco * h
        pk_mm = np.array([ccl.nonlin_matter_power(cosmo, k_ccl_here, a)
                          for a in self.a_s])*h**3
        if mm_only:
            return pk_mm

        cospar_and_a = self._bacco_pars_a(cospar, self.a_s)
        pks = np.array(self.lbias.get_nonlinear_pnn(**cospar_and_a)[1])

        return pk_mm, pks

    def _get_ccl_pks_from_bacco_pks(self, cosmo, pks):
        if pks.ndim == 2:  # Check if this is a single power spectrum
            pks = np.array([pks])

        npks, nas, nks = pks.shape
        assert nks == len(self.kh_bacco)
        assert nas == len(self.a_s)
        pks = pks.reshape([npks*nas, nks])

        h = cosmo['h']
        kh_ccl = self.k_ccl / h
        klo = kh_ccl < self.kh_bacco[0]
        khi = kh_ccl > self.kh_bacco[-1]
        kmd = ~(klo + khi)

        pks_out = np.zeros([npks*nas, self.nk_ccl])
        # Interpolate within range
        pks_out[:, kmd] = interp1d(np.log(self.kh_bacco), pks)(np.log(kh_ccl[kmd]))

        if np.any(klo) or np.any(khi):
            pkas = np.fabs(pks)

        if np.any(khi):
            # Log-extrapolate at high-k
            # Calculate logarithmic slope at high-k
            slopes = np.log(pkas[:, -2]/pkas[:, -1]) / \
                np.log(self.kh_bacco[-2]/self.kh_bacco[-1])
            # Set slope to zero if pk changes sign
            sign_swap = pks[:, -1] * pks[:, -2] < 0
            slopes[sign_swap] = 0
            slopes[slopes > 0] = 0
            # Extrapolate
            pk_extrap = (pks[:, -1])[:, None] * \
                ((kh_ccl[khi]/self.kh_bacco[-1])[None, :]**slopes[:, None])
            pks_out[:, khi] = pk_extrap

        if np.any(klo):
            # Log-extrapolate at low-k
            # Calculate logarithmic slope at low-k
            slopes = np.log(pkas[:, 0]/pkas[:, 1]) /  \
                np.log(self.kh_bacco[0]/self.kh_bacco[1])
            # Set slope to zero if pk changes sign
            sign_swap = pks[:, 0] * pks[:, 1] < 0
            slopes[sign_swap] = 0
            slopes[slopes < 0] = 0
            # Extrapolate
            pk_extrap = (pks[:, 0])[:, None] * \
                ((kh_ccl[klo]/self.kh_bacco[0])[None, :]**slopes[:, None])
            pks_out[:, klo] = pk_extrap

        pks_out = pks_out.reshape([npks, nas, self.nk_ccl])
        pks = pks.reshape([npks, nas, nks])

        # CCL units
        pks_out *= 1/h**3

        return pks_out.squeeze()

    def get_lbias_pks(self, cosmo):
        cospar = self._ccl2bacco(cosmo)
        in_bounds, cospar_0 = self._get_out_of_bounds(cospar)

        pk_mm_0, pks_0 = self._get_lbias_pks_exact(cospar_0)

        # Within bounds, no parameter extrapolation
        if np.all(list(in_bounds.values())):
            return self._get_ccl_pks_from_bacco_pks(cosmo, pks_0)

        # Sign
        sign = np.sign(pks_0)
        lx = np.log(np.fabs(pks_0)/pk_mm_0)
        x = pks_0 / pk_mm_0

        # Controlled parameter extrapolation
        for par, in_bound in in_bounds.items():
            if in_bound:
                continue
            # print(f"Extrapolating {par}, {cospar[par]}, {cospar_0[par]}")

            cpar = cospar_0.copy()
            cpar[par] += self.dx_par[par]
            pk_mm_p, pks_p = self._get_lbias_pks_exact(cpar)
            lxp = np.log(np.fabs(pks_p)/pk_mm_p)
            xp = pks_p / pk_mm_p

            cpar = cospar_0.copy()
            cpar[par] -= self.dx_par[par]
            pk_mm_m, pks_m = self._get_lbias_pks_exact(cpar)
            lxm = np.log(np.fabs(pks_m)/pk_mm_m)
            xm = pks_m / pk_mm_m

            dlx_dp = (lxp-lxm)/(2*self.dx_par[par])
            dx_dp = (xp-xm)/(2*self.dx_par[par])

            lx += dlx_dp * (cospar[par] - cospar_0[par])
            x += dx_dp * (cospar[par] - cospar_0[par])

        pk_mm = self._get_lbias_pks_exact(cospar, mm_only=True)
        pks = pk_mm * (sign*np.exp(lx)*self.log_der[:, None, None] +
                       x * self.lin_dr[:, None, None])

        return self._get_ccl_pks_from_bacco_pks(cosmo, pks)

    def update_pk(self, cosmo, bcmpar=None, **kwargs):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        self.pk_temp = self.get_lbias_pks(cosmo)
        self.pk2d_computed = {}

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

        pk = _heft_pfac[kind]*self.pk_temp[_heft_inds[kind], :, :]
        autos = ['mm', 'd1d1', 'd2d2', 's2s2', 'k2k2']
        is_logp = kind in autos
        if is_logp:
            pk = np.log(pk)
        pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.k_ccl),
                        pk_arr=pk, is_logp=is_logp)
        self.pk2d_computed[kind] = pk2d

        return pk2d
