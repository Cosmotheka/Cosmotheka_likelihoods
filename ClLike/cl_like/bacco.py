import numpy as np
import pyccl as ccl
import baccoemu


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
                 a_arr=None):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        # baccoemu only allows a's between 1 and 0.275.
        amin = 0.275
        if a_arr is None:
            zmax = 1/amin - 1
            # Only 20 a's to match the a's in the other PT classes with
            # a < 0.275
            a_arr = 1./(1+np.linspace(0., zmax, 20)[::-1])
        if np.any(a_arr < amin):
            # This check is done by baccoemu but is not printed by Cobaya, so I
            # add the test here.
            raise ValueError("baccoemu only defined for scale factors between "
                             f"0 and {amin}")
        self.a_s = a_arr
        self.lbias = baccoemu.Lbias_expansion(allow_extrapolate=False)

    def update_pk(self, cosmo):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        h = cosmo['h']
        cospar = {
            'omega_cold': cosmo['Omega_c'] + cosmo['Omega_b'],
            'omega_baryon': cosmo['Omega_b'],
            'sigma8_cold': cosmo.sigma8(),
            'ns': cosmo['n_s'],
            'hubble': h,
            'neutrino_mass': np.sum(cosmo['m_nu']),
            'w0': cosmo['w0'],
            'wa': cosmo['wa']}
        k_for_bacco = self.ks/h
        self.mask_ks_for_bacco = np.squeeze(np.where(k_for_bacco <= 0.75))
        k_for_bacco = k_for_bacco[self.mask_ks_for_bacco]
        self.pk_temp = np.array([self.lbias.get_nonlinear_pnn(k=self.ks/h,
                                                              allow_high_k_extrapolation=False, #this is bad, should be improved
                                                              expfactor=a,
                                                              **cospar)[1]/h**3
                                 for a in self.a_s])
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

        pk = pfac[kind]*self.pk_temp[:, inds[kind], :]
        if kind in ['mm']:
            pk = np.log(pk)
        pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks[self.mask_ks_for_bacco]),
                        pk_arr=pk, is_logp=kind in ['mm'])
        self.pk2d_computed[kind] = pk2d
        return pk2d