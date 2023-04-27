import numpy as np
import pyccl as ccl
import fastpt as fpt


class EPTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    Args:
        with_NC (bool): set to True if you'll want to use
            this calculator to compute correlations involving
            number counts.
        with_IA(bool): set to True if you'll want to use
            this calculator to compute correlations involving
            intrinsic alignments.
        with_dd(bool): set to True if you'll want to use
            this calculator to compute the one-loop matter power
            spectrum.
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        a_arr (array_like): array of scale factors at which
            growth/bias will be evaluated.
        pad_factor (float): fraction of the log(k) interval
            you want to add as padding for FFTLog calculations
            within FAST-PT.
        low_extrap (float): decimal logaritm of the minimum
            Fourier scale (in Mpc^-1) for which FAST-PT will
            extrapolate.
        high_extrap (float): decimal logaritm of the maximum
            Fourier scale (in Mpc^-1) for which FAST-PT will
            extrapolate.
        P_window (array_like or None): 2-element array describing
            the tapering window used by FAST-PT. See FAST-PT
            documentation for more details.
        C_window (float): `C_window` parameter used by FAST-PT
            to smooth the edges and avoid ringing. See FAST-PT
            documentation for more details.
    """
    def __init__(self, with_NC=False, with_IA=False, with_dd=True,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 a_arr=None, k_filter=None,
                 pad_factor=1, low_extrap=-5, high_extrap=3,
                 P_window=None, C_window=.75):
        self.with_dd = with_dd
        self.with_NC = with_NC
        self.with_IA = with_IA
        self.P_window = P_window
        self.C_window = C_window

        to_do = ['one_loop_dd']
        if self.with_NC:
            to_do.append('dd_bias')
        if self.with_IA:
            to_do.append('IA')

        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        n_pad = int(pad_factor * len(self.ks))
        if a_arr is None:
            a_arr = 1./(1+np.linspace(0., 4., 30)[::-1])
        self.a_s = a_arr
        if k_filter is not None:
            self.wk_low = 1-np.exp(-(self.ks/k_filter)**2)
        else:
            self.wk_low = np.ones(nk_total)

        self.pt = fpt.FASTPT(self.ks, to_do=to_do,
                             low_extrap=low_extrap,
                             high_extrap=high_extrap,
                             n_pad=n_pad)
        self.one_loop_dd = None
        self.dd_bias = None
        self.ia_ta = None
        self.ia_tt = None
        self.ia_mix = None
        self.g4 = None

    def update_pk(self, cosmo, bcmpar={}):
        """ Update the internal PT arrays.

        Args:
            cosmo:
        """
        pk = cosmo.linear_matter_power(self.ks, 1.)
        Dz = cosmo.growth_factor(self.a_s)
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        if Dz.shape != self.a_s.shape:
            raise ValueError("Input growth has wrong shape")
        if self.with_NC:
            self._get_dd_bias(pk)
            self.with_dd = True
        elif self.with_dd:
            self._get_one_loop_dd(pk)
        if self.with_IA:
            self._get_ia_bias(pk)
        self.g4 = Dz**4
        self.pk2d_computed = {}

    def _get_one_loop_dd(self, pk):
        # Precompute quantities needed for one-loop dd
        # power spectra. Only needed if dd_bias is not called.
        self.one_loop_dd = self.pt.one_loop_dd(pk,
                                               P_window=self.P_window,
                                               C_window=self.C_window)

    def _get_dd_bias(self, pk):
        # Precompute quantities needed for number counts
        # power spectra.
        self.dd_bias = self.pt.one_loop_dd_bias(pk,
                                                P_window=self.P_window,
                                                C_window=self.C_window)
        self.one_loop_dd = self.dd_bias[0:1]

    def _get_ia_bias(self, pk):
        # Precompute quantities needed for intrinsic alignment
        # power spectra.
        self.ia_ta = self.pt.IA_ta(pk,
                                   P_window=self.P_window,
                                   C_window=self.C_window)
        self.ia_tt = self.pt.IA_tt(pk,
                                   P_window=self.P_window,
                                   C_window=self.C_window)
        self.ia_mix = self.pt.IA_mix(pk,
                                     P_window=self.P_window,
                                     C_window=self.C_window)

    def get_pk(self, kind, pnl=None, cosmo=None, sub_lowk=False, alt=None):
        # Clarification:
        # We are expanding the galaxy overdensity as:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2/2 + bk k^2 d
        # (see cell 10 in
        # https://github.com/JoeMcEwen/FAST-PT/blob/master/examples/fastpt_examples.ipynb). # noqa
        # The `dd_bias` array below contains the following power
        # spectra in order:
        #  <d,d^2>
        #  <d^2,d^2> (!)
        #  <d,s^2>
        #  <d^2,s^2> (!)
        #  <s^2,s^2> (!)
        # So: the d^2 and s^2 are not divided by 2
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        # The power spectra provided below are (kind -> Pk):
        # mm -> <d*d> (returns pnl)
        # md1 -> <d*d> (returns pnl)
        # md2 -> <d*d^2/2>
        # ms2 -> <d*s^2/2>
        # mk2 -> k^2 <d*d> (with <d*d> as pnl)
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

        kind = kind.replace('m', 'd1')

        if kind in self.pk2d_computed:
            return self.pk2d_computed[kind]

        if kind == 'd1d1':
            return pnl

        if kind == 'd1k2':
            pk = np.array([pnl.eval(self.ks, a, cosmo)*self.ks**2
                           for a in self.a_s])
            pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks),
                            pk_arr=pk, is_logp=False)
            self.pk2d_computed[kind] = pk2d
            return pk2d

        inds = {'d1d2': 2,
                'd1s2': 4,
                'd2d2': 3,
                'd2s2': 5,
                's2s2': 6}
        filt = {'d1d2': 1.,
                'd1s2': 1.,
                'd2d2': self.wk_low,
                'd2s2': self.wk_low,
                's2s2': self.wk_low}
        sfac = {'d1d2': 0.,
                'd1s2': 0.,
                'd2d2': 2.,
                'd2s2': 4./3.,
                's2s2': 8./9.}
        pfac = {'d1d2': 0.5, # d^2
                'd1s2': 0.5, # s^2
                'd2d2': 0.25, # d^2, d^2
                'd2s2': 0.25, # d^2, s^2
                's2s2': 0.25} # s^2, s^2

        if kind not in inds:
            return alt

        s4 = 0.
        if sub_lowk:
            s4 = self.g4*self.dd_bias[7]
            s4 = s4[:, None]

        pk = pfac[kind]*self.g4[:, None]*((filt[kind]*self.dd_bias[inds[kind]])[None, :] -
                                          sfac[kind]*s4)
        pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks),
                        pk_arr=pk, is_logp=False)
        self.pk2d_computed[kind] = pk2d
        return pk2d
