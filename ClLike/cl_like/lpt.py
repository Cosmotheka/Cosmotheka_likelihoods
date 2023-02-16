import numpy as np
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
import pyccl as ccl


class LPTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on velocileptors
    (https://github.com/sfschen/velocileptors).

    Args:
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        a_arr (array_like): array of scale factors at which
            growth/bias will be evaluated.
    """
    def __init__(self, log10k_min=-4, log10k_max=2,
                 nk_per_decade=20, a_arr=None, h=None, k_filter=None):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        if a_arr is None:
            a_arr = 1./(1+np.linspace(0., 4., 30)[::-1])
        self.a_s = a_arr
        self.h = h
        self.lpt_table = None
        if k_filter is not None:
            self.wk_low = 1-np.exp(-(self.ks/k_filter)**2)
        else:
            self.wk_low = np.ones(nk_total)

    def update_pk(self, pk, Dz):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        if Dz.shape != self.a_s.shape:
            raise ValueError("Input growth has wrong shape")
        cleft = RKECLEFT(self.ks/self.h, pk*self.h**3)
        self.lpt_table = []
        for D in Dz:
            cleft.make_ptable(D=D, kmin=self.ks[0]/self.h,
                              kmax=self.ks[-1]/self.h, nk=self.ks.size)
            self.lpt_table.append(cleft.pktable)
        self.lpt_table = np.array(self.lpt_table)
        self.lpt_table /= self.h**3
        self.pk2d_computed = {}

    def get_pk(self, kind, pnl=None, cosmo=None, alt=None):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")
        # Clarification:
        # CLEFT uses the followint expansion for the galaxy overdensity:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2
        # (see Eq. 4.4 of https://arxiv.org/pdf/2005.00523.pdf).
        # But we want to use
        #   d_g = b1 d + b2 d2^2/2 + bs s^2/2 + bk k^2 d
        # So, to add to the confusion, this is different from the prescription
        # used by EPT (and which we want to use), where s^2 is divided by 2 :-|
        #
        # The LPT table below contains the following power spectra
        # in order:
        #  <1,1>
        #  2*<1,d>
        #  <d,d>
        #  2*<1,d^2/2>
        #  2*<d,d^2/2>
        #  <d^2/2,d^2/2> (!)
        #  2*<1,s^2>
        #  2*<d,s^2>
        #  2*<d^2/2,s^2> (!)
        #  <s^2,s^2> (!)
        #
        # So:
        #   a) The cross-correlations need to be divided by 2.
        #   b) The spectra involving b2 are for d^2/2, NOT d^2!!
        #   c) The spectra invoving bs are for s^2, NOT s^2/2!!
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        #
        # Importantly, we have corrected the spectra involving s2 to
        # make the definition of bs equivalent in the EPT and LPT
        # expansions.
        # The power spectra provided below are (kind -> Pk):
        # mm   -> <1*1> (returns pnl)
        # md1  -> <1*d> (returns pnl)
        # md2  -> <1*d^2/2>
        # ms2  -> <1*s^2/2>
        # mk2  -> k^2 <1*1> (with <1*1> as pnl)
        # d1d1 -> <d*d> (returns pnl)
        # d1d2 -> <d*d^2/2>
        # d1s2 -> <d*s^2/2>
        # d1k2 -> k^2 <1*d> (with <1*d> as pnl)
        # d2d2 -> <d^2/2*d^2/2>
        # d2s2 -> <d^2/2*s^2/2>
        # d2k2 -> k^2 <1*d^2/2>, not provided
        # s2s2 -> <s^2/2*s^2/2>
        # s2k2 -> k^2 <1*s^2/2>, not provided
        # k2k2 -> k^4 <1*1>, not provided

        if kind in self.pk2d_computed:
            return self.pk2d_computed[kind]

        if kind in ['mm', 'md1', 'd1d1']:
            return pnl

        if kind in ['mk2', 'd1k2']:
            pk = np.array([pnl.eval(self.ks, a, cosmo)*self.ks**2
                           for a in self.a_s])
            pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks),
                            pk_arr=pk, is_logp=False)
            self.pk2d_computed['mk2'] = pk2d
            self.pk2d_computed['d1k2'] = pk2d
            return pk2d

        inds = {'mm': 1,
                'md1': 2,
                'md2': 4,
                'ms2': 7,
                'd1d1': 3,
                'd1d2': 5,
                'd1s2': 8,
                'd2d2': 6,
                'd2s2': 9,
                's2s2': 10}
        filt = {'mm': 1.,
                'md1': 1.,
                'md2': 1.,
                'ms2': 1.,
                'd1d1': 1.,
                'd1d2': 1.,
                'd1s2': 1.,
                'd2d2': self.wk_low[None, :],
                'd2s2': self.wk_low[None, :],
                's2s2': self.wk_low[None, :]}
        pfac = {'mm': 1.0,
                'md1': 0.5, # x-corr
                'md2': 0.5, # x-corr
                'ms2': 0.25, # x-corr, s^2
                'd1d1': 1.0,
                'd1d2': 0.5, # x-corr
                'd1s2': 0.25, # x-corr, s^2
                'd2d2': 1.0,
                'd2s2': 0.25, # x-corr, s^2
                's2s2': 0.25} # s^2, s^2

        if kind not in inds:
            return alt

        pk = pfac[kind]*self.lpt_table[:, :, inds[kind]]*filt[kind]
        pk2d = ccl.Pk2D(a_arr=self.a_s, lk_arr=np.log(self.ks),
                        pk_arr=pk, is_logp=False)
        self.pk2d_computed[kind] = pk2d
        return pk2d
